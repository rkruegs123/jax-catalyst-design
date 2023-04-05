import pdb
import numpy as onp
from functools import partial
from typing import Optional, Tuple, Dict, Callable, List, Union
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

import jax.numpy as np

import jax
from jax import random, grad, value_and_grad, remat, jacfwd
from jax import jit
from jax import vmap, lax
from jax import ops
from jax.config import config
config.update('jax_enable_x64', True)

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition, rigid_body
from jax_md import dataclasses
from jax_md import util



Array = util.Array
f64 = util.f64
f32 = util.f32
dtype = f64


def get_unminimized_icosahedron(displacement_fn, icosahedron_vertex_radius, vertex_mass=1.0, patch_mass=1e-8):
    d = vmap(displacement_fn, (0, None))

    phi = 0.5*(1 + jnp.sqrt(5))
    icosahedron = icosahedron_vertex_radius * jnp.array([
        [phi, 1.0, 0.0],
        [phi, -1.0, 0.0],
        [-1*phi, 1.0, 0.0],
        [-1*phi, -1.0, 0.0],
        [1.0, 0.0, phi],
        [-1.0, 0.0, phi],
        [1.0, 0.0, -1*phi],
        [-1.0, 0.0, -1*phi],
        [0.0, phi, 1.0],
        [0.0, -1*phi, 1.0],
        [0.0, phi, -1.0],
        [0.0, -1*phi, -1.0],
    ], dtype=dtype)

    '''
    The vertex rigid body consists of the vertex and its 5 patches, where we
    use 5 patches because each vertex has 5 nearest neighbors
    '''
    def get_vertex_rigid_body_positions():
        anchor = icosahedron[0]
        displacement, shift = space.free()
        d = vmap(displacement, (0, None))
        dists = space.distance(d(icosahedron, anchor))
        self_distance_tolerance = 1e-5
        large_mask_distance = 100.0
        dists = jnp.where(dists < self_distance_tolerance, large_mask_distance, dists) #mask the diagonal
        # We use min because the distances to the nearest neighbors are all the same (they should be 1 diameter away)
        # this line is not differentiable, but that's fine: we keep the icosahedron fixed for the optimization
        ids = jnp.where(dists==jnp.min(dists))[0] #find IDs of neighbors
        neighbors = icosahedron[ids]
        vec = d(neighbors, anchor)
        norm = jnp.linalg.norm(vec, axis=1).reshape(-1, 1)
        vec /= norm
        patch_pos = anchor - icosahedron_vertex_radius * vec
        return jnp.concatenate([jnp.array([anchor], dtype=dtype), patch_pos]) - anchor


    positions = jnp.array(get_vertex_rigid_body_positions(), dtype=dtype) # first position is vertex, rest are patches
    num_patches = positions.shape[0] - 1 #don't count vertex particle
    species = jnp.zeros(num_patches + 1, dtype=jnp.int32)
    species = species.at[1:].set(1) # first particle is the vertex, rest are patches
    # species = onp.array(species, dtype = int)
    patch_mass = jnp.ones(num_patches)*patch_mass # should be zero, but zeros cause nans in the gradient
    mass = jnp.concatenate((jnp.array([vertex_mass], dtype=dtype), patch_mass), axis=0)
    # mass = onp.array(mass)
    vertex_shape = rigid_body.point_union_shape(positions, mass).set(point_species=species)


    '''
    Orient rigid body particles (vertex + patches) within the rigid body.
    We don't orient the rotation about the z axis (where the z axis points
    toward the center of the icosahedron). We correct this by running a short
    simulation with just the icosahedron.

    We reference this stack overflow link to handle reorientation with
    quaternions:
    https://math.stackexchange.com/questions/60511/quaternion-for-an-object-that-to-point-in-a-direction
    '''
    def create_icosahedron_rigid_body():
        central_point = jnp.mean(icosahedron, axis=0) # center of the icosahedron

        new_vectors = d(icosahedron, central_point)
        norm = jnp.linalg.norm(new_vectors, axis=1).reshape(-1, 1)
        new_vectors /= norm

        # orig_vec is with respect to the zeroth point because that's the point we
        # selected as the anchor in 'get_vertex_rigid_body_positions()'
        orig_vec = displacement_fn(vertex_shape.points[0], jnp.mean(vertex_shape.points[1:], axis=0))
        orig_vec /= jnp.linalg.norm(orig_vec)
        crossed = vmap(jnp.cross, (None, 0))(orig_vec, new_vectors)
        dotted = vmap(jnp.dot, (0, None))(new_vectors, orig_vec)

        theta = jnp.arccos(dotted)
        cos_part = jnp.cos(theta/2).reshape(-1, 1)
        mult = vmap(lambda v, s: s*v, (0, 0))
        sin_part = mult(crossed, jnp.sin(theta/2))
        orientation = jnp.concatenate([cos_part, sin_part], axis=1)
        norm = jnp.linalg.norm(orientation, axis=1).reshape(-1, 1)
        orientation /= norm
        orientation = rigid_body.Quaternion(orientation)
        return rigid_body.RigidBody(icosahedron, orientation)

    icosahedron_rigid_body = create_icosahedron_rigid_body()

    return icosahedron_rigid_body, vertex_shape


def minimize_icosahedron(displacement_fn, shift_fn, initial_rigid_body, key, vertex_shape, icosahedron_vertex_radius,
                         num_steps=40000, morse_eps=10.0, morse_alpha=4.0,
                         soft_sphere_eps=10000.0, kT_high=1.0, kT_low=0.1, dt=1e-4):
    N_2 = num_steps // 2
    kTs = jnp.array([kT_high for i in range(0, N_2)] + [kT_low for i in range(N_2, num_steps)], dtype=jnp.float32).flatten()

    morse_eps_mat = morse_eps * jnp.array([[0.0, 0.0],
                                           [0.0, 1.0]]) #only patches attract
    soft_sphere_eps_mat = soft_sphere_eps * jnp.array([[1.0, 0.0],
                                                       [0.0, 0.0]]) #only centers repel

    pair_energy_soft = energy.soft_sphere_pair(displacement_fn, species=2, sigma=icosahedron_vertex_radius*2, epsilon=soft_sphere_eps_mat)
    pair_energy_morse = energy.morse_pair(displacement_fn, species=2, sigma=0.0, epsilon=morse_eps_mat, alpha=morse_alpha)
    pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
    energy_fn = rigid_body.point_energy(pair_energy_fn, vertex_shape)

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kTs[0])
    step_fn = jit(step_fn)
    state = init_fn(key, initial_rigid_body, mass=vertex_shape.mass())

    do_step = lambda state, t: (step_fn(state, kT=kTs[t]), state.position)
    do_step = jit(do_step)

    state, traj = lax.scan(do_step, state, jnp.arange(num_steps))
    return state.position, traj


def get_icosahedron(key, displacement_fn, shift_fn, icosahedron_vertex_radius,
                    vertex_mass=1.0, patch_mass=1e-8, obj_dir="obj/"):

    obj_dir = Path(obj_dir)
    icosahedron_rb_path = obj_dir / "icosahedron_rb.pkl"
    vertex_shape_path = obj_dir / "vertex_shape.pkl"
    if icosahedron_rb_path.exists() and vertex_shape_path.exists():
        print(f"Loading minimized icosahedron rigid body and vertex shape from data directory: {obj_dir}")
        icosahedron_rigid_body = pickle.load(open(icosahedron_rb_path, "rb"))
        vertex_shape = pickle.load(open(vertex_shape_path, "rb"))
        return icosahedron_rigid_body, vertex_shape, None

    icosahedron_rigid_body_unminimized, vertex_shape = get_unminimized_icosahedron(displacement_fn, icosahedron_vertex_radius,
                                                            vertex_mass=vertex_mass, patch_mass=patch_mass)

    icosahedron_rigid_body, minimization_traj = minimize_icosahedron(displacement_fn, shift_fn, icosahedron_rigid_body_unminimized,
                                                                     key, vertex_shape, icosahedron_vertex_radius)

    # Save the stuff
    print(f"Saving minimized icosahedron rigid body and vertex shape to data directory: {obj_dir}")
    pickle.dump(icosahedron_rigid_body, open(icosahedron_rb_path, "wb"))
    pickle.dump(vertex_shape, open(vertex_shape_path, "wb"))

    return icosahedron_rigid_body, vertex_shape, minimization_traj



def get_spider_positions(base_radius, head_height, z=0.0):
    spider_pos = jnp.zeros((5, 3))

    def scan_fn(spider_pos, i):
        x = base_radius * jnp.cos(i * 2 * jnp.pi / 5)
        y = base_radius * jnp.sin(i * 2 * jnp.pi / 5)
        spider_pos = spider_pos.at[i, :].set(jnp.array([x, y, z]))
        # spider_pos[i] = onp.array([x, y, z])
        return spider_pos, i

    spider_base, _ = lax.scan(scan_fn, spider_pos, jnp.arange(len(spider_pos)))
    spider_head = jnp.array([[0., 0., -1 * (head_height + z)]], dtype=dtype)

    return jnp.array(jnp.concatenate([spider_base, spider_head]), dtype=dtype)


SHELL_VERTEX_RADIUS = 2.0

SPIDER_BASE_RADIUS = 5.0
SPIDER_HEAD_HEIGHT = 4.0
SPIDER_LEG_DIAMETER = 1.0
SPIDER_HEAD_DIAMETER = 2.0

VERTEX_TO_BIND = 5


displacement_fn, shift_fn = space.free()
d = vmap(displacement_fn, (0, None))
d_prod = space.map_product(displacement_fn)

_key = random.PRNGKey(0)
SHELL_RB, SHELL_VERTEX_SHAPE, _ = get_icosahedron(_key, displacement_fn, shift_fn, SHELL_VERTEX_RADIUS)

# Global variables for visualization
SHELL_DIAMETERS = 2*SHELL_VERTEX_RADIUS*jnp.array([[1.0] + [0.2] * 5]*12, dtype=dtype).flatten() # Note: Just for visualization. Fake patch radius.
c1 = [1, 0, 0]
c2 = [0, 1, 0]
SHELL_COLORS = jnp.array([c1 + c2*5]*12).reshape(-1, 3)
SHELL_BODY_POS = vmap(rigid_body.transform, (0, None))(SHELL_RB, SHELL_VERTEX_SHAPE).reshape(-1, 3)



parameter_ranges = {
    # catalyst shape
    'spider_base_radius': (3.0, 6.0),
    'spider_head_height': (3.0, 10.0),
    'spider_leg_diameter': (0.5, 2.5),
    'spider_head_diameter': (1.0, 4.0),

    # catalyst energy
    'morse_leg_eps': (0.1, 10.0),
    'morse_head_eps': (5.0, 10.0), # Note: don't forget to exponentiate
    'morse_leg_alpha': (1.0, 4.0),
    'morse_head_alpha': (0.1, 2.0)
}
def get_init_params(mode="fixed", key=None):
    if mode == "fixed":

        init_params = {
            # catalyst shape
            'spider_base_radius': 5.0,
            'spider_head_height': 6.0,
            'spider_leg_diameter': 1.0,
            'spider_head_diameter': 1.0,

            # catalyst energy
            'morse_leg_eps': 2.5,
            'morse_head_eps': 10000.0,
            'morse_leg_alpha': 1.0,
            'morse_head_alpha': 1.0
            # 'morse_leg_eps': 0.0,
            # 'morse_head_eps': 0.0,
            # 'morse_leg_alpha': 0.0,
            # 'morse_head_alpha': 0.0
        }
        return init_params
    elif mode == "random":

        param_keys = random.split(key, 8)

        init_params = {
            # catalyst shape
            'spider_base_radius': random.uniform(param_keys[0], minval=3.0, maxval=6.0),
            'spider_head_height': random.uniform(param_keys[1], minval=3.0, maxval=10.0),
            'spider_leg_diameter': random.uniform(param_keys[2], minval=0.5, maxval=2.5),
            'spider_head_diameter': random.uniform(param_keys[3], minval=1.0, maxval=4.0),

            # catalyst energy
            'morse_leg_eps': random.uniform(param_keys[4], minval=0.1, maxval=10.0),
            'morse_head_eps': jnp.exp(random.uniform(param_keys[5], minval=5.0, maxval=10.0)),
            'morse_leg_alpha': random.uniform(param_keys[6], minval=1.0, maxval=4.0),
            'morse_head_alpha': random.uniform(param_keys[7], minval=0.1, maxval=2.0),
        }
        return init_params
    else:
        raise NotImplementedError(f"Unrecognized mode: {mode}")

if __name__ == "__main__":
    def foo(height):
        spider_pos = get_spider_positions(5.0, height, z=0.0)
        return jnp.sum(spider_pos)
    print(grad(foo)(5.0))
