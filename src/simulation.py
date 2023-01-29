import pdb
import numpy as onp
import functools
from typing import Optional, Tuple, Dict, Callable, List, Union
import matplotlib.pyplot as plt

import jax.numpy as np

from jax import random, grad, value_and_grad, remat, jacfwd
from jax import jit
from jax import vmap, lax
from jax import ops
from jax.config import config
config.update('jax_enable_x64', True)

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition # , rigid_body
from jax_md import dataclasses
from jax_md import util

import mod_rigid_body as rigid_body

import common
from common import SHELL_VERTEX_RADIUS, SPIDER_BASE_RADIUS, SPIDER_HEAD_HEIGHT, \
    SPIDER_LEG_DIAMETER, SPIDER_HEAD_DIAMETER, VERTEX_TO_BIND, SHELL_RB, SHELL_VERTEX_SHAPE, \
    SHELL_DIAMETERS, SHELL_COLORS, SHELL_BODY_POS
from common import displacement_fn, shift_fn, d
from common import get_spider_positions
from checkpoint import checkpoint_scan


checkpoint_every = 1
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


spider_base_species = jnp.max(SHELL_VERTEX_SHAPE.point_species) + 1
spider_species = jnp.array([[spider_base_species] * 5 + [spider_base_species + 1]]).flatten()

# note: we need mass_err to avoid nans
def initialize_system(base_radius, head_height, leg_diameter,
                      initial_separation_coeff=1.1,
                      #spider_point_masses=1.0, mass_err=1e-6):
                      spider_point_masses=jnp.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06])):

    spider_points = get_spider_positions(base_radius, head_height)

    # Make spider rigid body
    vertex = SHELL_RB[VERTEX_TO_BIND]
    disp_vector = displacement_fn(vertex.center, jnp.mean(SHELL_RB.center, axis=0))
    disp_vector /= jnp.linalg.norm(disp_vector)
    center = vertex.center + disp_vector * (SHELL_VERTEX_RADIUS + leg_diameter / 2) * initial_separation_coeff # shift away from vertex
    spider_rb = rigid_body.RigidBody(center=jnp.array([center]),
                                     orientation=rigid_body.Quaternion(jnp.array([vertex.orientation.vec])))
    # Make spider rigid body shape
    # masses = jnp.full(spider_points.shape[0], spider_point_masses)
    # masses = jnp.ones(spider_points.shape[0]) * spider_point_masses + jnp.arange(spider_points.shape[0]) * mass_err
    masses = spider_point_masses
    # masses = jnp.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06])
    spider_shape = rigid_body.point_union_shape(spider_points, masses).set(point_species=spider_species)


    # Combine spider and icosahedron to initialize
    both_shapes = rigid_body.concatenate_shapes(SHELL_VERTEX_SHAPE, spider_shape)
    joint_center = jnp.concatenate([SHELL_RB.center, spider_rb.center])
    joint_orientation = rigid_body.Quaternion(jnp.concatenate([SHELL_RB.orientation.vec, spider_rb.orientation.vec]))
    initial_rigid_body = rigid_body.RigidBody(joint_center, joint_orientation)

    return initial_rigid_body, both_shapes, spider_rb, spider_shape


# radii = [leg_radius + head_radius]

shape_species = onp.array(list(onp.zeros(12)) + [1], dtype=int).flatten()
n_point_species = 4
# shape=both_shapes
def run_dynamics_helper(initial_rigid_body, shape,
                 icosahedron_vertex_radius, spider_leg_diameter, spider_head_diameter, key,
                 morse_ii_eps=10.0, morse_leg_eps=2.0, morse_head_eps=200.0,
                 morse_ii_alpha=5.0, morse_leg_alpha=2.0, morse_head_alpha=5.0,
                 soft_eps=10000.0, kT=1.0, dt=1e-4,
                 # num_inner_steps=100, num_outer_steps=100
                 num_steps=100
):

    spider_radii = jnp.array([spider_leg_diameter, spider_head_diameter]) * 0.5
    # the two shape species are (1) the patchy particles that make up the icosahedron
    # and (2) the catalyst. There are 12 patchy particles that make up the icosahedron
    # plus 1 catalyst rigid body, making a total of 13 bodies in the simulation
    # shape_species = onp.array(list(onp.zeros(initial_rigid_body.center.shape[0]-1)) + [1], dtype=int).flatten()

    # n_point_species = int(jnp.max(shape.point_species)+1) #starts at 0, depends on point_species being ordered
    zero_interaction = jnp.zeros((n_point_species, n_point_species))

    morse_eps = zero_interaction.at[1, 1].set(morse_ii_eps) #icosahedral patches attract eachother
    morse_eps = morse_eps.at[1, 2:-1].set(morse_leg_eps) # catalyst legts attract icosahedron patches
    morse_eps = morse_eps.at[2:-1, 1].set(morse_leg_eps) #symmetry
    morse_eps = morse_eps.at[1, -1].set(morse_head_eps)
    morse_eps = morse_eps.at[-1, 1].set(morse_head_eps)

    morse_alpha = zero_interaction.at[1, 1].set(morse_ii_alpha)
    morse_alpha = morse_alpha.at[1, 2:-1].set(morse_leg_alpha)
    morse_alpha = morse_alpha.at[2:-1, 1].set(morse_leg_alpha)
    morse_alpha = morse_alpha.at[1, -1].set(morse_head_alpha)
    morse_alpha = morse_alpha.at[-1, 1].set(morse_head_alpha)

    soft_sphere_eps = zero_interaction.at[0, 0].set(soft_eps) # icosahedral centers repel each other
    soft_sphere_eps = soft_sphere_eps.at[0, 2:].set(soft_eps) # icosahedral centers repel catalyst centers
    # soft_sphere_eps = soft_sphere_eps.at[0, 2:].set(0.0) # icosahedral centers repel catalyst centers
    soft_sphere_eps = soft_sphere_eps.at[2:, 0].set(soft_eps) # symmetry
    # soft_sphere_eps = soft_sphere_eps.at[2:, 0].set(0.0) # symmetry

    soft_sphere_sigma = zero_interaction.at[0, 0].set(icosahedron_vertex_radius*2)
    soft_sphere_sigma = soft_sphere_sigma.at[0, 2:].set(icosahedron_vertex_radius + spider_radii) #icosahedral centers repel catalyst centers
    soft_sphere_sigma = soft_sphere_sigma.at[2:, 0].set(icosahedron_vertex_radius + spider_radii)
    soft_sphere_sigma = jnp.where(soft_sphere_sigma==0.0, 1e-5, soft_sphere_sigma) #avoids nans


    pair_energy_soft = energy.soft_sphere_pair(displacement_fn, species=n_point_species, sigma=soft_sphere_sigma, epsilon=soft_sphere_eps)
    pair_energy_morse = energy.morse_pair(displacement_fn, species=n_point_species, sigma=0.0, epsilon=morse_eps, alpha=morse_alpha)
    pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
    energy_fn = rigid_body.point_energy(pair_energy_fn, shape, shape_species)
    # energy_fn = point_energy(pair_energy_fn, shape, shape_species) # use our, very special `point_energy`

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
    step_fn = jit(step_fn)
    mass = shape.mass(shape_species)
    state = init_fn(key, initial_rigid_body, mass=mass)

    # do_step = lambda state, t: (step_fn(state), 0.)#state.position) #uncomment to return trajectory
    do_step = lambda state, t: (step_fn(state), state.position)
    do_step = jit(do_step)

    """
    @remat
    def outer_step(state, t):
       state, _ = lax.scan(do_step, state, jnp.arange(num_inner_steps))
       return state, t
    """

    # state, traj = lax.scan(outer_step, state, jnp.arange(num_outer_steps))
    state, traj = scan(do_step, state, jnp.arange(num_steps))
    return state.position, traj


def run_dynamics(initial_rigid_body, shape,
                 icosahedron_vertex_radius, spider_leg_diameter, spider_head_diameter, key,
                 morse_ii_eps=10.0, morse_leg_eps=2.0, morse_head_eps=200.0,
                 morse_ii_alpha=5.0, morse_leg_alpha=2.0, morse_head_alpha=5.0,
                 soft_eps=10000.0, kT=1.0, dt=1e-4,
                 # num_inner_steps=100, num_outer_steps=100
                 num_steps=100
):
    state, _ = run_dynamics_helper(
        initial_rigid_body, shape,
        icosahedron_vertex_radius, spider_leg_diameter, spider_head_diameter, key,
        morse_ii_eps=10.0, morse_leg_eps=2.0, morse_head_eps=200.0,
        morse_ii_alpha=5.0, morse_leg_alpha=2.0, morse_head_alpha=5.0,
        soft_eps=10000.0, kT=1.0, dt=1e-4,
        # num_inner_steps=100, num_outer_steps=100
        num_steps=100)
    return state

"""
Preliminary loss function: maximizing the distance from VERTEX_TO_BIND to the rest
of the icosahedron
"""
# vertex_mask = jnp.where(jnp.arange(12) == VERTEX_TO_BIND, 0, 1)
def loss_fn(body):
    # body is of length 13 -- first 12 for shell, last 1 is catalyst
    shell_body = body[:-1]
    disps = d(shell_body.center, body[VERTEX_TO_BIND].center)
    dists = space.distance(disps)
    return -jnp.sum(dists) / (shell_body.center.shape[0] - 1)

if __name__ == "__main__":
    pass
