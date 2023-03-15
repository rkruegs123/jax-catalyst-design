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
#from jax_md import rigid_body

import common
from common import SHELL_VERTEX_RADIUS, SPIDER_BASE_RADIUS, SPIDER_HEAD_HEIGHT, \
    SPIDER_LEG_DIAMETER, SPIDER_HEAD_DIAMETER, VERTEX_TO_BIND, SHELL_RB, SHELL_VERTEX_SHAPE, \
    SHELL_DIAMETERS, SHELL_COLORS, SHELL_BODY_POS
from common import displacement_fn, shift_fn, d, d_prod
from common import get_spider_positions
from checkpoint import checkpoint_scan
import leg


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
                      spider_point_masses=1.0, mass_err=1e-6,
                      vertex_to_bind=VERTEX_TO_BIND):
                      # spider_point_masses=jnp.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06])):

    spider_points = get_spider_positions(base_radius, head_height)

    # Make spider rigid body
    vertex = SHELL_RB[vertex_to_bind]
    disp_vector = displacement_fn(vertex.center, jnp.mean(SHELL_RB.center, axis=0))
    disp_vector /= jnp.linalg.norm(disp_vector)
    center = vertex.center + disp_vector * (SHELL_VERTEX_RADIUS + leg_diameter / 2) * initial_separation_coeff # shift away from vertex
    spider_rb = rigid_body.RigidBody(center=jnp.array([center]),
                                     orientation=rigid_body.Quaternion(jnp.array([vertex.orientation.vec])))
                                     #orientation=rigid_body.Quaternion(jnp.array([[0.0, 0.0, 1.0, 0.0]])))
    # Make spider rigid body shape
    # masses = jnp.full(spider_points.shape[0], spider_point_masses)
    masses = jnp.ones(spider_points.shape[0]) * spider_point_masses + jnp.arange(spider_points.shape[0]) * mass_err
    # masses = spider_point_masses
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


def get_energy_fn(icosahedron_vertex_radius, spider_leg_diameter, spider_head_diameter,
                  morse_ii_eps, morse_leg_eps, morse_head_eps,
                  morse_ii_alpha, morse_leg_alpha, morse_head_alpha,
                  soft_eps, shape):

    spider_radii = jnp.array([spider_leg_diameter, spider_head_diameter]) * 0.5

    zero_interaction = jnp.zeros((n_point_species, n_point_species))



    morse_eps = zero_interaction.at[1, 1].set(morse_ii_eps) #icosahedral patches attract eachother

    """
    morse_eps = morse_eps.at[1, 2:-1].set(morse_leg_eps) # catalyst legts attract icosahedron patches
    morse_eps = morse_eps.at[2:-1, 1].set(morse_leg_eps) #symmetry
    morse_eps = morse_eps.at[1, -1].set(morse_head_eps)
    morse_eps = morse_eps.at[-1, 1].set(morse_head_eps)
    """
    morse_eps = morse_eps.at[0, 2:-1].set(morse_leg_eps) # catalyst legts attract icosahedron vertices
    morse_eps = morse_eps.at[2:-1, 0].set(morse_leg_eps) #symmetry
    morse_eps = morse_eps.at[0, -1].set(morse_head_eps)
    morse_eps = morse_eps.at[-1, 0].set(morse_head_eps)


    morse_alpha = zero_interaction.at[1, 1].set(morse_ii_alpha)

    """
    morse_alpha = morse_alpha.at[1, 2:-1].set(morse_leg_alpha)
    morse_alpha = morse_alpha.at[2:-1, 1].set(morse_leg_alpha)
    morse_alpha = morse_alpha.at[1, -1].set(morse_head_alpha)
    morse_alpha = morse_alpha.at[-1, 1].set(morse_head_alpha)
    """
    morse_alpha = morse_alpha.at[0, 2:-1].set(morse_leg_alpha)
    morse_alpha = morse_alpha.at[2:-1, 0].set(morse_leg_alpha)
    morse_alpha = morse_alpha.at[0, -1].set(morse_head_alpha)
    morse_alpha = morse_alpha.at[-1, 0].set(morse_head_alpha)


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
    pair_energy_morse = energy.morse_pair(displacement_fn, species=n_point_species,
                                          sigma=0.0, epsilon=morse_eps, alpha=morse_alpha,
                                          r_onset=10.0, r_cutoff=12.0
    )
    pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
    energy_fn = rigid_body.point_energy(pair_energy_fn, shape, shape_species)

    return energy_fn




def run_dynamics_helper(initial_rigid_body, shape,
                 icosahedron_vertex_radius, spider_leg_diameter, spider_head_diameter, key,
                 morse_ii_eps=10.0, morse_leg_eps=2.0, morse_head_eps=200.0,
                 morse_ii_alpha=5.0, morse_leg_alpha=2.0, morse_head_alpha=5.0,
                 soft_eps=10000.0, kT=1.0, dt=1e-4,
                 # num_inner_steps=100, num_outer_steps=100
                 num_steps=100, gamma=0.1
):


    # Code for generating the energy function
    base_energy_fn = get_energy_fn(icosahedron_vertex_radius, spider_leg_diameter, spider_head_diameter,
                                   morse_ii_eps, morse_leg_eps, morse_head_eps,
                                   morse_ii_alpha, morse_leg_alpha, morse_head_alpha,
                                   soft_eps, shape)
    leg_energy_fn = leg.get_leg_energy_fn(soft_eps, (spider_leg_diameter/2 + SHELL_VERTEX_RADIUS), shape, shape_species) # TODO: unrestrict leg diameter
    energy_fn = lambda body: base_energy_fn(body) + leg_energy_fn(body)
    # energy_fn = lambda body: base_energy_fn(body)

    #init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
    init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma_rb)
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
                 num_steps=100, gamma=0.1
):
    state, _ = run_dynamics_helper(
        initial_rigid_body, shape,
        icosahedron_vertex_radius, spider_leg_diameter, spider_head_diameter, key,
        morse_ii_eps=morse_ii_eps, morse_leg_eps=morse_leg_eps, morse_head_eps=morse_head_eps,
        morse_ii_alpha=morse_ii_alpha, morse_leg_alpha=morse_leg_alpha, morse_head_alpha=morse_head_alpha,
        soft_eps=soft_eps, kT=kT, dt=dt,
        num_steps=num_steps, gamma=gamma)
    return state

"""
Preliminary loss function: maximizing the distance from VERTEX_TO_BIND to the rest
of the icosahedron
"""
# vertex_mask = jnp.where(jnp.arange(12) == VERTEX_TO_BIND, 0, 1)
INF = 1e6
def loss_fn_helper(body, eta, min_com_dist=3.4, max_com_dist=4.25):
    # body is of length 13 -- first 12 for shell, last 1 is catalyst
    shell_body = body[:-1]
    disps = d(shell_body.center, body[VERTEX_TO_BIND].center)
    dists = space.distance(disps)
    vertex_far_from_icos = -jnp.sum(dists)
    

    # Term that keeps the rest together
    """
    center_dists = space.distance(d_prod(shell_body.center, shell_body.center))
    centers_dists = center_dists.at[VERTEX_TO_BIND, :].set(0.0)
    centers_dists = center_dists.at[:, VERTEX_TO_BIND].set(0.0)
    term2 = jnp.sum(centers_dists) * (1/10)
    """
    remaining_vertices = jnp.concatenate(
        [shell_body.center[:VERTEX_TO_BIND], shell_body.center[VERTEX_TO_BIND+1:]],
        axis=0)
    remaining_com = jnp.mean(remaining_vertices, axis=0)
    com_dists = space.distance(d(remaining_vertices, remaining_com))
    #icos_stays_together = com_dists.sum()
    
    # mult_iso_cutoff_right = energy.multiplicative_isotropic_cutoff(lambda x: 1e6, r_onset=4.25, r_cutoff=4.4)
    # mult_iso_cutoff_left_inv = energy.multiplicative_isotropic_cutoff(lambda x: 1e6, r_onset=3.0, r_cutoff=3.4)
    mult_iso_cutoff_right = energy.multiplicative_isotropic_cutoff(lambda x: INF, r_onset=min_com_dist-eta, r_cutoff=min_com_dist)
    mult_iso_cutoff_left_inv = energy.multiplicative_isotropic_cutoff(lambda x: INF, r_onset=max_com_dist, r_cutoff=max_com_dist+eta)
    tight_range = lambda dr: mult_iso_cutoff_right(dr) + (INF - mult_iso_cutoff_left_inv(dr)) 
    icos_stays_together = jnp.sum(tight_range(com_dists))

    # Term that asks the catalyst to detach from the icosahedron
    catalyst_body = body[-1]
    dists_from_cat = space.distance(d(remaining_vertices, catalyst_body.center))
    catalyst_detaches_from_icos = -1*dists_from_cat.sum()

    norm = (shell_body.center.shape[0] - 1)

    return vertex_far_from_icos / norm, icos_stays_together / norm, catalyst_detaches_from_icos / norm

def loss_fn(body, eta, min_com_dist=3.4, max_com_dist=4.25):
    vertex_far_from_icos, icos_stays_together, catalyst_detaches_from_icos = loss_fn_helper(body, eta, min_com_dist, max_com_dist)
    # return vertex_far_from_icos + 5.0 * icos_stays_together + catalyst_detaches_from_icos
    # return vertex_far_from_icos + 2.0 * icos_stays_together
    # return vertex_far_from_icos + jnp.exp(eta * (icos_stays_together - 0.34))
    return vertex_far_from_icos + icos_stays_together

if __name__ == "__main__":

    from pprint import pprint
    import time


    base_radius = 5.0
    head_height = 3.0
    leg_diameter = 1.0
    initial_separation_coeff_close = 0.5
    initial_separation_coeff_far = 2.0

    initial_rigid_body_far, both_shapes, spider_rb, spider_shape = initialize_system(
        base_radius, head_height, leg_diameter,
        initial_separation_coeff=initial_separation_coeff_far,
        spider_point_masses=1.0, mass_err=1e-6,
        vertex_to_bind=VERTEX_TO_BIND)

    initial_rigid_body_close, _, _, _ = initialize_system(
        base_radius, head_height, leg_diameter,
        initial_separation_coeff=initial_separation_coeff_close,
        spider_point_masses=1.0, mass_err=1e-6,
        vertex_to_bind=VERTEX_TO_BIND)

    head_diameter = 1.0


    energy_params = {
        "icosahedron_vertex_radius": SHELL_VERTEX_RADIUS,
        "spider_leg_diameter": leg_diameter,
        "spider_head_diameter": head_diameter,
        "morse_ii_eps": 10.0,
        "morse_leg_eps": 10.0,
        "morse_head_eps": 100000.0,
        "morse_ii_alpha": 5.0,
        "morse_leg_alpha": 1.0,
        "morse_head_alpha": 1.0,
        "soft_eps": 1000.0,
    }

    init_params = {
        "base_radius": base_radius,
        "head_height": head_height,
        # "leg_diameter": leg_diameter,
        "initial_separation_coeff": initial_separation_coeff_close
    }
    params = {"energy": energy_params, "init": init_params}


    """
    energy_fn = get_energy_fn(icosahedron_vertex_radius=SHELL_VERTEX_RADIUS,
                              spider_leg_diameter=leg_diameter, spider_head_diameter=head_diameter,
                              morse_ii_eps=10.0, morse_leg_eps=10.0, morse_head_eps=100000.0,
                              morse_ii_alpha=5.0, morse_leg_alpha=1.0, morse_head_alpha=1.0,
                              soft_eps=1000.0, shape=both_shapes)
    """
    # energy_fn = get_energy_fn(**params)


    def eval_params_init(params):
        energy_params = params['energy']
        init_params = params['init']
        init_params["leg_diameter"] = energy_params["spider_leg_diameter"]

        initial_rigid_body_close, both_shapes, spider_rb, spider_shape = initialize_system(
            **init_params
        )

        energy_fn = get_energy_fn(shape=both_shapes, **energy_params)
        # far_val = energy_fn(initial_rigid_body_far)
        close_val = energy_fn(initial_rigid_body_close)
        return close_val
    # close_val = energy_fn(initial_rigid_body_close)
    # eval_params_init_grad = grad(eval_params_init)
    # the_thing = eval_params_init_grad(params)



    sim_params = {
        "spider_leg_diameter": leg_diameter,
        "spider_head_diameter": head_diameter,
        "morse_ii_eps": 100.0,
        "morse_leg_eps": 1.0,
        "morse_head_eps": 1.0,
        "morse_ii_alpha": 5.0,
        "morse_leg_alpha": 2.0,
        "morse_head_alpha": 2.0,
        "soft_eps": 1000.0,
    }

    params = {'init': init_params, 'sim': sim_params}

    key = random.PRNGKey(0)
    def eval_params_sim(params):
        sim_params = params['sim']
        init_params = params['init']
        init_params["leg_diameter"] = sim_params["spider_leg_diameter"]

        initial_rigid_body_close, both_shapes, spider_rb, spider_shape = initialize_system(
            **init_params
        )


        fin_state = run_dynamics(initial_rigid_body_close, both_shapes,
                                 SHELL_VERTEX_RADIUS, key=key, num_steps=1000,
                                 **sim_params
        )

        energy_fn = get_energy_fn(SHELL_VERTEX_RADIUS, shape=both_shapes, **sim_params)
        # far_val = energy_fn(initial_rigid_body_far)
        # fin_val = energy_fn(fin_state)
        fin_val = loss_fn(fin_state)
        return fin_val
    start = time.time()
    eval_params_sim_grad = value_and_grad(eval_params_sim)
    end = time.time()
    the_other_thing_val, the_other_thing_grad = eval_params_sim_grad(params)
    pdb.set_trace()
    pprint(the_other_thing_grad)
    print(the_other_thing_val)
    print(f"Total time: {onp.round(end - start, 2)}")

    pdb.set_trace()
