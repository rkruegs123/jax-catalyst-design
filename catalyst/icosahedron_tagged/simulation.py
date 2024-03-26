import pdb
import functools
import unittest
from tqdm import tqdm
import time

from jax import jit, random, vmap, lax, grad, value_and_grad

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition
# from jax_md import rigid_body
from jax_md import dataclasses
from jax_md import util

import catalyst.icosahedron_tagged.rigid_body as rigid_body
from catalyst.checkpoint import checkpoint_scan
from catalyst.icosahedron_tagged.shell import Shell
from catalyst.icosahedron_tagged.complex import Complex
from catalyst.icosahedron_tagged.utils import get_body_frame_positions
from catalyst.icosahedron_tagged import utils

from jax.config import config
config.update('jax_enable_x64', True)


checkpoint_every = None
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def simulation(complex_, complex_energy_fn, num_steps,
               gamma, kT, shift_fn, dt, key):

    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
    init_fn, step_fn = simulate.nvt_langevin(complex_energy_fn, shift_fn, dt,
                                             kT, gamma=gamma_rb)
    step_fn = jit(step_fn)

    mass = complex_.shape.mass(complex_.shape_species)
    state = init_fn(key, complex_.rigid_body, mass=mass)

    do_step = lambda state, t: (step_fn(state), state.position)
    do_step = jit(do_step)

    state, traj = scan(do_step, state, jnp.arange(num_steps))
    return state.position, traj


def test_sim(complex_, complex_energy_fn, num_steps,
             gamma, kT, shift_fn, dt, key):

    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
    init_fn, step_fn = simulate.nvt_langevin(complex_energy_fn, shift_fn, dt,
                                             kT, gamma=gamma_rb)
    step_fn = jit(step_fn)

    mass = complex_.shape.mass(complex_.shape_species)
    state = init_fn(key, complex_.rigid_body, mass=mass)
    return complex_energy_fn(state.position), state.position

    # return complex_energy_fn(complex_.rigid_body), complex_.rigid_body


class TestSimulate(unittest.TestCase):

    def test_grad(self):

        displacement_fn, shift_fn = space.free()
        initial_separation_coefficient = 5.5
        vertex_to_bind_idx = 5
        dt = 1e-3
        kT = 1.0
        gamma = 10.0
        n_steps = 100
        init_log_head_eps = 6.0
        init_alpha = 1.0
        def loss_fn(params, key):
            complex_ = Complex(
                initial_separation_coeff=initial_separation_coefficient,
                vertex_to_bind_idx=vertex_to_bind_idx,
                displacement_fn=displacement_fn, shift_fn=shift_fn,
                spider_base_radius=params['spider_base_radius'],
                spider_head_height=params['spider_head_height'],
                spider_base_particle_radius=params['spider_base_particle_radius'],
                spider_attr_particle_radius=params['spider_attr_particle_radius'],
                spider_head_particle_radius=params['spider_head_particle_radius'],
                spider_point_mass=1.0, spider_mass_err=1e-6
            )

            complex_energy_fn = complex_.get_energy_fn(
                morse_attr_eps=params['log_morse_attr_eps'],
                morse_attr_alpha=params['morse_attr_alpha'],
                morse_r_onset=params['morse_r_onset'],
                morse_r_cutoff=params['morse_r_cutoff'])

            """
            fin_state, traj = test_sim(complex_, complex_energy_fn,
                                       n_steps, gamma, kT, shift_fn, dt, key)
            return fin_state.sum(), traj
            """

            fin_state, traj = simulation(complex_, complex_energy_fn,
                                         n_steps, gamma, kT, shift_fn, dt, key)
            return traj[-1].center.sum(), traj
            

        params = {
            # catalyst shape
            'spider_base_radius': 5.0,
            'spider_head_height': 5.0,
            'spider_base_particle_radius': 0.5,
            'spider_attr_particle_radius': 0.5,
            'spider_head_particle_radius': 0.5,

            # catalyst energy
            'log_morse_attr_eps': init_log_head_eps,
            'morse_attr_alpha': init_alpha,
            'morse_r_onset': 10.0,
            'morse_r_cutoff': 12.0
        }
        test_complex_ = Complex(
            initial_separation_coeff=initial_separation_coefficient,
            vertex_to_bind_idx=vertex_to_bind_idx,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=params['spider_base_radius'],
            spider_head_height=params['spider_head_height'],
            spider_base_particle_radius=params['spider_base_particle_radius'],
            spider_attr_particle_radius=params['spider_attr_particle_radius'],
            spider_head_particle_radius=params['spider_head_particle_radius'],
            spider_point_mass=1.0, spider_mass_err=1e-6
        )
        spider_body, shell_body = test_complex_.split_body(test_complex_.rigid_body)

        bind_body_flat = test_complex_.rigid_body[test_complex_.vertex_to_bind_idx]
        combined_body_center = jnp.concatenate([bind_body_flat.center.reshape(1, -1), spider_body.center])
        combined_body_qvec = jnp.concatenate([bind_body_flat.orientation.vec.reshape(1, -1), spider_body.orientation.vec])
        combined_body = rigid_body.RigidBody(combined_body_center, rigid_body.Quaternion(combined_body_qvec))
        
        key = random.PRNGKey(0)

        loss_fn = jit(loss_fn)
        start = time.time()
        loss_val = loss_fn(params, key)
        end = time.time()
        print(f"1st loss evaluation: {end - start}")

        start = time.time()
        loss_val = loss_fn(params, key)
        end = time.time()
        print(f"2nd loss evaluation: {end - start}")

        grad_fn = value_and_grad(loss_fn, has_aux=True)
        grad_fn = jit(grad_fn)
        
        start = time.time()
        (loss, traj), grads = grad_fn(params, key)
        end = time.time()
        print(f"1st grad evaluation: {end - start}")

        start = time.time()
        (loss, traj), grads = grad_fn(params, key)
        end = time.time()
        print(f"2nd grad evaluation: {end - start}")

    def test_sim(self):

        displacement_fn, shift_fn = space.free()
        initial_separation_coefficient = 2.5
        vertex_to_bind_idx = 5
        dt = 1e-3
        kT = 1.0
        # gamma = 10
        gamma = 10
        key = random.PRNGKey(0)
        n_steps = 25000
        init_log_head_eps = 4.0
        init_alpha = 1.0

        """
        params = {
            # catalyst shape
            'spider_base_radius': 5.0,
            'spider_head_height': 5.0,
            'spider_base_particle_radius': 0.5,
            'spider_attr_particle_radius': 0.5,
            'spider_head_particle_radius': 0.5,

            # catalyst energy
            'log_morse_attr_eps': init_log_head_eps,
            'morse_attr_alpha': init_alpha,
            'morse_r_onset': 10.0,
            'morse_r_cutoff': 12.0
        }
        """

        params = {
            'log_morse_attr_eps': 4.05178597, 
            'morse_attr_alpha': 1.31493759, 
            'morse_r_cutoff': 12., 
            'morse_r_onset': 10., 
            'spider_attr_particle_radius': 1.07176809, 
            'spider_base_particle_radius': 1.03204563, 
            'spider_base_radius': 4.49771056, 
            'spider_head_height': 5.21336873, 
            'spider_head_particle_radius': 0.330227
        }

        complex_ = Complex(
            initial_separation_coeff=initial_separation_coefficient,
            vertex_to_bind_idx=vertex_to_bind_idx,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=params['spider_base_radius'],
            spider_head_height=params['spider_head_height'],
            spider_base_particle_radius=params['spider_base_particle_radius'],
            spider_attr_particle_radius=params['spider_attr_particle_radius'],
            spider_head_particle_radius=params['spider_head_particle_radius'],
            spider_point_mass=1.0, spider_mass_err=1e-6
        )

        complex_energy_fn, _ = complex_.get_energy_fn(
            morse_attr_eps=jnp.exp(params['log_morse_attr_eps']),
            morse_attr_alpha=params['morse_attr_alpha'],
            morse_r_onset=params['morse_r_onset'],
            morse_r_cutoff=params['morse_r_cutoff'])

        complex_energy_fn(complex_.rigid_body)


        fin_state, traj = simulation(complex_, complex_energy_fn,
                                     n_steps, gamma, kT, shift_fn, dt, key)

        traj_injavis_lines = list()
        n_vis_states = len(traj.center)
        box_size = 30.0
        vis_every = 50
        for i in tqdm(range(n_vis_states), desc="Generating injavis output"):
            if i % vis_every == 0:
                s = traj[i]
                traj_injavis_lines += complex_.body_to_injavis_lines(s, box_size=box_size)[0]
            
        with open("test_sim.pos", 'w+') as of:
            of.write('\n'.join(traj_injavis_lines))

        
