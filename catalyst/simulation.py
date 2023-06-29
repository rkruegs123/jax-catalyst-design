import pdb
import functools
import unittest

from jax import jit, random, vmap, lax

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition, rigid_body
from jax_md import dataclasses
from jax_md import util

from catalyst.checkpoint import checkpoint_scan
from catalyst.complex_getter import ComplexInfo

from jax.config import config
config.update('jax_enable_x64', True)


checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def simulation(complex_info, complex_energy_fn, num_steps, gamma, kT, shift_fn, dt, key):

    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
    init_fn, step_fn = simulate.nvt_langevin(complex_energy_fn, shift_fn, dt, kT, gamma=gamma_rb)
    step_fn = jit(step_fn)

    mass = complex_info.shape.mass(complex_info.shape_species)
    state = init_fn(key, complex_info.rigid_body, mass=mass)

    do_step = lambda state, t: (step_fn(state), state.position)
    do_step = jit(do_step)

    state, traj = scan(do_step, state, jnp.arange(num_steps))
    return state.position, traj


class TestSimulate(unittest.TestCase):

    def test_energy_fn(self):
        displacement_fn, shift_fn = space.free()
        complex_info = ComplexInfo(
            initial_separation_coeff=0.1, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn,
            spider_base_radius=5.0, spider_head_height=4.0,
            spider_base_particle_radius=0.5, spider_head_particle_radius=-.5,
            spider_point_mass=1.0, spider_mass_err=1e-6
        )
        energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_base_eps=2.0, morse_shell_center_spider_base_alpha=2.0,
            morse_shell_center_spider_head_eps=200.0, morse_shell_center_spider_head_alpha=5.0,
        )

        key = random.PRNGKey(0)
        fin_state, traj = simulation(
            complex_info, energy_fn, num_steps=100,
            gamma=50.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)




if __name__ == "__main__":
    unittest.main()
