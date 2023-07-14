import pdb
import functools
import unittest
from tqdm import tqdm

from jax import jit, random, vmap, lax

from jax_md.util import *
# from jax_md import space, smap, energy, minimize, quantity, simulate, partition, rigid_body
from jax_md import space, smap, energy, minimize, quantity, simulate, partition
from jax_md import dataclasses
from jax_md import util

import catalyst.rigid_body as rigid_body
from catalyst.checkpoint import checkpoint_scan
from catalyst.complex_getter import ComplexInfo
from catalyst.utils import get_body_frame_positions, traj_to_pos_file

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
            spider_base_radius=5.0, spider_head_height=5.0,
            spider_base_particle_radius=0.5, spider_head_particle_radius=0.5,
            spider_point_mass=1.0, spider_mass_err=1e-6
        )
        energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_base_eps=2.5, morse_shell_center_spider_base_alpha=1.0,
            morse_shell_center_spider_head_eps=jnp.exp(9.21), morse_shell_center_spider_head_alpha=1.5,
        )

        key = random.PRNGKey(0)
        fin_state, traj = simulation(
            complex_info, energy_fn, num_steps=10000,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)

        # Write final states to file -- visualize with `java -Xmx4096m -jar injavis.jar <name>.pos`

        ## Shell
        # fin_shell_rb = fin_state[:12]
        # shell_lines = complex_info.shell_info.body_to_injavis_lines(fin_shell_rb, box_size=30.0)
        # with open('shell_state.pos', 'w+') as of:
        #     of.write('\n'.join(shell_lines))

        ## Spider
        # fin_spider_rb = fin_state[-1]
        # spider_lines = complex_info.spider_info.body_to_injavis_lines(fin_spider_rb, box_size=30.0)
        # with open('spider_state.pos', 'w+') as of:
        #     of.write('\n'.join(spider_lines))

        ## Complex
        """
        complex_lines, _, _, _ = complex_info.body_to_injavis_lines(fin_state, box_size=30.0)
        with open('complex_state.pos', 'w+') as of:
            of.write('\n'.join(complex_lines))
        """

        # Write trajectory to file

        traj = traj[::1000]

        traj_to_pos_file(traj, complex_info, "traj.pos", box_size=30.0)




if __name__ == "__main__":
    unittest.main()
