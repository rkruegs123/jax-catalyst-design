import pdb
import functools
import unittest
from tqdm import tqdm

from jax import jit, random, vmap, lax

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition
from jax_md import rigid_body
from jax_md import dataclasses
from jax_md import util

from catalyst.checkpoint import checkpoint_scan
from catalyst.icosahedron_cargo.complex_getter import ComplexInfo, PENTAPOD_LEGS, BASE_LEGS
from catalyst.icosahedron_cargo.shell_getter import ShellInfo
from catalyst.icosahedron_cargo.utils import get_body_frame_positions, traj_to_pos_file

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


    """
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.467912900697836,
        "morse_shell_center_spider_head_alpha": 1.2654897136989913,
        "spider_base_particle_radius": 0.5328196552783585,
        "spider_base_radius": 4.965124458025015,
        "spider_head_height": 4.764709630665588,
        "spider_head_particle_radius": 0.1828697409842395,
    }
    """

    sim_params = {
        "log_morse_shell_center_spider_head_eps": 8.933010009519583,
        "morse_shell_center_spider_head_alpha": 1.7910141878127712,
        "spider_base_particle_radius": 0.6394784533750115,
        "spider_base_radius": 4.787588005279043,
        "spider_head_height": 5.331054263095673,
        "spider_head_particle_radius": 0.17076660748048453,
        'morse_r_onset': 9.803544376687315,
        'morse_r_cutoff': 11.742239951761889
    }

    """
    sim_params = {
        # catalyst shape
        'spider_base_radius': 4.939,
        'spider_head_height': 5.09,
        'spider_base_particle_radius': 0.5619,
        'spider_head_particle_radius': 0.404,

        # catalyst energy
        'log_morse_shell_center_spider_head_eps': 6.9, # ln(10000.0)
        'morse_shell_center_spider_head_alpha': 1.637,
        'morse_r_onset': 9.987,
        'morse_r_cutoff': 11.93
    }
    """

    def test_energy_fn(self):

        displacement_fn, shift_fn = space.free()

        # both-stable-shell-loss-stiffness50 Iteration 99
        spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])

        complex_info = ComplexInfo(
            initial_separation_coeff=0.0, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=self.sim_params["spider_base_radius"],
            spider_head_height=self.sim_params["spider_head_height"],
            spider_base_particle_radius=self.sim_params["spider_base_particle_radius"],
            spider_head_particle_radius=self.sim_params["spider_head_particle_radius"],
            spider_point_mass=1.0, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs, spider_leg_radius=1.0
        )
        energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_head_eps=jnp.exp(self.sim_params["log_morse_shell_center_spider_head_eps"]),
            morse_shell_center_spider_head_alpha=self.sim_params["morse_shell_center_spider_head_alpha"]
        )

        n_steps = 2000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(0)
        fin_state, traj = simulation(
            complex_info, energy_fn, num_steps=n_steps,
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

        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        traj = traj[vis_traj_idxs]

        traj_to_pos_file(traj, complex_info, "traj.pos", box_size=30.0)

    def _test_simulate_shell(self):

        displacement_fn, shift_fn = space.free()

        shell_info = ShellInfo(displacement_fn=displacement_fn, shift_fn=shift_fn)
        shell_energy_fn = shell_info.get_energy_fn()

        n_steps = 5000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(0)

        fin_state, traj = simulation(
            shell_info, shell_energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)


        # Write trajectory to file
        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        vis_traj = traj[vis_traj_idxs]
        traj_to_pos_file(vis_traj, shell_info, "traj_shell.pos", box_size=30.0)

    def _test_simulate_shell_remainder(self):

        displacement_fn, shift_fn = space.free()

        shell_info = ShellInfo(displacement_fn=displacement_fn, shift_fn=shift_fn)
        shell_info.rigid_body = shell_info.rigid_body[:-1]
        shell_energy_fn = shell_info.get_energy_fn()

        n_steps = 5000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(0)

        fin_state, traj = simulation(
            shell_info, shell_energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)


        # Write trajectory to file
        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        vis_traj = traj[vis_traj_idxs]
        traj_to_pos_file(vis_traj, shell_info, "traj_shell.pos", box_size=30.0)




if __name__ == "__main__":
    unittest.main()
