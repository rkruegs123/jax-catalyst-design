import pdb
import functools
import unittest
from tqdm import tqdm

from jax import jit, random, vmap, lax

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition
# from jax_md import rigid_body
from jax_md import dataclasses
from jax_md import util

import catalyst.dodecahedron.rigid_body as rigid_body
from catalyst.checkpoint import checkpoint_scan
# from catalyst.dodecahedron.complex_getter import ComplexInfo, TETRAPOD_LEGS, BASE_LEGS
from catalyst.dodecahedron.shell_getter import ShellInfo
from catalyst.dodecahedron.utils import get_body_frame_positions, traj_to_pos_file
from catalyst.dodecahedron import utils

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

    sim_params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': 5.0,
        'spider_base_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,

        # catalyst energy
        # 'log_morse_shell_center_spider_head_eps': 9.21, # ln(10000.0)
        'log_morse_shell_center_spider_head_eps': 3.5,
        'morse_shell_center_spider_head_alpha': 1.5,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0
    }




    def test_simulate_shell(self):

        displacement_fn, shift_fn = space.free()

        shell_info = ShellInfo(displacement_fn=displacement_fn, shift_fn=shift_fn)
        shell_energy_fn = shell_info.get_energy_fn()

        n_steps = 5000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(0)

        fin_state, traj = simulation(
            shell_info, shell_energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)


        mapped_displacement = vmap(displacement_fn, (0, None))
        def get_com_dists(shell_body):
            remaining_vertices = jnp.concatenate(
                [shell_body.center[:utils.vertex_to_bind_idx],
                 shell_body.center[utils.vertex_to_bind_idx+1:]],
                axis=0)

            remaining_com = jnp.mean(remaining_vertices, axis=0)
            com_dists = space.distance(mapped_displacement(remaining_vertices, remaining_com))
            return com_dists

        min_com_dist = 1e6
        max_com_dist = -1e6
        for i in tqdm(range(n_steps)):
            i_com_dists = get_com_dists(traj[i])
            min_i_com_dist = i_com_dists.min()
            if min_i_com_dist < min_com_dist:
                min_com_dist = min_i_com_dist

            max_i_com_dist = i_com_dists.max()
            if max_i_com_dist > max_com_dist:
                max_com_dist = max_i_com_dist


        pdb.set_trace()


        # Write trajectory to file
        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        vis_traj = traj[vis_traj_idxs]
        traj_to_pos_file(vis_traj, shell_info, "traj_shell.pos", box_size=30.0)



if __name__ == "__main__":
    unittest.main()
