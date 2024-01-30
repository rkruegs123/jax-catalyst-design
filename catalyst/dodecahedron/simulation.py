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
from catalyst.dodecahedron.complex_getter import ComplexInfo, TRIPOD_LEGS, BASE_LEGS
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
        'spider_base_radius': 2.5,
        'spider_head_height': 4.0,
        'spider_base_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,

        # catalyst energy
        # 'log_morse_shell_center_spider_head_eps': 9.21, # ln(10000.0)
        'log_morse_shell_center_spider_head_eps': 7.5,
        'morse_shell_center_spider_head_alpha': 1.5,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0
    }


    def test_simulate_complex(self):

        displacement_fn, shift_fn = space.free()

        # both-stable-shell-loss-stiffness50 Iteration 99
        spider_bond_idxs = jnp.concatenate([TRIPOD_LEGS, BASE_LEGS])

        complex_info = ComplexInfo(
            initial_separation_coeff=0.75, vertex_to_bind_idx=utils.vertex_to_bind_idx,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=self.sim_params["spider_base_radius"],
            spider_head_height=self.sim_params["spider_head_height"],
            spider_base_particle_radius=self.sim_params["spider_base_particle_radius"],
            spider_head_particle_radius=self.sim_params["spider_head_particle_radius"],
            spider_point_mass=1.0, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs, spider_leg_radius=0.5
        )
        energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_head_eps=jnp.exp(self.sim_params["log_morse_shell_center_spider_head_eps"]),
            morse_shell_center_spider_head_alpha=self.sim_params["morse_shell_center_spider_head_alpha"]
        )

        n_steps = 500
        assert(n_steps % 50 == 0)
        key = random.PRNGKey(0)
        fin_state, traj = simulation(
            complex_info, energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)


        # Write trajectory to file
        vis_traj_idxs = jnp.arange(0, n_steps+1, 50)
        traj = traj[vis_traj_idxs]

        traj_to_pos_file(traj, complex_info, "traj.pos", box_size=30.0)


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

    def test_simulate_shell_remainder(self):

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

    def test_init_energy_terms(self):
        displacement_fn, shift_fn = space.free()

        # both-stable-shell-loss-stiffness50 Iteration 99
        spider_bond_idxs = jnp.concatenate([TRIPOD_LEGS, BASE_LEGS])

        complex_info = ComplexInfo(
            initial_separation_coeff=0.75,
            # vertex_to_bind_idx=utils.vertex_to_bind_idx,
            vertex_to_bind_idx=0,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=self.sim_params["spider_base_radius"],
            spider_head_height=self.sim_params["spider_head_height"],
            spider_base_particle_radius=self.sim_params["spider_base_particle_radius"],
            spider_head_particle_radius=self.sim_params["spider_head_particle_radius"],
            spider_point_mass=1.0, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs, spider_leg_radius=0.5
        )
        energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_head_eps=jnp.exp(self.sim_params["log_morse_shell_center_spider_head_eps"]),
            morse_shell_center_spider_head_alpha=self.sim_params["morse_shell_center_spider_head_alpha"]
        )

        init_body = complex_info.rigid_body
        """
        kT = 1.0
        dt = 1e-3
        key = random.PRNGKey(0)
        gamma = 10.0
        gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma_rb)

        mass = complex_info.shape.mass(complex_info.shape_species)
        state = init_fn(key, init_body, mass=mass)
        body = state.position
        """

        soft_eps = 10000.0
        leg_energy_fn = complex_info.get_leg_energy_fn(
            soft_eps, 2 * complex_info.spider_leg_radius, leg_alpha=2.0)
        leg_energy = leg_energy_fn(init_body)

        all_lines, _, _, _ =  complex_info.body_to_injavis_lines(init_body, box_size=30.0)

        with open("init_complex_pre_init.pos", 'w+') as of:
            of.write('\n'.join(all_lines))

    def test_save_things(self):
        displacement_fn, shift_fn = space.free()

        for v_idx in range(20):
            spider_bond_idxs = jnp.concatenate([TRIPOD_LEGS, BASE_LEGS])

            complex_info = ComplexInfo(
                initial_separation_coeff=0.75,
                # vertex_to_bind_idx=utils.vertex_to_bind_idx,
                vertex_to_bind_idx=v_idx,
                displacement_fn=displacement_fn, shift_fn=shift_fn,
                spider_base_radius=self.sim_params["spider_base_radius"],
                spider_head_height=self.sim_params["spider_head_height"],
                spider_base_particle_radius=self.sim_params["spider_base_particle_radius"],
                spider_head_particle_radius=self.sim_params["spider_head_particle_radius"],
                spider_point_mass=1.0, spider_mass_err=1e-6,
                spider_bond_idxs=spider_bond_idxs, spider_leg_radius=0.5
            )
            energy_fn = complex_info.get_energy_fn(
                morse_shell_center_spider_head_eps=jnp.exp(self.sim_params["log_morse_shell_center_spider_head_eps"]),
                morse_shell_center_spider_head_alpha=self.sim_params["morse_shell_center_spider_head_alpha"]
            )

            init_body = complex_info.rigid_body
            all_lines, _, _, _ =  complex_info.body_to_injavis_lines(init_body, box_size=30.0)

            with open(f"init_complex_pre_init_v{v_idx}.pos", 'w+') as of:
                of.write('\n'.join(all_lines))





if __name__ == "__main__":
    unittest.main()
