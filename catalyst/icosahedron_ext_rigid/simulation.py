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

import catalyst.icosahedron_ext_rigid.rigid_body as rigid_body
from catalyst.checkpoint import checkpoint_scan
from catalyst.icosahedron_ext_rigid.complex import Complex, PENTAPOD_LEGS, BASE_LEGS
from catalyst.icosahedron_ext_rigid.shell import Shell
from catalyst.icosahedron_ext_rigid.utils import get_body_frame_positions, traj_to_pos_file

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
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': 12.0,
        'spider_base_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,
        'spider_attr_particle_pos_norm': 0.2,
        'spider_attr_site_radius': 0.3,

        # catalyst energy
        'log_morse_attr_eps': 6.0,
        'morse_attr_alpha': 1.0,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0
    }
    """
    sim_params = {
        "log_morse_attr_eps": 7.827934784259134,
        # "log_morse_attr_eps": 7.4,
        "morse_attr_alpha": 1.21726197904724,
        "morse_r_cutoff": 11.10868582814947,
        "morse_r_onset": 8.456814325421334,
        "spider_attr_particle_pos_norm": 0.7058241812815469,
        "spider_attr_site_radius": 0.10350310125699402,
        "spider_base_particle_radius": 0.47970176763955025,
        "spider_base_radius": 5.190019721103641,
        "spider_head_height": 10.22536600175864,
        "spider_head_particle_radius": 0.2940990010043857
    }

    def test_simulate_complex(self):

        displacement_fn, shift_fn = space.free()

        spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])

        complex_info = Complex(
            initial_separation_coeff=0.1, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=self.sim_params["spider_base_radius"],
            spider_head_height=self.sim_params["spider_head_height"],
            spider_base_particle_radius=self.sim_params["spider_base_particle_radius"],
            spider_head_particle_radius=self.sim_params["spider_head_particle_radius"],
            spider_attr_particle_pos_norm=jnp.clip(self.sim_params['spider_attr_particle_pos_norm'], 0.0, 1.0),
            spider_attr_site_radius=self.sim_params['spider_attr_site_radius'],
            spider_point_mass=1.0, spider_mass_err=1e-6,
            # spider_point_mass=0.5, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs, spider_leg_radius=0.25
        )
        energy_fn = complex_info.get_energy_fn(
            morse_attr_eps=jnp.exp(self.sim_params["log_morse_attr_eps"]),
            morse_attr_alpha=self.sim_params["morse_attr_alpha"]
        )

        n_steps = 25000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(0)
        fin_state, traj = simulation(
            complex_info, energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)

        # Write trajectory to file

        complex_lines, _, _, _ = complex_info.body_to_injavis_lines(traj[0], box_size=30.0)
        with open('another_state.pos', 'w+') as of:
            of.write('\n'.join(complex_lines))

        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        traj = traj[vis_traj_idxs]

        traj_to_pos_file(traj, complex_info, "traj.pos", box_size=30.0)




if __name__ == "__main__":
    unittest.main()
