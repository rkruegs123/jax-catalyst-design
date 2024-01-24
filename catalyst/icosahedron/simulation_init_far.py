import pdb
import functools
import unittest
from tqdm import tqdm
import argparse
from pathlib import Path

from jax import jit, random, vmap, lax
from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition
# from jax_md import rigid_body
from jax_md import dataclasses
from jax_md import util

import catalyst.icosahedron.rigid_body as rigid_body
from catalyst.checkpoint import checkpoint_scan
from catalyst.icosahedron.complex_getter import ComplexInfo, PENTAPOD_LEGS, BASE_LEGS
from catalyst.icosahedron.shell_getter import ShellInfo
from catalyst.icosahedron.utils import get_body_frame_positions, traj_to_pos_file
from catalyst.icosahedron.loss import get_loss_fn
from catalyst.icosahedron.simulation import simulation

from jax.config import config
config.update('jax_enable_x64', True)


def run(args):
    opt_params_type = args['opt_params_type']

    if opt_params_type == "hi":
        # Note: Iteration 1999 of `abduction-limit-min-head-no-stable-shell`
        sim_params = {
            "log_morse_shell_center_spider_head_eps": 9.172761093104889,
            "morse_r_cutoff": 11.572416127114405,
            "morse_r_onset": 9.686903030834102,
            "morse_shell_center_spider_head_alpha": 1.849240368553664,
            "spider_base_particle_radius": 0.8557382945137544,
            "spider_base_radius": 4.33392895531777,
            "spider_head_height": 5.5094075448189015,
            "spider_head_particle_radius": 0.09881308349103084,
        }
    else:
        # Note: Iteration 1999 of `diffusive-limit-min-head-no-stable-shell`
        sim_params = {
            "log_morse_shell_center_spider_head_eps": 9.072351997659718,
            "morse_r_cutoff": 7.005022426371568,
            "morse_r_onset": 5.804326499350028,
            "morse_shell_center_spider_head_alpha": 0.7093345904237273,
            "spider_base_particle_radius": 1.0347066737817279,
            "spider_base_radius": 4.3999652877275555,
            "spider_head_height": 6.23877507045952,
            "spider_head_particle_radius": 0.02815188196466319,
        }

    data_dir = Path(args['data_dir'])
    run_name = args['run_name']
    init_sep_coeff = args['init_sep_coeff']
    run_dir = output_basedir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    displacement_fn, shift_fn = space.free()
    spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])

    complex_info = ComplexInfo(
        initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=5,
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

    n_steps = 20000
    assert(n_steps % 100 == 0)
    key = random.PRNGKey(0)
    fin_state, traj = simulation(
        complex_info, energy_fn, num_steps=n_steps,
        gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)

    # Write trajectory to file
    vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
    traj = traj[vis_traj_idxs]

    traj_to_pos_file(traj, complex_info, run_dir / "traj.pos", box_size=30.0)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Running optimized parameters at different distances")
    parser.add_argument('--data-dir', type=str, help='Output base directory',
                        default="data/icosahedron/"
    )
    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--init-sep-coeff', type=float, default=0.0,
                        help="Initial separation coefficient")

    parser.add_argument('--opt-params-type', type=str,
                        choices=["lo", "hi"],
                        help="Type of optimized parameters to run")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
