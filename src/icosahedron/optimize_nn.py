import pdb
import numpy as onp
from functools import partial
from typing import Optional, Tuple, Dict, Callable, List, Union
import matplotlib.pyplot as plt
import time
import datetime
from pathlib import Path
from tqdm import tqdm
import argparse

import jax.numpy as np

import jax
from jax import random, grad, value_and_grad
from jax import jit
from jax import vmap, lax
from jax import ops
from jax.tree_util import tree_map
from jax.config import config
config.update('jax_enable_x64', True)
# config.update("jax_debug_nans", True)

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition # , rigid_body
from jax_md import dataclasses
from jax_md import util

import optax

import common
from common import SHELL_VERTEX_RADIUS, dtype, get_init_params, displacement_fn
import simulation
from simulation import run_dynamics_nn, initialize_system, loss_fn



# fixme: we're not passing a key here but run_dynamics takes one

def get_eval_params_fn(nn_energy_fn,
                       soft_eps, kT, dt, num_steps,
                       morse_ii_eps, morse_ii_alpha,
                       initial_separation_coeff, gamma,
                       min_com_dist, max_com_dist, eta
):
    def eval_params(params, key):

        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
        spider_leg_diameter = params['spider_leg_diameter']
        spider_head_diameter = params['spider_head_diameter']
        nn_params = params["nn"]

        initial_rigid_body, both_shapes, _, spider_shape = initialize_system(
            spider_base_radius, spider_head_height,
            spider_leg_diameter, initial_separation_coeff=initial_separation_coeff)

        # For now, we omit the full trajectory and rely on JAX compiler.
        # If still slow, try taking its computation out of the function
        fin_state, traj = run_dynamics_nn(
            initial_rigid_body,
            nn_energy_fn, nn_params,
            spider_shape, both_shapes, key,
            SHELL_VERTEX_RADIUS,
            spider_leg_diameter, spider_head_diameter,
            morse_ii_eps=morse_ii_eps,
            morse_ii_alpha=morse_ii_alpha,
            soft_eps=soft_eps, kT=kT, dt=dt,
            num_steps=num_steps, gamma=gamma
        )
        return loss_fn(fin_state, eta, min_com_dist, max_com_dist)
    return eval_params



def train(args):

    batch_size = args['batch_size']
    n_iters = args['n_iters']
    n_steps = args['n_steps']
    data_dir = args['data_dir']
    lr = args['lr']
    init_method = args['init_method']
    key_seed = args['key_seed']
    kT = args['temperature']
    initial_separation_coefficient = args['init_separate']
    gamma = args['gamma']
    min_com_dist = args['min_com_dist']
    max_com_dist = args['max_com_dist']
    eta = args['eta']

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise RuntimeError(f"No data directory exists at location: {data_dir}")

    key = random.PRNGKey(key_seed)
    keys = random.split(key, n_iters)

    optimizer = optax.adam(lr)


    params = simulation.get_init_params_spider_shape(mode="fixed", key=None)
    initial_rigid_body, both_shapes, _, spider_shape = initialize_system(
        params["spider_base_radius"], params["spider_head_height"],
        params["spider_leg_diameter"], initial_separation_coeff=initial_separation_coefficient
    )

    init_nn_params, nn_energy_fn = simulation.get_init_params_nn(
        spider_shape, initial_rigid_body, key
    )

    params["nn"] = init_nn_params

    opt_state = optimizer.init(params)

    eval_params_fn = get_eval_params_fn(
        nn_energy_fn,
        soft_eps=100000.0, kT=kT, dt=1e-3,
        num_steps=n_steps,
        morse_ii_eps=10.0, morse_ii_alpha=5.0,
        initial_separation_coeff=initial_separation_coefficient,
        gamma=gamma,
        min_com_dist=min_com_dist, max_com_dist=max_com_dist,
        eta=eta)
    grad_eval_params_fn = jit(value_and_grad(eval_params_fn))
    batched_grad_fn = jit(vmap(grad_eval_params_fn, in_axes=(None, 0)))


    run_name = f"catalyst_nn_b{batch_size}_n{n_steps}_lr{lr}_i{init_method}_s{initial_separation_coefficient}_kT{kT}_g{gamma}_min{min_com_dist}_max{max_com_dist}_e{eta}_k{key_seed}"
    run_dir = data_dir / run_name
    print(f"Making directory: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)

    for i in tqdm(range(n_iters)):
        print(f"\nIteration: {i}")
        iter_key = keys[i]
        batch_keys = random.split(iter_key, batch_size)
        start = time.time()
        vals, grads = batched_grad_fn(params, batch_keys)
        end = time.time()

        avg_grads = tree_map(lambda x: jnp.mean(x, axis=0), grads)
        # avg_grads = {k: jnp.mean(grads[k], axis=0) for k in grads}

        updates, opt_state = optimizer.update(avg_grads, opt_state)
        params = optax.apply_updates(params, updates)


def get_argparse():
    parser = argparse.ArgumentParser(description="Simulation for spider catalyst design")

    parser.add_argument('--batch-size', type=int, default=3, help="Num. batches for each round of gradient descent")
    parser.add_argument('--n-iters', type=int, default=1, help="Num. iterations of gradient descent")
    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    parser.add_argument('--n-steps', type=int, default=5000, help="Num. steps per simulation")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for optimization")
    parser.add_argument('--init-separate', type=float, default=0.0, help="Initial separation coefficient")
    parser.add_argument('-d', '--data-dir', type=str,
                        default="data/",
                        help='Path to base data directory')
    parser.add_argument('--init-method', type=str,
                        default="random",
                        choices=['random', 'fixed'],
                        help='Method for initializing parameters')
    parser.add_argument('-kT', '--temperature', type=float, default=2.0, help="Temperature in kT")
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help="friction coefficient")
    parser.add_argument('-e', '--eta', type=float, default=2.5, help="steepness of anti-explosion wall")
    parser.add_argument('-min', '--min-com-dist', type=float, default=3.4, help="low end of anti-explosion wall")
    parser.add_argument('-max', '--max-com-dist', type=float, default=4.25, help="high end of anti-explosion wall")

    return parser

if __name__ == "__main__":
    parser = get_argparse()
    args = vars(parser.parse_args())

    print(f"Testing optimization...")
    train(args)
