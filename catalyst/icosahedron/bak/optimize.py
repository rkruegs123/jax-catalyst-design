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
from jax import random, grad, value_and_grad, remat, jacfwd
from jax import jit
from jax import vmap, lax
from jax import ops
from jax.config import config
config.update('jax_enable_x64', True)
# config.update("jax_debug_nans", True)

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition # , rigid_body
from jax_md import dataclasses
from jax_md import util

import optax

import common
from common import SHELL_VERTEX_RADIUS, dtype, get_init_params
import simulation
from simulation import run_dynamics, initialize_system, loss_fn



# fixme: we're not passing a key here but run_dynamics takes one

def get_eval_params_fn(soft_eps, kT, dt,
                       # num_inner_steps, num_outer_steps,
                       num_steps,
                       morse_ii_eps, morse_ii_alpha,
                       initial_separation_coeff, gamma,
                       min_com_dist, max_com_dist,
                       eta

):
    def eval_params(params, key):

        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
        # spider_leg_diameter = 1.5 #params['spider_leg_diameter']
        spider_leg_diameter = params['spider_leg_diameter']
        spider_head_diameter = params['spider_head_diameter']

        morse_leg_eps = params['morse_leg_eps']
        log_morse_head_eps = params['log_morse_head_eps']
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params['morse_leg_alpha']
        morse_head_alpha = params['morse_head_alpha']

        initial_rigid_body, both_shapes, _, _ = initialize_system(
            spider_base_radius, spider_head_height,
            spider_leg_diameter, initial_separation_coeff=initial_separation_coeff)

        # For now, we omit the full trajectory and rely on JAX compiler.
        # If still slow, try taking its computation out of the function
        fin_state, traj = run_dynamics(initial_rigid_body, both_shapes, SHELL_VERTEX_RADIUS,
                                 spider_leg_diameter, spider_head_diameter, key,
                                 morse_ii_eps=morse_ii_eps, morse_leg_eps=morse_leg_eps, morse_head_eps=morse_head_eps,
                                 morse_ii_alpha=morse_ii_alpha, morse_leg_alpha=morse_leg_alpha, morse_head_alpha=morse_head_alpha,
                                 soft_eps=soft_eps, kT=kT, dt=dt,
                                 # num_inner_steps=num_inner_steps, num_outer_steps=num_outer_steps
                                 num_steps=num_steps, gamma=gamma
        )
        # v_loss_fn = vmap(loss_fn, (0, None, None, None))
        # return jnp.mean(v_loss_fn(traj, eta, min_com_dist, max_com_dist))
        return loss_fn(fin_state, eta, min_com_dist, max_com_dist)
        # return loss_fn(fin_state, eta=eta)
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
    params = get_init_params(mode=init_method, key=key)
    opt_state = optimizer.init(params)

    eval_params_fn = get_eval_params_fn(soft_eps=100000.0, kT=kT, dt=1e-3,
                                        # num_inner_steps=n_inner_steps, num_outer_steps=n_outer_steps,
                                        num_steps=n_steps,
                                        morse_ii_eps=10.0, morse_ii_alpha=5.0,
                                        initial_separation_coeff=initial_separation_coefficient,
                                        gamma=gamma,
                                        min_com_dist=min_com_dist, max_com_dist=max_com_dist,
                                        eta=eta) # FIXME: separation coefficient is hardcoded for now
    grad_eval_params_fn = jit(value_and_grad(eval_params_fn))
    batched_grad_fn = jit(vmap(grad_eval_params_fn, in_axes=(None, 0)))




    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # run_name = f"catalyst_{timestamp}_b{batch_size}_n{n_steps}_lr{lr}"
    run_name = f"catalyst_b{batch_size}_n{n_steps}_lr{lr}_i{init_method}_s{initial_separation_coefficient}_kT{kT}_g{gamma}_min{min_com_dist}_max{max_com_dist}_e{eta}_k{key_seed}"
    run_dir = data_dir / run_name
    print(f"Making directory: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"

    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    loss_path = run_dir / "loss.txt"
    losses_path = run_dir / "losses.txt"
    std_path = run_dir / "std.txt"
    # loss_file = open(loss_path, "a")
    grad_path = run_dir / "grads.txt"
    # grad_file = open(grad_path, "a")
    params_path = run_dir / "params_per_iter.txt"
    # params_file = open(params_path, "a")


    for i in tqdm(range(n_iters)):
        print(f"\nIteration: {i}")
        iter_key = keys[i]
        batch_keys = random.split(iter_key, batch_size)
        # val, grads = grad_eval_params_fn(params, iter_key)
        start = time.time()
        vals, grads = batched_grad_fn(params, batch_keys)
        end = time.time()

        avg_grads = {k: jnp.mean(grads[k], axis=0) for k in grads}
        print(f"Average gradients: {avg_grads}")
        print(f"Batch losses: {vals}")
        print(f"Avg. loss: {onp.mean(vals)}")
        print(f"Gradient calculation time: {onp.round(end - start, 2)}")

        '''
        avg_grads = dict()
        for k in grads[0].keys():
            all_k_grads = list()
            for j in range(batch_size):
                all_k_grads.append(grads[j][k])
            avg_k_grad = onp.mean(all_k_grads, axis=0)
            avg_grads[k] = avg_k_grad
        '''

        updates, opt_state = optimizer.update(avg_grads, opt_state)
        params = optax.apply_updates(params, updates)
        with open(std_path, "a") as f:
            f.write(f"{onp.std(vals)}\n")
        with open(losses_path, "a") as f:
            f.write(f"{vals}\n")
        with open(loss_path, "a") as f:
            f.write(f"{onp.mean(vals)}\n")
        with open(grad_path, "a") as f:
            f.write(str(grads) + '\n')
        with open(params_path, "a") as f:
            params_to_print = {k: float(v) for k, v in params.items()}
            f.write(str(params_to_print) + '\n')

    # loss_file.close()
    # grad_file.close()
    # params_file.close()

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


    """
    start = time.time()
    init_params = get_init_params()
    key = random.PRNGKey(0)

    eval_params = get_eval_params_fn(soft_eps=10000.0, kT=1.0, dt=1e-4, num_steps=args['n_steps'],
                                     morse_ii_eps=10.0, morse_ii_alpha=5.0)
    val = eval_params(init_params, key=key)
    # val, _grad = value_and_grad(eval_params)(init_params, key)
    end = time.time()
    print(f"Total time: {onp.round(end - start, 2)}")
    print(f"Value: {val}")
    # print(f"Grad: {_grad}")
    """
