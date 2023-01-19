import pdb
import numpy as onp
from functools import partial
from typing import Optional, Tuple, Dict, Callable, List, Union
import matplotlib.pyplot as plt

import jax.numpy as np

from jax import random, grad, value_and_grad, remat, jacfwd
from jax import jit
from jax import vmap, lax
from jax import ops
from jax.config import config
config.update('jax_enable_x64', True)
config.update("jax_debug_nans", True)

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition # , rigid_body
from jax_md import dataclasses
from jax_md import util

import optax

import common
from common import SHELL_VERTEX_RADIUS
import simulation
from simulation import run_dynamics, initialize_system, loss_fn



# fixme: we're not passing a key here but run_dynamics takes one

def get_eval_params_fn(soft_eps, kT, dt, num_steps, morse_ii_eps, morse_ii_alpha):
    def eval_params(params, key):
        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
        spider_leg_diameter = params['spider_leg_diameter']
        spider_head_diameter = params['spider_head_diameter']

        morse_leg_eps = params['morse_leg_eps']
        morse_head_eps = params['morse_head_eps']
        morse_leg_alpha = params['morse_leg_alpha']
        morse_head_alpha = params['morse_head_alpha']

        initial_rigid_body, both_shapes, _, _ = initialize_system(spider_base_radius, spider_head_height)

        # For now, we omit the full trajectory and rely on JAX compiler.
        # If still slow, try taking its computation out of the function
        fin_state = run_dynamics(initial_rigid_body, both_shapes, SHELL_VERTEX_RADIUS,
                                 spider_leg_diameter, spider_head_diameter, key,
                                 morse_ii_eps=morse_ii_eps, morse_leg_eps=morse_leg_eps, morse_head_eps=morse_head_eps,
                                 morse_ii_alpha=morse_ii_alpha, morse_leg_alpha=morse_leg_alpha, morse_head_alpha=morse_head_alpha,
                                 soft_eps=soft_eps, kT=kT, dt=dt, num_steps=num_steps)
        return loss_fn(fin_state)
    return eval_params


def get_init_params():
    init_params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': 3.0,
        'spider_leg_diameter': 1.0,
        'spider_head_diameter': 3.0,

        # catalyst energy
        'morse_leg_eps': 2.0,
        'morse_head_eps': 200.0,
        'morse_leg_alpha': 2.0,
        'morse_head_alpha': 5.0
    }
    return init_params

def train(n_iters=10, batch_size=1, n_steps=5):

    key = random.PRNGKey(0)
    keys = random.split(key, n_iters)

    lr = 1e-2
    optimizer = optax.adam(lr)
    params = get_init_params()
    opt_state = optimizer.init(init_params)

    eval_params_fn = get_eval_params_fn(soft_eps=10000.0, kT=1.0, dt=1e-4, num_steps=n_steps,
                                        morse_ii_eps=10.0, morse_ii_alpha=5.0) # FIXME: naming
    grad_eval_params_fn = jit(value_and_grad(eval_params_fn)) # FIXME: naming
    batched_grad_fn = jit(vmap(grad_eval_params_fn, in_axes=(None, 0)))

    loss_file = open("loss_file.txt", "a+")
    grad_file = open("grad_file.txt", "a+")

    for i in range(n_iters):
        iter_key = keys[i]
        batch_keys = random.split(iter_key, batch_size)
        #val, grads = grad_eval_params_fn(params, iter_key)
        val, grads = batched_grad_fn(params, batch_keys)

        avg_grads = dict()
        for k in grads[0].keys():
            all_k_grads = list()
            for j in range(batch_size):
                all_k_grads.append(grads[j][k])
            avg_k_grad = onp.mean(all_k_grads, axis=0)
            avg_grads[k] = avg_k_grad

        updates, opt_state = optimizer.update(avg_grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_file.write(val)
        grad_file.write(grads)

    loss_file.close()
    grad_file.close()

if __name__ == "__main__":

    import time

    print(f"Testing optimization...")

    start = time.time()
    init_params = get_init_params()
    key = random.PRNGKey(0)

    eval_params = get_eval_params_fn(soft_eps=10000.0, kT=1.0, dt=1e-4, num_steps=20,
                                     morse_ii_eps=10.0, morse_ii_alpha=5.0)
    # eval_params(init_params, key=key)
    val, _grad = value_and_grad(eval_params)(init_params, key)
    end = time.time()
    print(f"Total time: {onp.round(end - start, 2)}")
    print(f"Value: {val}")
    print(f"Grad: {_grad}")
