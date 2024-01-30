import pdb
import unittest
import numpy as onp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import functools
from pathlib import Path

from jax import random, jit, vmap, lax, grad, value_and_grad
import jax.numpy as jnp
from jax_md import space, simulate, energy, smap, rigid_body
import optax

from catalyst import checkpoint


checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)

def compute_rmin(rc, sigma):
    return rc * (3 / (1 + 2 * (rc/sigma)**2))**(1/2)

def compute_alpha(rc, sigma):
    return 2 * (rc / sigma)**2 * (3 / (2 * ((rc/sigma)**2 - 1)))**3

def get_argparse():
    parser = argparse.ArgumentParser(description="Eummy optimization over long time scales")

    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    
    parser.add_argument('--num-steps', type=int, help="Num steps per simulation")
    parser.add_argument('--num-iters', type=int, help="Num iterations of gradient descent")
    parser.add_argument('--init-eps', type=float, help="Epsilon for the not-LJ potential")
    parser.add_argument('--dt', type=float, default=1e-4, help="Timestep")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate")
    parser.add_argument('--run-name', type=str, default="",
                        help='Name of run directory')

    return parser

if __name__ == "__main__":
    parser = get_argparse()
    args = vars(parser.parse_args())

    sigma = 1.0
    rc = 1.1
    rmin = compute_rmin(rc, sigma)
    init_eps = args['init_eps']
    dt = args['dt']
    lr = args['lr']
    key_seed = args['key_seed']
    num_steps = args['num_steps']
    num_iters = args['num_iters']
    key = random.PRNGKey(key_seed)
    assert(rc / sigma == 1.1)

    # Make a run directory
    output_dir = Path("data/dimer")
    run_name = args['run_name']
    run_suffix = f"opt_test_e{init_eps}_k{key_seed}_n{num_steps}_i{num_iters}_dt{dt}"
    run_name += run_suffix
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    # Save the parameters in a file
    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"

    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Set up the simulation
    box_size = 15.0


    def get_not_lj(eps, sigma, rc):
        alpha = compute_alpha(rc, sigma)

        def not_lj(r):
            val = eps * alpha * ((sigma/r)**2 - 1) * ((rc/r)**2 - 1)**2
            return jnp.where(r <= rc, val, 0.0)

        return not_lj

    init_monomers = jnp.array([[box_size / 2, box_size / 2],
                               [box_size / 2, box_size / 2 + rmin]])
    def loss_fn(params, iter_key):
        eps = params['eps']

        displacement_fn, shift_fn = space.periodic(box_size)
        not_lj = get_not_lj(eps, sigma, rc)

        @jit
        def energy_fn(positions):
            m1 = positions[0]
            m2 = positions[1]
            dr = displacement_fn(m1, m2)
            return not_lj(space.distance(dr))

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt=dt, kT=1.0, gamma=12.5)

        init_state = init_fn(iter_key, init_monomers, mass=1.0)

        @jit
        def fori_step_fn(t, iter_info):
            state = iter_info['state']
            curr_sum = iter_info['sm']
            state = step_fn(state)
            dist = space.distance(displacement_fn(state.position[0], state.position[1]))
            return {'state': state, 'sm': curr_sum + dist}
            
        fin_info = lax.fori_loop(0, num_steps, fori_step_fn, {'state': init_state, 'sm': 0.0})
        avg_dist = fin_info['sm'] / num_steps
        return -avg_dist

    # Do the optimization
    params = {"eps": init_eps}
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    grad_fn = value_and_grad(loss_fn)
    loss_path = run_dir / "loss.txt"
    params_path = run_dir / "eps.txt"
    grads_path = run_dir / "grads.txt"
    times_path = run_dir / "times.txt"
    for i in tqdm(range(num_iters)):
        key, iter_key = random.split(key)
        start = time.time()
        loss, grads = grad_fn(params, iter_key)
        end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(grads_path, "a") as f:
            f.write(f"{grads}\n")
        curr_eps = params['eps']
        with open(params_path, "a") as f:
            f.write(f"{curr_eps}\n")
        with open(times_path, "a") as f:
            f.write(f"{end-start}\n")


        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
