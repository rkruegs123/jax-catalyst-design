import pdb
import unittest
import numpy as onp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from pathlib import Path

from jax import random, jit, vmap, lax
import jax.numpy as jnp
from jax_md import space, simulate, energy, smap, rigid_body

# import catalyst.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)


def compute_alpha(rc, sigma):
    return 2 * (rc / sigma)**2 * (3 / (2 * ((rc/sigma)**2 - 1)))**3

def compute_rmin(rc, sigma):
    return rc * (3 / (1 + 2 * (rc/sigma)**2))**(1/2)

def get_not_lj(eps, sigma, rc):
    alpha = compute_alpha(rc, sigma)
    
    def not_lj(r):
        val = eps * alpha * ((sigma/r)**2 - 1) * ((rc/r)**2 - 1)**2
        return jnp.where(r <= rc, val, 0.0)

    return not_lj


def get_first_dissociation_time(key, eps, sigma, rc, max_steps=int(1e5)):

    rmin = compute_rmin(rc, sigma)

    init_dimer_dist = rmin # start bonded
    box_size = 15.0

    monomers = jnp.array([[box_size / 2, box_size / 2],
                          [box_size / 2, box_size / 2 + init_dimer_dist]])

    displacement_fn, shift_fn = space.periodic(box_size)

    not_lj = get_not_lj(eps, sigma, rc)

    @jit
    def energy_fn(positions):
        m1 = positions[0]
        m2 = positions[1]
        dr = displacement_fn(m1, m2)
        return not_lj(space.distance(dr))

    init_fn, apply_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt=1e-4, kT=1.0, gamma=12.5)

    state = init_fn(key, monomers, mass=1.0)

    # do_step = lambda state, t: (apply_fn(state), state.position)
    do_step = lambda state, t: (apply_fn(state), space.distance(displacement_fn(state.position[0], state.position[1])))
    do_step = jit(do_step)


    fin_state, dists = lax.scan(do_step, state, jnp.arange(max_steps))
    check_dists = (dists > rc).astype(jnp.int32)
    if jnp.sum(check_dists) == 0:
        print("Did not find dissociation time")
        return -1
    return jnp.nonzero(check_dists)[0][0]
    # return jnp.where(jnp.sum(check_dists) == 0, -1, jnp.argmax(check_dists))


def get_dissociation_distribution(key, batch_size, eps, sigma, rc, output_path, max_steps=int(1e5), dt=1e-4):
    diss_times = []

    for b in tqdm(range(batch_size)):
        key, split = random.split(key)
        t = get_first_dissociation_time(split, eps, sigma, rc, max_steps)
        if t != -1:
            diss_times += [t*dt]
        else:
            with open(output_path, "a") as f:
                f.write(f"warning: no dissociation at batch {b}\n")
            
    return diss_times


def get_argparse():
    parser = argparse.ArgumentParser(description="Computing monomer dissociation time")

    parser.add_argument('--batch-size', type=int, default=3, help="Num. batches")
    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    
    parser.add_argument('--max-steps', type=int, help="Max steps per simulation")
    parser.add_argument('--eps', type=float, help="Epsilon for the not-LJ potential")
    parser.add_argument('--run-name', type=str, default="",
                        help='Name of run directory')

    return parser


if __name__ == "__main__":

    # Load all the arguments
    parser = get_argparse()
    args = vars(parser.parse_args())

    sigma = 1.0
    rc = 1.1
    eps = args['eps']
    dt = 1e-4
    batch_size = args['batch_size']
    key_seed = args['key_seed']
    key = random.PRNGKey(key_seed)
    max_steps = args['max_steps']
    assert(rc / sigma == 1.1)

    # Make a run directory
    output_dir = Path("data/dimer")
    run_name = args['run_name']
    run_suffix = f"b{batch_size}_k{key_seed}_m{max_steps}_e{eps}"
    run_name += run_suffix
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    # Save the parameters in a file
    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"

    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Compute the expected values
    output_path = run_dir / "output.txt"
        
    expected_avg_diss_time = -0.91 * eps + 2.2
    expected_avg_num_steps = 1 / jnp.exp(expected_avg_diss_time) / dt

    with open(output_path, "a") as f:
        f.write(f"expected average ln(k): {expected_avg_diss_time}\n")
        f.write(f"expected avg. number of steps: {expected_avg_num_steps}\n")

    start = time.time()
    diss_times = get_dissociation_distribution(
        key, batch_size, eps, sigma, rc, output_path, max_steps=max_steps, dt=dt)
    end = time.time()
    onp.save(run_dir / "diss_times.npy", diss_times, allow_pickle=False)
    
    ln_k_measured = jnp.log( 1 / jnp.mean(jnp.array(diss_times)))
    with open(output_path, "a") as f:
        f.write(f"measured average ln k: {ln_k_measured}\n")
        f.write(f"avg time per batch: {(end - start) / batch_size}\n")

    ## Plot a running average
    all_means = list()
    all_is = list()
    for i in range(1, len(diss_times)):
        i_mean = jnp.mean(jnp.array(diss_times[:i]))
        i_ln_k_measured = jnp.log(1 / i_mean)

        all_means.append(i_ln_k_measured)
        all_is.append(i)
    plt.plot(all_is, all_means)
    plt.savefig(run_dir / "running_avg.png")

