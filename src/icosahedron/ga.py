import pdb
import numpy as onp
from functools import partial
from typing import Optional, Tuple, Dict, Callable, List, Union
import matplotlib.pyplot as plt
import time
import datetime
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint

import jax.numpy as np

import jax
from jax import random, grad, value_and_grad, remat, jacfwd
from jax import jit
from jax import vmap, lax
from jax import ops
from jax.config import config
config.update('jax_enable_x64', True)

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition # , rigid_body
from jax_md import dataclasses
from jax_md import util

import common
from common import SHELL_VERTEX_RADIUS, dtype, get_init_params, parameter_ranges
import simulation
from simulation import run_dynamics, initialize_system, loss_fn
from optimize import get_eval_params_fn, get_argparse



def run(args, selection_method="competition"):


    if selection_method == "competition":
        def selection_method(mutant_losses, mutants):
            n_mutants = len(mutants)
            new_population = list()

            # Always keep a copy of the best!
            best_mutant_idx = onp.argsort(mutant_losses)[0]
            new_population.append(deepcopy(mutants[best_mutant_idx]))

            for _ in range(n_mutants-1):
                competitor1, competitor2 = onp.random.choice(onp.arange(n_mutants), 2, replace=False)
                if mutant_losses[competitor1] < mutant_losses[competitor2]:
                    new_population.append(deepcopy(mutants[competitor1]))
                else:
                    new_population.append(deepcopy(mutants[competitor2]))
            return new_population # A set of mutants
    else:
        raise NotImplementedError(f"Invalid selection method: {selection_method}")

    # https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)
    def mutation_fn(mutant, denom=6):
        key_to_mutate = str(onp.random.choice(list(mutant.keys())))
        lo, hi = parameter_ranges[key_to_mutate]
        sigma = (hi - lo) / denom
        perturbation = onp.random.normal(scale=sigma)
        if key_to_mutate == "morse_head_eps":
            mutated_value = onp.exp(onp.log(mutant[key_to_mutate] + perturbation))
        else:
            mutated_value = mutant[key_to_mutate] + perturbation
        if mutated_value < 0.0:
            mutated_value = mutant[key_to_mutate]
        mutant[key_to_mutate] = mutated_value
        # FIXME: check that it's truly pass-by-reference and that we don't have to return mutant

    population_size = args['population_size']
    batch_size = args['batch_size']
    n_iters = args['n_iters'] # Number of generations
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

    key = random.PRNGKey(key_seed)
    key, pop_key = random.split(key, 2)
    init_pop_keys = random.split(pop_key, population_size)
    population = [get_init_params("random", k) for k in init_pop_keys]

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise RuntimeError(f"No data directory exists at location: {data_dir}")
    run_name = f"ga_b{batch_size}_n{n_steps}_s{initial_separation_coefficient}_kT{kT}_g{gamma}_min{min_com_dist}_max{max_com_dist}_e{eta}_pop{population_size}_k{key_seed}"
    run_dir = data_dir / run_name
    print(f"Making directory: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)
    losses_path = run_dir / "losses.txt"
    std_path = run_dir / "std.txt"
    generation_path = run_dir / "generation.txt"
    best_per_gen_path = run_dir / "best_per_generation.txt"
    best_loss_per_gen_path = run_dir / "best_loss_per_generation.txt"

    eval_params_fn = get_eval_params_fn(
        soft_eps=100000.0,
        kT=kT,
        dt=1e-3,
        num_steps=n_steps,
        morse_ii_eps=10.0,
        morse_ii_alpha=5.0,
        initial_separation_coeff=initial_separation_coefficient,
        gamma=gamma,
        min_com_dist=min_com_dist, max_com_dist=max_com_dist,
        eta=eta)
    eval_params_fn = jit(eval_params_fn)
    mapped_eval_params_fn = vmap(eval_params_fn, in_axes=(None, 0))

    iter_keys = random.split(key, n_iters)
    best_losses = list()
    for g in tqdm(range(n_iters)):
        iter_key = iter_keys[g]

        # Evaluate the losses of the current population
        m_keys = random.split(iter_key, population_size)
        losses = list()
        stds = list()
        for mutant, m_key in zip(population, m_keys):
            batch_keys = random.split(m_key, batch_size)
            mutant_losses = mapped_eval_params_fn(mutant, batch_keys)
            mutant_losses = onp.array(mutant_losses)

            mutant_std = onp.std(mutant_losses)
            mutant_loss = onp.mean(mutant_losses)
            losses.append(mutant_loss)
            stds.append(mutant_std)
        ## Check the best loss
        best_mutant_idx = onp.argsort(losses)[0]
        best_mutant = population[best_mutant_idx]
        best_mutant_loss = losses[best_mutant_idx]

        # Selection
        if g != n_iters - 1: # Don't select if last generation!!
            population = selection_method(losses, population)

        # Mutation
        # FIXME: option to not mutate the current best?
        if g != n_iters - 1: # Don't mutate if last generation!!
            for mutant in population:
                mutation_fn(mutant)

        with open(best_per_gen_path, "a") as f:
            f.write(f"Iteration {g}, loss {best_mutant_loss}, mutant: {best_mutant}\n")
        with open(best_loss_per_gen_path, "a") as f:
            f.write(f"{best_mutant_loss}\n")
        with open(generation_path, "a") as f:
            f.write(f"{population}\n")
        with open(losses_path, "a") as f:
            f.write(f"{losses}\n")
        with open(std_path, "a") as f:
            f.write(f"{stds}\n")

    return best_mutant, best_mutant_loss, population, losses






if __name__ == "__main__":
    parser = get_argparse()
    parser.add_argument('--population-size', type=int, default=10, help="Populationsize")
    args = vars(parser.parse_args())

    run(args=args)
