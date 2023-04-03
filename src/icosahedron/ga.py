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



def run(population_size, args, selection_method="competition"):


    if selection_method == "competition":
        def selection_method(mutant_losses, mutants):
            n_mutants = len(mutants)
            new_population = list()
            for _ in range(n_mutants):
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
        sigma = (hi - low) / denom
        perturbation = onp.random.normal(scale=sigma)
        mutated_value = mutant[key_to_mutate] + perturbation
        if mutated_value < 0.0:
            mutated_value = mutant[key_to_mutate]
        mutant[key_to_mutate] = mutated_value
        # FIXME: check that it's truly pass-by-reference and that we don't have to return mutant

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


    iter_keys = random.split(key, n_iters)
    best_losses = list()
    for g in range(n_iters):
        iter_key = iter_keys[g]

        # Evaluate the losses of the current population
        m_keys = random.split(iter_key, population_size)
        losses = list()
        for mutant, m_key in zip(population, m_keys):
            mutant_loss = eval_params_fn(mutant, m_key)
            losses.append(mutant_loss)
        ## Check the best loss
        best_mutant_idx = onp.argsort(losses)[0]
        best_mutant = population[best_mutant_idx]
        best_mutant_loss = losses[best_mutant_idx]

        # Selection
        if g != n_iter - 1: # Don't select if last generation!!
            population = selection_method(losses, population)

        # Mutation
        # FIXME: option to not mutate the current best?
        if g != n_iter - 1: # Don't mutate if last generation!!
            for mutant in population:
                mutation_fn(mutant)

    return best_mutant, best_mutant_loss, population, losses






if __name__ == "__main__":
    parser = get_argparse()
    args = vars(parser.parse_args())

    run(population_size=10, args)
