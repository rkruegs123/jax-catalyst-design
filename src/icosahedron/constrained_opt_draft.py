# Note: this file is going to be a mess. Lot's of code copied from `optimize.py`
import pdb
import time
import numpy as onp
import argparse
from pathlib import Path

import jax.numpy as jnp
from jax import random, jit, vmap, jacrev
import jax.debug
from jax.config import config
config.update('jax_enable_x64', True)

from common import SHELL_VERTEX_RADIUS, get_init_params, VERTEX_TO_BIND
import mod_rigid_body as rigid_body
import modified_ipopt as mipopt
from simulation import run_dynamics, initialize_system, loss_fn, shape_species
import simulation
import leg


MORSE_II_EPS = 10.0
MORSE_II_ALPHA = 5.0
SOFT_EPS = 100000.0
TARGET_ENERGY = 264.0

def get_sim_fn(
        soft_eps, kT, dt,
        num_steps,
        morse_ii_eps, morse_ii_alpha,
        initial_separation_coeff, gamma,
        min_com_dist, max_com_dist,
        eta
):
    def sim_fn(params, key):
        """
        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
        spider_leg_diameter = params['spider_leg_diameter']
        spider_head_diameter = params['spider_head_diameter']

        morse_leg_eps = params['morse_leg_eps']
        log_morse_head_eps = params['log_morse_head_eps']
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params['morse_leg_alpha']
        morse_head_alpha = params['morse_head_alpha']
        """
        spider_base_radius = params[0]
        spider_head_height = params[1]
        spider_leg_diameter = params[2]
        spider_head_diameter = params[3]

        morse_leg_eps = params[4]
        log_morse_head_eps = params[5]
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params[6]
        morse_head_alpha = params[7]


        initial_rigid_body, both_shapes, _, _ = initialize_system(
            spider_base_radius, spider_head_height,
            spider_leg_diameter, initial_separation_coeff=initial_separation_coeff)

        # For now, we omit the full trajectory and rely on JAX compiler.
        # If still slow, try taking its computation out of the function
        fin_state, traj = run_dynamics(
            initial_rigid_body, both_shapes, SHELL_VERTEX_RADIUS,
            spider_leg_diameter, spider_head_diameter, key,
            morse_ii_eps=morse_ii_eps, morse_leg_eps=morse_leg_eps,
            morse_head_eps=morse_head_eps, morse_ii_alpha=morse_ii_alpha,
            morse_leg_alpha=morse_leg_alpha, morse_head_alpha=morse_head_alpha,
            soft_eps=soft_eps, kT=kT, dt=dt,
            num_steps=num_steps, gamma=gamma
        )
        return fin_state
        # return loss_fn(fin_state, eta, min_com_dist, max_com_dist)
    return sim_fn






def train(args):

    # 0.) Setup
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

    key = random.PRNGKey(key_seed)
    key, params_key = random.split(key)
    init_params_dict = get_init_params(mode=init_method, key=key)
    init_params_arr = jnp.array(list(init_params_dict.values()))


    # 1.) Define our loss function
    # FIXME: we are hardcoding keys for now. We think the correct solution would be to use some sort of callback to the minimizer that updates the key
    key, batch_key = random.split(key)
    batch_keys = random.split(batch_key, batch_size)
    sim_fn = jit(get_sim_fn(
        soft_eps=SOFT_EPS, kT=kT, dt=1e-3,
        num_steps=n_steps,
        morse_ii_eps=MORSE_II_EPS, morse_ii_alpha=MORSE_II_ALPHA,
        initial_separation_coeff=initial_separation_coefficient,
        gamma=gamma,
        min_com_dist=min_com_dist, max_com_dist=max_com_dist,
        eta=eta
    ))
    mapped_sim_fn = jit(vmap(sim_fn, (None, 0)))


    tmp_path = Path("another_tmp_output.txt")


    def avg_loss_fn(fin_states):
        losses = vmap(loss_fn, (0, None, None, None))(fin_states, eta, min_com_dist, max_com_dist)
        return jnp.mean(losses)


    """
    Takes in a bound state which is a RigidBody with N=13 elements.
    The last element is the COM of the spider/catalyst, and VERTEX_TO_BIND
    is the index of the vertex we are going to pull off.

    In our equality constraint, we assume abduction (with the expectation
    that the fitness landscape, under this constraint, will be biased
    towards such behavior.

    So, to get an unbound state, we simply add a fixed z-value
    to both the spider and the vertex to bind.
    """
    @jit
    def get_unbound_state(bound_state, z_offset=20.0):
        new_spider_pos = bound_state.center[-1] + z_offset
        new_bound_vertex_pos = bound_state.center[VERTEX_TO_BIND] + z_offset

        unbound_state_center = bound_state.center.at[-1].set(new_spider_pos)
        unbound_state_center = unbound_state_center.at[VERTEX_TO_BIND].set(new_bound_vertex_pos)

        unbound_state = rigid_body.RigidBody(
            center=unbound_state_center,
            orientation=bound_state.orientation)

        return unbound_state
    mapped_get_unbound_states = jit(vmap(get_unbound_state, (0, None)))

    @jit
    def problem(params):

        # (i) Unpack our arguments
        """
        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
        spider_leg_diameter = params['spider_leg_diameter']
        spider_head_diameter = params['spider_head_diameter']

        morse_leg_eps = params['morse_leg_eps']
        log_morse_head_eps = params['log_morse_head_eps']
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params['morse_leg_alpha']
        morse_head_alpha = params['morse_head_alpha']
        """
        spider_base_radius = params[0]
        spider_head_height = params[1]
        spider_leg_diameter = params[2]
        spider_head_diameter = params[3]

        morse_leg_eps = params[4]
        log_morse_head_eps = params[5]
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params[6]
        morse_head_alpha = params[7]


        # (ii) Construct our energy function
        _, tmp_both_shapes, _, _ = initialize_system(
            spider_base_radius, spider_head_height,
            spider_leg_diameter, initial_separation_coeff=initial_separation_coefficient)
        base_energy_fn = simulation.get_energy_fn(
            SHELL_VERTEX_RADIUS, spider_leg_diameter,
            spider_head_diameter,
            morse_ii_eps=MORSE_II_EPS,
            morse_leg_eps=morse_leg_eps, morse_head_eps=morse_head_eps,
            morse_ii_alpha=MORSE_II_ALPHA, morse_leg_alpha=morse_leg_alpha,
            morse_head_alpha=morse_head_alpha,
            soft_eps=SOFT_EPS, shape=tmp_both_shapes)
        leg_energy_fn = leg.get_leg_energy_fn(SOFT_EPS, (spider_leg_diameter/2 + SHELL_VERTEX_RADIUS), tmp_both_shapes, shape_species)
        energy_fn = lambda body: base_energy_fn(body) + leg_energy_fn(body)
        energy_fn = jit(energy_fn)
        mapped_energy_fn = vmap(energy_fn)


        # (iii) Compute the loss
        fin_states = mapped_sim_fn(params, batch_keys) # Assumed to be bound

        # (iv) Compute our equality constraint
        # Note: We begin by computing the mean energy difference. The potential
        # issue here is that high variance values could lead to the desired mean.
        bound_energies = mapped_energy_fn(fin_states)
        mean_bound_energy = jnp.mean(bound_energies)

        unbound_states = mapped_get_unbound_states(fin_states, 10.0)
        unbound_energies = mapped_energy_fn(unbound_states)
        mean_unbound_energy = jnp.mean(unbound_energies)

        eq_constraint = mean_unbound_energy - mean_bound_energy - TARGET_ENERGY

        return avg_loss_fn(fin_states), eq_constraint, 0.0

    @jit
    def problem_debug(params):

        # (i) Unpack our arguments
        """
        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
        spider_leg_diameter = params['spider_leg_diameter']
        spider_head_diameter = params['spider_head_diameter']

        morse_leg_eps = params['morse_leg_eps']
        log_morse_head_eps = params['log_morse_head_eps']
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params['morse_leg_alpha']
        morse_head_alpha = params['morse_head_alpha']
        """
        spider_base_radius = params[0]
        spider_head_height = params[1]
        spider_leg_diameter = params[2]
        spider_head_diameter = params[3]

        morse_leg_eps = params[4]
        log_morse_head_eps = params[5]
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params[6]
        morse_head_alpha = params[7]


        # (ii) Construct our energy function
        _, tmp_both_shapes, _, _ = initialize_system(
            spider_base_radius, spider_head_height,
            spider_leg_diameter, initial_separation_coeff=initial_separation_coefficient)
        base_energy_fn = simulation.get_energy_fn(
            SHELL_VERTEX_RADIUS, spider_leg_diameter,
            spider_head_diameter,
            morse_ii_eps=MORSE_II_EPS,
            morse_leg_eps=morse_leg_eps, morse_head_eps=morse_head_eps,
            morse_ii_alpha=MORSE_II_ALPHA, morse_leg_alpha=morse_leg_alpha,
            morse_head_alpha=morse_head_alpha,
            soft_eps=SOFT_EPS, shape=tmp_both_shapes)
        leg_energy_fn = leg.get_leg_energy_fn(SOFT_EPS, (spider_leg_diameter/2 + SHELL_VERTEX_RADIUS), tmp_both_shapes, shape_species)
        energy_fn = lambda body: base_energy_fn(body) + leg_energy_fn(body)
        energy_fn = jit(energy_fn)
        mapped_energy_fn = vmap(energy_fn)


        # (iii) Compute the loss
        fin_states = mapped_sim_fn(params, batch_keys) # Assumed to be bound

        # (iv) Compute our equality constraint
        # Note: We begin by computing the mean energy difference. The potential
        # issue here is that high variance values could lead to the desired mean.
        bound_energies = mapped_energy_fn(fin_states)
        mean_bound_energy = jnp.mean(bound_energies)

        unbound_states = mapped_get_unbound_states(fin_states, 10.0)
        unbound_energies = mapped_energy_fn(unbound_states)
        mean_unbound_energy = jnp.mean(unbound_energies)

        eq_constraint = mean_unbound_energy - mean_bound_energy - TARGET_ENERGY

        return mean_unbound_energy, mean_bound_energy

    # Do the optimization
    obj_jit = mipopt.ObjectiveWrapper(jit(problem))
    grad_reverse = mipopt.GradWrapper(jit(jacrev(problem, argnums=0)))

    options = {
        'max_iter': n_iters,
        'disp': 5, 'tol': 1e-6,
        'print_timing_statistics': 'yes'
        #,'acceptable_constr_viol_tol':1e-1,'acceptable_obj_change_tol':1e-3}
    }

    cons = [{'type': 'eq', 'fun': obj_jit.const, 'jac': grad_reverse.const},
            {'type': 'ineq', 'fun': obj_jit.ineqconst, 'jac': grad_reverse.ineqconst}]

    bounds = None
    traj_iter = None
    # Note that params is a dictionary

    init_mean_unbound_energy, init_mean_bound_energy = problem_debug(init_params_arr)
    with open(tmp_path, "a") as of:
        of.write(f"Init mean unbound energy: {init_mean_unbound_energy}")
        of.write(f"Init mean bound energy: {init_mean_bound_energy}")

    res, trajectory, objective_list, grad_list = mipopt.minimize_ipopt(
        obj_jit, x0=init_params_arr,
        jac=grad_reverse, constraints=cons,
        bounds=bounds, options=options, traj_iter=traj_iter
    )

    fin_mean_unbound_energy, fin_mean_bound_energy = problem_debug(res.x)
    with open(tmp_path, "a") as of:
        of.write(f"Fin mean unbound energy: {fin_mean_unbound_energy}")
        of.write(f"Fin mean bound energy: {fin_mean_bound_energy}")

    return res






def get_argparse():
    parser = argparse.ArgumentParser(description="Simulation for spider catalyst design")

    parser.add_argument('--batch-size', type=int, default=10, help="Num. batches for each round of gradient descent")
    parser.add_argument('--n-iters', type=int, default=100, help="Num. iterations of gradient descent")
    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    parser.add_argument('--n-steps', type=int, default=1000, help="Num. steps per simulation")
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
    import pickle

    parser = get_argparse()
    args = vars(parser.parse_args())

    print(f"About to constrain our optimization...")
    result = train(args)

    of_name = f"first_constrained_opt_results.pkl"
    of_file = open(of_name, "wb")
    pickle.dump(result, of_file)
    of_file.close()














