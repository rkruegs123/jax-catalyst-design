# Note: this file is going to be a mess. Lot's of code copied from `optimize.py`


import pdb

import numpy as onp
import jax.numpy as jnp
from jax import jit, jacrev, grad

import modified_ipopt as mipopt


FIXED_SUM = 10.0

def loss(params):
    return jnp.var(params)

def equality_constraint(params):
    return jnp.sum(params) - FIXED_SUM

def inequality_constraints(params):
    return 0.0

def problem(params):
    return loss(params), equality_constraint(params), inequality_constraints(params)












MORSE_II_EPS = 10.0
MORSE_II_ALPHA = 5.0
SOFT_EPS = 100000.0

def get_sim_fn(
        soft_eps, kT, dt,
        num_steps,
        morse_ii_eps, morse_ii_alpha,
        initial_separation_coeff, gamma,
        min_com_dist, max_com_dist,
        eta
):
    def sim_fn(params, key):
        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
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
    params = get_init_params(mode=init_method, key=key)

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


    def avg_loss_fn(fin_states):
        losses = vmap(loss_fn, (0, None, None, None))(fin_states, eta, min_com_dist, max_com_dist)
        return jnp.mean(losses)


    # 2.) Define our equality constrain

    def equality_constraint(fin_states):
        raise NotImplementedError

    def inequality_constraints(fin_states):
        return 0.0

    def problem(params):

        # (i) Prepare
        spider_base_radius = params['spider_base_radius']
        spider_head_height = params['spider_head_height']
        spider_leg_diameter = params['spider_leg_diameter']
        spider_head_diameter = params['spider_head_diameter']

        morse_leg_eps = params['morse_leg_eps']
        log_morse_head_eps = params['log_morse_head_eps']
        morse_head_eps = jnp.exp(log_morse_head_eps)
        morse_leg_alpha = params['morse_leg_alpha']
        morse_head_alpha = params['morse_head_alpha']


        ## Have to first get our energy function
        _, tmp_both_shapes, _, _ = initialize_system(
            spider_base_radius, spider_head_height,
            spider_leg_diameter, initial_separation_coeff=initial_separation_coeff)
        base_energy_fn = simulation.get_energy_fn(
            SHELL_VERTEX_RADIUS, spider_leg_diameter,
            spider_head_diameter,
            morse_ii_eps=MORSE_II_EPS,
            morse_leg_eps=morse_leg_eps, morse_head_eps=morse_head_eps,
            morse_ii_alpha=MORSE_II_ALPHA, morse_leg_alpha=morse_leg_alpha,
            morse_head_alpha=morse_head_alpha,
            soft_eps=SOFT_EPS, tmp_both_shapes)
        leg_energy_fn = leg.get_leg_energy_fn(soft_eps, (spider_leg_diameter/2 + SHELL_VERTEX_RADIUS), shape, shape_species)
        energy_fn = lambda body: base_energy_fn(body) + leg_energy_fn(body)


        # (ii) Do the thing
        fin_states = mapped_sim_fn(params, batch_keys)
        return avg_loss_fn(fin_states), FIXME, 0.0







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

    print(f"About to constrain our optimization...")
    train(args)














    obj_jit = mipopt.ObjectiveWrapper(jit(problem))
    grad_reverse = mipopt.GradWrapper(jit(jacrev(problem, argnums=0)))
    # grad_reverse = mipopt.GradWrapper(jit(grad(problem, argnums=0)))

    max_iter = 1000
    options = {'max_iter': max_iter,
               'disp': 5, 'tol': 1e-12,
               'print_timing_statistics': 'yes'} #,'acceptable_constr_viol_tol':1e-1,'acceptable_obj_change_tol':1e-3}


    cons = [{'type': 'eq', 'fun': obj_jit.const, 'jac': grad_reverse.const},
            {'type': 'ineq', 'fun': obj_jit.ineqconst, 'jac': grad_reverse.ineqconst}]


    bounds = [
        [onp.NINF, onp.inf],
        [onp.NINF, onp.inf],
        [onp.NINF, onp.inf],
    ]

    traj_iter = None
    # parameters = jnp.array([1.0, 2.0, 7.0])
    parameters = jnp.array([1.0, 2.0, 3.0])
    res, trajectory, objective_list, grad_list = mipopt.minimize_ipopt(
        obj_jit, x0=parameters,
        jac=grad_reverse, constraints=cons,
        bounds=bounds, options=options, traj_iter=traj_iter
    )
    pdb.set_trace()
