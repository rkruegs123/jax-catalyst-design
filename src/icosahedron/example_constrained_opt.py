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


if __name__ == "__main__":

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
