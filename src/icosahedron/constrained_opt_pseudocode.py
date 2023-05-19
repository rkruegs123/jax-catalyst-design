import modified_ipopt as mipopt



def loss(trajectory):
    return
def equality_constraint(trajectory):
    return


def inequality_constraints(trajectory):
    return


def problem(parameters):
    # loss function returns constraint
    trajectory = Simulate(parameters)

    return loss(trajectory), equality_constraint(trajectory), inequality_constraints(trajectory)


obj_jit = mipopt.ObjectiveWrapper(jax.jit(problem))
grad_reverse = mipopt.GradWrapper(jax.jit(jax.jacrev(problem, argnums=0)))

options = {'max_iter': max_iter,
           'disp': 5, 'tol':1e-12,
           'print_timing_statistics':'yes'} #,'acceptable_constr_viol_tol':1e-1,'acceptable_obj_change_tol':1e-3}


cons = [{'type': 'eq', 'fun': obj_jit.const, 'jac': grad_reverse.const},
        {'type': 'ineq', 'fun': obj_jit.ineqconst, 'jac': grad_reverse.ineqconst}]


bounds = [[0, np.inf], [0.2, 1]]

parameters = [1.0, 2.0]
res, trajectory, objective_list, grad_list = mipopt.minimize_ipopt(
    obj_jit, x0=parameters,
    jac=grad_reverse, constraints=cons,
    bounds=bounds, options=options, traj_iter=traj_iter
)
