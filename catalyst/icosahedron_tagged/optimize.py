import argparse
from pathlib import Path
import optax
from tqdm import tqdm
import time
import pdb
import numpy as onp

from jax import random, grad, jit, vmap, value_and_grad
import jax.numpy as jnp
import jax.debug

from jax_md import space

from catalyst.icosahedron_tagged.complex_getter import Complex
from catalyst.icosahedron_tagged.simulation import simulation
import catalyst.icosahedron.utils as utils

from jax.config import config
config.update('jax_enable_x64', True)



def optimize(args):
    batch_size = args['batch_size']
    n_iters = args['n_iters']
    n_steps = args['n_steps']
    init_log_head_eps = args['init_log_head_eps']
    init_alpha = args['init_alpha']
    data_dir = args['data_dir']
    lr = args['lr']
    dt = args['dt']
    key_seed = args['key_seed']
    kT = args['temperature']
    initial_separation_coefficient = args['init_separate']
    gamma = args['gamma']
    vertex_to_bind_idx = args['vertex_to_bind']

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise RuntimeError(f"No data directory exists at location: {data_dir}")

    if args['run_name'] is not None:
        run_name = args['run_name']
    else:
        run_name = f"optimize_n{n_steps}_i{n_iters}_b{batch_size}_lr{lr}_kT{kT}_g{gamma}_k{key_seed}"
    run_dir = data_dir / run_name
    print(f"Making directory: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)
    traj_dir = run_dir / "trajs"
    traj_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"

    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    key = random.PRNGKey(key_seed)
    keys = random.split(key, n_iters)

    displacement_fn, shift_fn = space.free()


    # Dummy loss function for now
    def loss_fn(params, key):

        complex_ = Complex(
            initial_separation_coeff=initial_separation_coefficient,
            vertex_to_bind_idx=vertex_to_bind_idx,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=params['spider_base_radius'],
            spider_head_height=params['spider_head_height'],
            spider_base_particle_radius=params['spider_base_particle_radius'],
            spider_attr_particle_radius=params['spider_attr_particle_radius'],
            spider_head_particle_radius=params['spider_head_particle_radius'],
            spider_point_mass=1.0, spider_mass_err=1e-6
        )

        complex_energy_fn = complex_.get_energy_fn(
            morse_attr_eps=params['log_morse_attr_eps'],
            morse_attr_alpha=params['morse_attr_alpha'],
            morse_r_onset=params['morse_r_onset'],
            morse_r_cutoff=params['morse_r_cutoff'])

        fin_state, traj = simulation(complex_, complex_energy_fn,
                                     n_steps, gamma, kT, shift_fn, dt, key)


        return traj[-1]_.center.sum()

    grad_fn = value_and_grad(loss_fn, has_aux=False)
    batched_grad_fn = jit(vmap(grad_fn, in_axes=(None, 0)))


    optimizer = optax.adam(lr)
    params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': 5.0,
        'spider_base_particle_radius': 0.5,
        'spider_attr_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,

        # catalyst energy
        'log_morse_attr_eps': init_log_head_eps,
        'morse_attr_alpha': init_alpha,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0
    }
    opt_state = optimizer.init(params)

    loss_path = run_dir / "loss.txt"
    losses_path = run_dir / "losses.txt"


    for i in tqdm(range(n_iters)):
        print(f"\nIteration: {i}")
        iter_key = keys[i]
        batch_keys = random.split(iter_key, batch_size)
        start = time.time()
        vals, grads = batched_grad_fn(params, batch_keys)
        end = time.time()

        avg_grads = {k: jnp.mean(grads[k], axis=0) for k in grads}
        updates, opt_state = optimizer.update(avg_grads, opt_state)

        with open(losses_path, "a") as f:
            f.write(f"{vals}\n")
        avg_loss = onp.mean(vals)
        all_avg_losses.append(avg_loss)
        with open(loss_path, "a") as f:
            f.write(f"{avg_loss}\n")

        # Update the parameters once we are done logging everyting
        params = optax.apply_updates(params, updates)

    return params


def get_argparse():
    parser = argparse.ArgumentParser(description="Optimization for catalyst design")

    parser.add_argument('--batch-size', type=int, default=3, help="Num. batches for each round of gradient descent")
    parser.add_argument('--n-iters', type=int, default=2, help="Num. iterations of gradient descent")
    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    parser.add_argument('--n-steps', type=int, default=1000, help="Num. steps per simulation")
    parser.add_argument('--vertex-to-bind', type=int, default=5, help="Index of vertex to bind")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for optimization")
    parser.add_argument('--init-separate', type=float, default=0.0, help="Initial separation coefficient")

    parser.add_argument('-d', '--data-dir', type=str,
                        default="data/icosahedron",
                        help='Path to base data directory')
    parser.add_argument('-kT', '--temperature', type=float, default=1.0, help="Temperature in kT")
    parser.add_argument('--dt', type=float, default=1e-3, help="Time step")
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help="friction coefficient")


    parser.add_argument('--run-name', type=str, nargs='?',
                        help='Name of run directory')

    parser.add_argument('--init-log-head-eps', type=float, default=5.5,
                        help="Initial value for parameter: log_morse_shell_center_spider_head_eps")
    parser.add_argument('--init-alpha', type=float, default=1.5,
                        help="Initial value for parameter: morse_shell_center_spider_head_alpha")


    return parser


if __name__ == "__main__":
    parser = get_argparse()
    args = vars(parser.parse_args())

    optimize(args)
