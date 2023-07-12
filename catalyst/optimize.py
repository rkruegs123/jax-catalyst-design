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

from catalyst.complex_getter import ComplexInfo
from catalyst.loss import get_loss_fn
from catalyst.simulation import simulation


def optimize(args):
    batch_size = args['batch_size']
    n_iters = args['n_iters']
    n_steps = args['n_steps']
    data_dir = args['data_dir']
    lr = args['lr']
    dt = args['dt']
    key_seed = args['key_seed']
    kT = args['temperature']
    initial_separation_coefficient = args['init_separate']
    gamma = args['gamma']
    vertex_to_bind_idx = args['vertex_to_bind']
    use_abduction_loss = args['use_abduction_loss']
    use_stable_shell_loss = args['use_stable_shell_loss']

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise RuntimeError(f"No data directory exists at location: {data_dir}")

    run_name = f"optimize_n{n_steps}_i{n_iters}_b{batch_size}_k{key_seed}_lr{lr}_kT{kT}"
    run_dir = data_dir / run_name
    print(f"Making directory: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"

    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    key = random.PRNGKey(key_seed)
    keys = random.split(key, n_iters)

    displacement_fn, shift_fn = space.free()
    complex_loss_fn = get_loss_fn(
        displacement_fn, vertex_to_bind_idx,
        use_abduction=use_abduction_loss, use_stable_shell=use_stable_shell_loss)

    def loss_fn(params, key):
        complex_info = ComplexInfo(
            initial_separation_coefficient, vertex_to_bind_idx, displacement_fn,
            params['spider_base_radius'], params['spider_head_height'],
            params['spider_base_particle_radius'], params['spider_head_particle_radius'],
            spider_point_mass=1.0, spider_mass_err=1e-6
        )

        complex_energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_base_eps=params['morse_shell_center_spider_base_eps'],
            morse_shell_center_spider_base_alpha=params['morse_shell_center_spider_base_alpha'],
            morse_shell_center_spider_head_eps=jnp.exp(params['log_morse_shell_center_spider_head_eps']),
            morse_shell_center_spider_head_alpha=params['morse_shell_center_spider_head_alpha']
        )
        fin_state, traj = simulation(complex_info, complex_energy_fn, n_steps, gamma, kT, shift_fn, dt, key)
        return complex_loss_fn(fin_state)
    grad_fn = value_and_grad(loss_fn)
    batched_grad_fn = jit(vmap(grad_fn, in_axes=(None, 0)))


    optimizer = optax.adam(lr)
    # FIXME: assume that we start with a fixed set of parameters for now
    params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': 6.0,
        'spider_base_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,

        # catalyst energy
        'morse_shell_center_spider_base_eps': 2.5,
        'log_morse_shell_center_spider_head_eps': 9.21, # ln(10000.0)
        'morse_shell_center_spider_base_alpha': 1.0,
        'morse_shell_center_spider_head_alpha': 4.5
    }
    opt_state = optimizer.init(params)

    loss_path = run_dir / "loss.txt"
    losses_path = run_dir / "losses.txt"
    std_path = run_dir / "std.txt"
    grad_path = run_dir / "grads.txt"
    params_path = run_dir / "params_per_iter.txt"

    for i in tqdm(range(n_iters)):
        print(f"\nIteration: {i}")
        iter_key = keys[i]
        batch_keys = random.split(iter_key, batch_size)
        start = time.time()
        vals, grads = batched_grad_fn(params, batch_keys)
        end = time.time()

        avg_grads = {k: jnp.mean(grads[k], axis=0) for k in grads}
        updates, opt_state = optimizer.update(avg_grads, opt_state)
        params = optax.apply_updates(params, updates)

        with open(std_path, "a") as f:
            f.write(f"{onp.std(vals)}\n")
        with open(losses_path, "a") as f:
            f.write(f"{vals}\n")
        with open(loss_path, "a") as f:
            f.write(f"{onp.mean(vals)}\n")
        with open(grad_path, "a") as f:
            f.write(str(grads) + '\n')
        with open(params_path, "a") as f:
            params_to_print = {k: float(v) for k, v in params.items()}
            f.write(str(params_to_print) + '\n')

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
                        default="data/",
                        help='Path to base data directory')
    parser.add_argument('-kT', '--temperature', type=float, default=2.0, help="Temperature in kT")
    parser.add_argument('--dt', type=float, default=1e-3, help="Time step")
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help="friction coefficient")
    parser.add_argument('--use-abduction-loss', action='store_true')
    parser.add_argument('--use-stable-shell-loss', action='store_true')

    return parser


if __name__ == "__main__":
    parser = get_argparse()
    args = vars(parser.parse_args())

    optimize(args)
