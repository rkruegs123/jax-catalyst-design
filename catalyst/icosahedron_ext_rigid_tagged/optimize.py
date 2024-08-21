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

from catalyst.icosahedron_ext_rigid_tagged.complex import Complex, PENTAPOD_LEGS, BASE_LEGS
from catalyst.icosahedron_ext_rigid_tagged.loss import get_loss_fn
from catalyst.icosahedron_ext_rigid_tagged.simulation import simulation
import catalyst.icosahedron_ext_rigid_tagged.utils as utils

from jax.config import config
config.update('jax_enable_x64', True)



def run(args):

    batch_size = args['batch_size']
    n_steps = args['n_steps']
    dt = args['dt']
    key_seed = args['key_seed']
    kT = args['temperature']
    initial_separation_coefficient = args['init_separate']
    gamma = args['gamma']
    vertex_to_bind_idx = args['vertex_to_bind']
    release_coeff = args['release_coeff']

    init_base_particle_radius = args['init_base_particle_radius']
    init_spider_head_radius = args['init_spider_head_radius']
    init_attr_site_radius = args['init_attr_site_radius']
    leg_radius = args['leg_radius']

    n_iters = args['n_iters']
    lr = args['lr']

    init_log_head_eps = args['init_log_head_eps']
    init_alpha = args['init_alpha']
    init_head_height = args['init_head_height']
    init_rel_attr_pos = args['init_rel_attr_pos']
    vis_frame_rate = args['vis_frame_rate']

    output_basedir = args['output_basedir']

    min_head_radius = args['min_head_radius']
    perturb_init_head_eps = args['perturb_init_head_eps']

    if perturb_init_head_eps:
        init_log_head_eps += onp.random.normal(0.0, 0.1)

    assert(n_steps % vis_frame_rate == 0)
    num_frames = n_steps // vis_frame_rate
    max_num_frames = 50
    if num_frames > max_num_frames:
        raise RuntimeError(f"The number of frames to be saved per trajectory must be less than {max_num_frames} for computational efficiency")

    output_basedir = Path(output_basedir)
    if not output_basedir.exists():
        raise RuntimeError(f"No output directory exists at location: {output_basedir}")

    run_name = args['run_name']
    assert(run_name is not None)

    run_dir = output_basedir / run_name
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
    state_loss_fn, state_loss_terms_fn = get_loss_fn(displacement_fn, vertex_to_bind_idx)
    spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])

    def loss_fn(params, key):

        complex_info = Complex(
            initial_separation_coefficient, vertex_to_bind_idx,
            displacement_fn, shift_fn,
            params['spider_base_radius'], params['spider_head_height'],
            params['spider_base_particle_radius'],
            jnp.clip(params['spider_attr_particle_pos_norm'], 0.0, 1.0),
            params['spider_attr_site_radius'],
            jnp.max(jnp.array([min_head_radius, params['spider_head_particle_radius']])),
            spider_point_mass=1.0, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs,
            spider_leg_radius=leg_radius
        )

        complex_energy_fn, vertex_energy_fn = complex_info.get_energy_fn(
            morse_attr_eps=jnp.exp(params['log_morse_attr_eps']),
            morse_attr_alpha=params['morse_attr_alpha'],
            morse_r_onset=params['morse_r_onset'], morse_r_cutoff=params['morse_r_cutoff']
        )
        fin_state, traj = simulation(complex_info, complex_energy_fn, n_steps, gamma, kT, shift_fn, dt, key)
        extract_loss = state_loss_fn(fin_state, params, complex_info)

        release_loss = -vertex_energy_fn(traj[-1]) * release_coeff

        loss = release_loss + extract_loss

        return loss, (traj, extract_loss, release_loss)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    batched_grad_fn = jit(vmap(grad_fn, in_axes=(None, 0)))


    optimizer = optax.adam(lr)
    params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': init_head_height,
        'spider_base_particle_radius': init_base_particle_radius,
        'spider_head_particle_radius': init_spider_head_radius,
        'spider_attr_site_radius': init_attr_site_radius,

        # catalyst energy
        'log_morse_attr_eps': init_log_head_eps,
        'morse_attr_alpha': init_alpha,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0,

        'spider_attr_particle_pos_norm': init_rel_attr_pos
    }
    opt_state = optimizer.init(params)

    loss_path = run_dir / "loss.txt"
    extract_loss_path = run_dir / "extract_loss.txt"
    release_loss_path = run_dir / "release_loss.txt"
    loss_terms_path = run_dir / "loss_terms.txt"
    losses_path = run_dir / "losses.txt"
    std_path = run_dir / "std.txt"
    grad_path = run_dir / "grads.txt"
    avg_grad_path = run_dir / "avg_grads.txt"
    params_path = run_dir / "params_per_iter.txt"

    all_avg_losses = list()
    all_params = list()

    for i in tqdm(range(n_iters)):
        print(f"\nIteration: {i}")
        iter_key = keys[i]
        batch_keys = random.split(iter_key, batch_size)
        start = time.time()
        (losses, (trajs, extract_losses, release_losses)), grads = batched_grad_fn(params, batch_keys)
        end = time.time()

        avg_grads = {k: jnp.mean(grads[k], axis=0) for k in grads}
        updates, opt_state = optimizer.update(avg_grads, opt_state)

        with open(std_path, "a") as f:
            f.write(f"{onp.std(losses)}\n")
        with open(losses_path, "a") as f:
            f.write(f"{losses}\n")
        avg_loss = onp.mean(losses)
        all_avg_losses.append(avg_loss)
        with open(loss_path, "a") as f:
            f.write(f"{avg_loss}\n")
        with open(extract_loss_path, "a") as f:
            f.write(f"{onp.mean(extract_losses)}\n")
        with open(release_loss_path, "a") as f:
            f.write(f"{onp.mean(release_losses)}\n")
        with open(grad_path, "a") as f:
            f.write(str(grads) + '\n')

        avg_grads_str = f"\nIteration {i}:\n"
        for param_name, param_avg_grad in avg_grads.items():
            avg_grads_str += f"- {param_name}: {param_avg_grad}\n"
        with open(avg_grad_path, "a") as f:
            f.write(avg_grads_str)

        iter_params_str = f"\nIteration {i}:\n"
        all_params.append(params)
        for param_name, param_val in params.items():
            iter_params_str += f"- {param_name}: {float(param_val)}\n"
        with open(params_path, "a") as f:
            f.write(iter_params_str + '\n')

        # Save a representative trajectory to an injavis-compatible .pos file
        ## note: last index will be 1 higher than the true last index, so it will retrieve the final state (with an interval from the previous frame less than 1)

        min_loss_sample_idx = onp.argmin(losses)
        max_loss_sample_idx = onp.argmax(losses)
        rep_traj_idxs = jnp.arange(0, n_steps+1, vis_frame_rate)
        # rep_traj  = trajs[0][::vis_frame_rate]
        rep_traj  = trajs[min_loss_sample_idx][rep_traj_idxs]
        rep_traj_bad  = trajs[max_loss_sample_idx][rep_traj_idxs]
        rep_complex_info = Complex(
            initial_separation_coefficient, vertex_to_bind_idx,
            displacement_fn, shift_fn,
            params['spider_base_radius'], params['spider_head_height'],
            params['spider_base_particle_radius'],
            jnp.clip(params['spider_attr_particle_pos_norm'], 0.0, 1.0),
            params['spider_attr_site_radius'],
            jnp.max(jnp.array([min_head_radius, params['spider_head_particle_radius']])),
            spider_point_mass=1.0, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs,
            spider_leg_radius=leg_radius
        )
        rep_traj_fname = traj_dir / f"traj_i{i}_b{min_loss_sample_idx}.pos"
        rep_traj_fname_bad = traj_dir / f"traj_i{i}_maxloss_b{max_loss_sample_idx}.pos"
        utils.traj_to_pos_file(rep_traj, rep_complex_info, rep_traj_fname, box_size=30.0)
        utils.traj_to_pos_file(rep_traj_bad, rep_complex_info, rep_traj_fname_bad, box_size=30.0)

        loss_terms_str = f"\nIteration {i}:\n"
        loss_terms_str += f"- Best:\n\t- Loss: {losses[min_loss_sample_idx]}\n"
        loss_terms_str += f"- Worst:\n\t- Loss: {losses[max_loss_sample_idx]}\n"
        loss_terms_str += f"- Average:\n\t- Loss: {onp.mean(losses)}\n"
        with open(loss_terms_path, "a") as f:
            f.write(loss_terms_str + '\n')

        # Update the parameters once we are done logging everyting
        params = optax.apply_updates(params, updates)

    best_iter_idx = onp.argmin(all_avg_losses)
    best_iter_loss = all_avg_losses[best_iter_idx]
    best_iter_params = all_params[best_iter_idx]
    summary_path = run_dir / "summary.txt"
    with open(summary_path, "a") as f:
        f.write(f"Best iteration index: {best_iter_idx}\n")
        f.write(f"Best iteration loss: {best_iter_loss}\n")
        f.write(f"Best iteration params: {best_iter_params}\n")

    return params


def get_argparse():
    parser = argparse.ArgumentParser(description="Optimization for catalyst design")

    parser.add_argument('--batch-size', type=int, default=3, help="Num. batches for each round of gradient descent")
    parser.add_argument('--n-iters', type=int, default=2, help="Num. iterations of gradient descent")
    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    parser.add_argument('--n-steps', type=int, default=1000, help="Num. steps per simulation")
    parser.add_argument('--vertex-to-bind', type=int, default=5, help="Index of vertex to bind")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for optimization")
    parser.add_argument('--init-separate', type=float, default=0.1, help="Initial separation coefficient")

    parser.add_argument('--output-basedir', type=str,
                        default="output",
                        help='Path to base output directory')
    parser.add_argument('-kT', '--temperature', type=float, default=1.0, help="Temperature in kT")
    parser.add_argument('--dt', type=float, default=1e-3, help="Time step")
    parser.add_argument('-g', '--gamma', type=float, default=10.0, help="friction coefficient")
    parser.add_argument('--vis-frame-rate', type=int, default=100,
                        help="The sample rate for saving a representative trajectory from each optimization iteration")
    parser.add_argument('--run-name', type=str, required=True,
                        help='Name of run directory')

    parser.add_argument('--min-head-radius', type=float, default=0.1, help="Minimum radius for spider head")

    parser.add_argument('--init-log-head-eps', type=float, default=3.0,
                        help="Initial value for parameter: log_morse_shell_center_spider_head_eps")
    parser.add_argument('--init-alpha', type=float, default=1.5,
                        help="Initial value for parameter: morse_shell_center_spider_head_alpha")
    parser.add_argument('--init-head-height', type=float, default=10.0,
                        help="Initial value for spider head height")

    parser.add_argument('--perturb-init-head-eps', action='store_true')

    parser.add_argument('--release-coeff', type=float, default=0.0,
                        help="Coefficient for release")
    parser.add_argument('--init-rel-attr-pos', type=float, default=0.5,
                        help="Initial value for rel. attr pos")

    parser.add_argument('--init-base-particle-radius', type=float, default=1.0,
                        help="Initial value for base particle radius")
    parser.add_argument('--init-spider-head-radius', type=float, default=1.0,
                        help="Initial value for spider head particle radius")
    parser.add_argument('--init-attr-site-radius', type=float, default=0.75,
                        help="Initial value for attractive site particle radius")

    parser.add_argument('--leg-radius', type=float, default=0.25,
                        help="The leg radius")
    

    return parser


if __name__ == "__main__":
    parser = get_argparse()
    args = vars(parser.parse_args())

    run(args)
