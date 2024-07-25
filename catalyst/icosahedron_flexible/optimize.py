import argparse
from pathlib import Path
import optax
from tqdm import tqdm
import time
import pdb
import numpy as onp
import pprint

from jax import random, grad, jit, vmap, value_and_grad
import jax.numpy as jnp

from jax_md import space

from catalyst.icosahedron_flexible.complex import Complex
from catalyst.icosahedron_flexible.simulation import simulation
from catalyst.icosahedron_flexible.loss import get_loss_fn
import catalyst.icosahedron_flexible.utils as utils

from jax.config import config
config.update('jax_enable_x64', True)



def run(args):
    batch_size = args['batch_size']
    n_iters = args['n_iters']
    n_steps = args['n_steps']
    init_log_attr_eps = args['init_log_attr_eps']
    init_alpha = args['init_alpha']
    init_head_height = args['init_head_height']
    data_dir = args['data_dir']
    lr = args['lr']
    dt = args['dt']
    key_seed = args['key_seed']
    kT = args['temperature']
    initial_separation_coefficient = args['init_separate']
    gamma = args['gamma']
    vertex_to_bind_idx = args['vertex_to_bind']
    init_rel_attr_pos = args['init_rel_attr_pos']
    min_head_radius = args['min_head_radius']
    spider_leg_radius = args['spider_leg_radius']

    opt_log_leg_spring_eps = args['opt_log_leg_spring_eps']
    init_log_leg_spring_eps = args['init_log_leg_spring_eps']


    init_particle_radii = args['init_particle_radii']
    init_attr_site_radius = args['init_attr_site_radius']

    use_abduction_loss = args['use_abduction_loss']
    use_remaining_shell_vertices_loss = args['use_remaining_shell_vertices_loss']
    remaining_shell_vertices_loss_coeff = args['remaining_shell_vertices_loss_coeff']
    use_release_loss = args['use_release_loss']
    release_loss_coeff = args['release_loss_coeff']
    assert(not use_release_loss)

    vis_frame_rate = args['vis_frame_rate']
    assert(n_steps % vis_frame_rate == 0)

    displacement_fn, shift_fn = space.free()

    init_state_loss_fn, init_state_loss_terms_fn = get_loss_fn(
        displacement_fn, vertex_to_bind_idx,
        use_abduction=False,
        use_remaining_shell_vertices_loss=use_remaining_shell_vertices_loss,
        remaining_shell_vertices_loss_coeff=remaining_shell_vertices_loss_coeff
    )

    fin_state_loss_fn, fin_state_loss_terms_fn = get_loss_fn(
        displacement_fn, vertex_to_bind_idx,
        use_abduction=use_abduction_loss,
        use_remaining_shell_vertices_loss=False
    )


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



    # Dummy loss function for now
    def loss_fn(params, key):

        clipped_head_radius = jnp.max(jnp.array([min_head_radius, params['spider_head_particle_radius']]))

        if opt_log_leg_spring_eps:
            leg_eps = jnp.exp(params['log_leg_spring_eps'])
        else:
            leg_eps = 100000.

        complex_ = Complex(
            initial_separation_coeff=initial_separation_coefficient,
            vertex_to_bind_idx=vertex_to_bind_idx,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=params['spider_base_radius'],
            spider_head_height=params['spider_head_height'],
            spider_base_particle_radius=params['spider_base_particle_radius'],
            spider_attr_particle_radius=params['spider_attr_particle_radius'],
            spider_head_particle_radius=clipped_head_radius,
            spider_point_mass=1.0, spider_mass_err=1e-6,
            rel_attr_particle_pos=jnp.clip(params['rel_attr_pos'], 0.0, 1.0),
            bond_radius=spider_leg_radius,
            leg_spring_eps=leg_eps
        )

        complex_energy_fn = complex_.get_energy_fn(
            morse_attr_eps=jnp.exp(params['log_morse_attr_eps']),
            morse_attr_alpha=params['morse_attr_alpha'],
            morse_r_onset=params['morse_r_onset'],
            morse_r_cutoff=params['morse_r_cutoff'])

        fin_state, traj = simulation(complex_, complex_energy_fn,
                                     n_steps, gamma, kT, shift_fn, dt, key)
        init_state = traj[0]

        # extract_loss = state_loss_fn(fin_state, params, complex_)
        _, remaining_energy_loss, _ = init_state_loss_terms_fn(init_state, params, complex_)
        extract_loss, _, _ = fin_state_loss_terms_fn(fin_state, params, complex_)
        loss = extract_loss + remaining_energy_loss
        return loss, (traj, extract_loss, remaining_energy_loss)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    batched_grad_fn = jit(vmap(grad_fn, in_axes=(None, 0)))

    optimizer = optax.adam(lr)
    init_log_leg_spring_eps_vals = jnp.array([init_log_leg_spring_eps for _ in range(10)])
    params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': init_head_height,
        'spider_base_particle_radius': init_particle_radii,
        'spider_attr_particle_radius': init_attr_site_radius,
        'spider_head_particle_radius': init_particle_radii,

        # catalyst energy
        'log_morse_attr_eps': init_log_attr_eps,
        'morse_attr_alpha': init_alpha,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0,

        'rel_attr_pos': init_rel_attr_pos,

        'log_leg_spring_eps': init_log_leg_spring_eps_vals
    }
    opt_state = optimizer.init(params)

    loss_path = run_dir / "loss.txt"
    loss_terms_path = run_dir / "loss_terms.txt"
    extract_loss_path = run_dir / "extract_loss.txt"
    remaining_energy_loss_path = run_dir / "remaining_energy_loss.txt"
    grads_path = run_dir / "grads.txt"
    losses_path = run_dir / "losses.txt"
    times_path = run_dir / "times.txt"
    iter_params_path = run_dir / "iter_params.txt"

    all_avg_losses = list()
    all_params = list()

    for i in tqdm(range(n_iters)):
        print(f"\nIteration: {i}")
        iter_key = keys[i]
        batch_keys = random.split(iter_key, batch_size)
        start = time.time()
        (vals, (trajs, extract_losses, remaining_energy_losses)), grads = batched_grad_fn(params, batch_keys)
        end = time.time()

        avg_grads = {k: jnp.mean(grads[k], axis=0) for k in grads}
        updates, opt_state = optimizer.update(avg_grads, opt_state)

        with open(losses_path, "a") as f:
            f.write(f"{vals}\n")
        avg_loss = onp.mean(vals)
        all_avg_losses.append(avg_loss)
        with open(loss_path, "a") as f:
            f.write(f"{avg_loss}\n")
        with open(extract_loss_path, "a") as f:
            f.write(f"{onp.mean(extract_losses)}\n")
        with open(remaining_energy_loss_path, "a") as f:
            f.write(f"{onp.mean(remaining_energy_losses)}\n")
        with open(times_path, "a") as f:
            f.write(f"{end - start}\n")
        all_params.append(params)
        with open(iter_params_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(avg_grads)}\n")



        box_size = 30.0
        min_loss_sample_idx = onp.argmin(vals)
        max_loss_sample_idx = onp.argmax(vals)
        rep_traj_idxs = jnp.arange(0, n_steps+1, vis_frame_rate)
        rep_traj  = trajs[min_loss_sample_idx][rep_traj_idxs]
        rep_traj_bad  = trajs[max_loss_sample_idx][rep_traj_idxs]

        clipped_head_radius = jnp.max(jnp.array([min_head_radius, params['spider_head_particle_radius']]))
        rep_complex_ = Complex(
            initial_separation_coeff=initial_separation_coefficient,
            vertex_to_bind_idx=vertex_to_bind_idx,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=params['spider_base_radius'],
            spider_head_height=params['spider_head_height'],
            spider_base_particle_radius=params['spider_base_particle_radius'],
            spider_attr_particle_radius=params['spider_attr_particle_radius'],
            spider_head_particle_radius=clipped_head_radius,
            spider_point_mass=1.0, spider_mass_err=1e-6,
            rel_attr_particle_pos=jnp.clip(params['rel_attr_pos'], 0.0, 1.0),
            bond_radius=spider_leg_radius
        )

        if i % 10 == 0:
            rep_traj_fname = traj_dir / f"traj_i{i}_b{min_loss_sample_idx}.pos"
            rep_traj_fname_bad = traj_dir / f"traj_i{i}_maxloss_b{max_loss_sample_idx}.pos"
            utils.traj_to_pos_file(rep_traj, rep_complex_, rep_traj_fname, box_size=30.0)
            utils.traj_to_pos_file(rep_traj_bad, rep_complex_, rep_traj_fname_bad, box_size=30.0)


        loss_terms_str = f"\nIteration {i}:\n"
        loss_terms_str += f"- Best:\n\t- Extraction: {extract_losses[min_loss_sample_idx]}\n\t- Remaining Energy: {remaining_energy_losses[min_loss_sample_idx]}\n"
        loss_terms_str += f"- Worst:\n\t- Extraction: {extract_losses[max_loss_sample_idx]}\n\t- Remaining Energy: {remaining_energy_losses[max_loss_sample_idx]}\n"
        loss_terms_str += f"- Average:\n\t- Extraction: {onp.mean(extract_losses)}\n\t- Remaining Energy: {onp.mean(remaining_energy_losses)}"
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

    parser.add_argument('--batch-size', type=int, default=10, help="Num. batches for each round of gradient descent")
    parser.add_argument('--n-iters', type=int, default=5000, help="Num. iterations of gradient descent")
    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    parser.add_argument('--n-steps', type=int, default=1000, help="Num. steps per simulation")
    parser.add_argument('--vertex-to-bind', type=int, default=5, help="Index of vertex to bind")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for optimization")
    parser.add_argument('--init-separate', type=float, default=0.2, help="Initial separation coefficient")
    parser.add_argument('--spider-leg-radius', type=float, default=0.25, help="Spider leg radius")

    parser.add_argument('-d', '--data-dir', type=str,
                        default="data/icosahedron-flexible",
                        help='Path to base data directory')
    parser.add_argument('-kT', '--temperature', type=float, default=1.0, help="Temperature in kT")
    parser.add_argument('--dt', type=float, default=1e-3, help="Time step")
    parser.add_argument('-g', '--gamma', type=float, default=10.0, help="friction coefficient")

    parser.add_argument('--use-abduction-loss', action='store_true')
    parser.add_argument('--use-remaining-shell-vertices-loss', action='store_true')
    parser.add_argument('--remaining-shell-vertices-loss-coeff', type=float, default=1.0,
                        help="Multiplicative scalar for the remaining energy loss term")
    parser.add_argument('--use-release-loss', action='store_true')
    parser.add_argument('--release-loss-coeff', type=float, default=1.0,
                        help="Multiplicative scalar for the release loss term")


    parser.add_argument('--vis-frame-rate', type=int, default=100,
                        help="The sample rate for saving a representative trajectory from each optimization iteration")


    parser.add_argument('--run-name', type=str, nargs='?',
                        help='Name of run directory')

    parser.add_argument('--init-log-attr-eps', type=float, default=4.0,
                        help="Initial value for epsilon value for Morse")
    parser.add_argument('--init-alpha', type=float, default=1.5,
                        help="Initial value for alpha parameter for morsoe potential")

    parser.add_argument('--init-head-height', type=float, default=10.0,
                        help="Initial value for spider head height")
    parser.add_argument('--init-rel-attr-pos', type=float, default=0.5,
                        help="Initial value for rel. attr pos")
    parser.add_argument('--min-head-radius', type=float, default=0.1, help="Minimum radius for spider head")

    parser.add_argument('--init-particle-radii', type=float, default=1.0,
                        help="Initial value for base particle and head radii")
    parser.add_argument('--init-attr-site-radius', type=float, default=0.75,
                        help="Initial value for attractive site particle radius")

    parser.add_argument('--opt-log-leg-spring-eps', action='store_true')
    parser.add_argument('--init-log-leg-spring-eps', type=float, default=10.0,
                        help="Initial value for log of leg spring epsilons")



    return parser


if __name__ == "__main__":
    parser = get_argparse()
    args = vars(parser.parse_args())

    run(args)
