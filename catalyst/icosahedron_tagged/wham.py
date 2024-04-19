import numpy as onp
import pdb
import decimal
from tqdm import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import subprocess

from jax_md import space, simulate
import catalyst.icosahedron_tagged.rigid_body as rigid_body
from jax import random, jit, vmap, lax
import jax.numpy as jnp

from catalyst.icosahedron_tagged.complex import Complex, combined_body_to_injavis_lines




def run(args, params):


    wham_basedir = Path(args['wham_basedir'])
    assert(wham_basedir.exists() and wham_basedir.is_dir())
    wham_exec_path = wham_basedir / "wham" / "wham"
    assert(wham_exec_path.exists())
    wham_tol = args['wham_tol']

    run_name = args['run_name']
    output_basedir = Path(args['output_basedir'])


    # Setup the run directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    run_dir = output_basedir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    wham_dir = run_dir / "wham"
    wham_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)





    initial_separation_coefficient = 5.5
    vertex_to_bind_idx = 5
    dt = 1e-3
    kT = 1.0
    gamma = 10
    key = random.PRNGKey(0)
    n_steps = 250
    init_log_head_eps = 4.0
    init_alpha = 1.0

    displacement_fn, shift_fn = space.free()

    complex_ = Complex(
        initial_separation_coeff=initial_separation_coefficient,
        vertex_to_bind_idx=vertex_to_bind_idx,
        displacement_fn=displacement_fn, shift_fn=shift_fn,
        spider_base_radius=params['spider_base_radius'],
        spider_head_height=params['spider_head_height'],
        spider_base_particle_radius=params['spider_base_particle_radius'],
        spider_attr_particle_radius=params['spider_attr_particle_radius'],
        spider_head_particle_radius=params['spider_head_particle_radius'],
        spider_point_mass=1.0, spider_mass_err=1e-6,
        rel_attr_particle_pos=0.75
    )

    combined_body, base_energy_fn = complex_.get_extracted_rb_info(
        morse_ii_eps=10.0, morse_ii_alpha=5.0,
        morse_attr_eps=350.0, morse_attr_alpha=2.0, morse_r_onset=12.0, morse_r_cutoff=14.0,
        soft_eps=10000.0)
    combined_shape_species = onp.array([0, 1, 1, 1, 1, 1])
    mass = complex_.shape.mass(combined_shape_species)
    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))


    # Dummy simulation
    run_dummy = False
    if run_dummy:
        init_fn, step_fn = simulate.nvt_langevin(base_energy_fn, shift_fn, dt,
                                                 kT, gamma=gamma_rb)
        step_fn = jit(step_fn)

        state = init_fn(key, combined_body, mass=mass)
        n_steps = 10000
        traj = list()
        for i in tqdm(range(n_steps)):
            state = step_fn(state)
            traj.append(state.position)

        traj_injavis_lines = list()
        n_vis_states = len(traj)
        box_size = 30.0
        vis_every = 250
        for i in tqdm(range(n_vis_states), desc="Generating injavis output"):
            if i % vis_every == 0:
                s = traj[i]
                traj_injavis_lines += combined_body_to_injavis_lines(complex_, s, box_size=box_size)[0]

        with open("test_wham_combined_sim.pos", 'w+') as of:
            of.write('\n'.join(traj_injavis_lines))





    # Start stuff for WHAM

    """
    min_dist = 2.5
    max_dist = 10.0
    # min_dist = 2.5
    # max_dist = 4.0
    dist_diff = max_dist - min_dist
    bin_width = 0.5

    dec_difference = decimal.Decimal(str(max_dist)) - decimal.Decimal(str(min_dist))
    assert(dec_difference % decimal.Decimal(str(bin_width)) == decimal.Decimal('0.0'))
    num_centers = int(dec_difference // decimal.Decimal(str(bin_width)))
    bin_centers = onp.array([min_dist+i*bin_width for i in range(num_centers)])
    bin_centers = jnp.array(bin_centers)
    """
    # bin_centers = list(onp.linspace(2.5, 4.75, 250)) + list(onp.linspace(5, 10, 20))
    """
    bin_centers = list(onp.linspace(2.5, 3.46, 25)) + list(onp.linspace(3.46, 3.47, 100)) + list(onp.linspace(3.5, 5, 25)) + list(onp.linspace(5, 10, 20))
    num_centers = len(bin_centers)
    bin_centers= jnp.array(bin_centers)

    k_biases = list()
    for center in bin_centers:
        if center < 3.25:
            # k_biases.append(1e4)
            k_biases.append(5e3)
        elif center < 4.75:
            k_biases.append(5e3)
        else:
            k_biases.append(1e3)
    k_biases = jnp.array(k_biases)
    """

    min_center = 2.5
    max_center = 6.0
    bin_centers = list(onp.linspace(min_center, max_center, 400))
    num_centers = len(bin_centers)
    bin_centers= jnp.array(bin_centers)
    n_bins = len(bin_centers)

    k_biases = list()
    k_bias = 5e4
    for center in bin_centers:
        k_biases.append(k_bias)
    k_biases = jnp.array(k_biases)


    def order_param_fn(R):
        leg_rbs = R[-5:] # the spider
        spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
        attr_site_pos = spider_space_frame_pos[1::3]

        vertex_com = R[0].center
        disps = vmap(displacement_fn, (None, 0))(vertex_com, attr_site_pos)
        drs = vmap(space.distance)(disps)
        return jnp.mean(drs)
    get_traj_order_params = vmap(order_param_fn)


    def _harmonic_bias(op, center_idx):
        center = bin_centers[center_idx]
        k_bias = k_biases[center_idx]
        return 1/2*k_bias * (center - op)**2

    def harmonic_bias(R, center_idx):
        op = order_param_fn(R)
        return _harmonic_bias(op, center_idx)



    n_eq_steps = 5000
    def eq_fn(R_init, center_idx, eq_key):

        @jit
        def energy_fn(R):
            bias_val = harmonic_bias(R, center_idx)
            base_val = base_energy_fn(R)
            return bias_val + base_val

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt,
                                                 kT, gamma=gamma_rb)
        step_fn = jit(step_fn)
        init_state = init_fn(eq_key, R_init, mass=mass)

        eq_state = lax.fori_loop(0, n_eq_steps, lambda i, state: step_fn(state), init_state)
        return eq_state.position


    key, eq_key = random.split(key)
    eq_keys = random.split(eq_key, num_centers)
    start = time.time()
    # R_eq = vmap(eq_fn, (None, 0, 0))(combined_body, bin_centers, eq_keys)
    R_eq = vmap(eq_fn, (None, 0, 0))(combined_body, jnp.arange(num_centers), eq_keys)
    end = time.time()
    eq_time = end - start


    traj_injavis_lines = list()
    box_size = 30.0
    for i in tqdm(range(num_centers), desc="Generating injavis output"):
        center = bin_centers[i]
        s = R_eq[i]
        s_op = order_param_fn(s)
        print(f"Target dist: {center}")
        print(f"Eq dist: {s_op}")
        traj_injavis_lines += combined_body_to_injavis_lines(complex_, s, box_size=box_size)[0]

    with open("wham_eq_states.pos", 'w+') as of:
        of.write('\n'.join(traj_injavis_lines))



    sample_every = 100
    n_sample_states_per_sim = 50
    def sim_fn(R_init, center_idx, sim_key):
        def energy_fn(R):
            bias_val = harmonic_bias(R, center_idx)
            base_val = base_energy_fn(R)
            return bias_val + base_val

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt,
                                                 kT, gamma=gamma_rb)
        step_fn = jit(step_fn)
        init_state = init_fn(eq_key, R_init, mass=mass)

        @jit
        def scan_fn(state, idx):
            state = lax.fori_loop(0, sample_every,
                                  lambda i, state: step_fn(state), state)
            return state, state.position

        fin_state, traj = lax.scan(scan_fn, init_state, jnp.arange(n_sample_states_per_sim))

        return traj

    key, sim_key = random.split(key)
    sim_keys = random.split(sim_key, num_centers)
    start = time.time()
    # all_traj = vmap(sim_fn, (0, 0, 0))(R_eq, bin_centers, sim_keys)
    all_traj = vmap(sim_fn, (0, 0, 0))(R_eq, jnp.arange(num_centers), sim_keys)
    end = time.time()
    sim_time = end - start



    # Compute order parameters and plot histograms
    all_traj_order_params = list()
    all_traj_bias_vals = list()
    for t_idx in range(num_centers):
        traj = all_traj[t_idx]

        traj_ops = get_traj_order_params(traj)
        all_traj_order_params.append(traj_ops)

        traj_bias_vals = vmap(_harmonic_bias, (0, None))(traj_ops, t_idx)
        all_traj_bias_vals.append(traj_bias_vals)



    all_traj_order_params = jnp.array(all_traj_order_params)
    all_traj_bias_vals = onp.array(all_traj_bias_vals)

    for t_idx in range(num_centers):
        center = bin_centers[t_idx]
        traj_ops = all_traj_order_params[t_idx]

        sns.histplot(traj_ops, label=f"{center}")

    # plt.legend()
    plt.tight_layout()
    plt.savefig("all_hist.png")
    plt.clf()

    for t_idx in range(num_centers):
        center = bin_centers[t_idx]
        traj_ops = all_traj_order_params[t_idx]

        sns.kdeplot(traj_ops, label=f"{center}")

    # plt.legend()
    plt.tight_layout()
    plt.savefig("all_kde.png")
    plt.clf()

    for t_idx in range(num_centers-1):
        if not all_traj_order_params[t_idx].max() > all_traj_order_params[t_idx+1].min():
            print(f"WARNING: no overlap between windows {t_idx} and {t_idx+1}")

    def write_wham_timeseries(times, ops, fpath):
        lines = list()
        for t, op in zip(times, ops):
            lines.append(f"{t}\t{op}\n")

        with open(fpath, "w+") as f:
            f.writelines(lines)

        return

    # Do WHAM

    ## Write the timeseries files
    fpaths = list()
    for t_idx in range(num_centers):
        center = bin_centers[t_idx]
        traj_ops = all_traj_order_params[t_idx]
        n_ops = len(traj_ops)

        times = [sample_every*(i+1) for i in range(n_ops)]
        fpath = wham_dir / f"timeseries_c{center}.txt"
        fpaths.append(fpath)

        write_wham_timeseries(times, traj_ops, fpath)

    ## Write the metadata file
    metadata_lines = list()
    for t_idx in range(num_centers):
        fpath = fpaths[t_idx]
        center = bin_centers[t_idx]
        t_line = f"{fpath}\t{center}\t{k_bias}\n"
        metadata_lines.append(t_line)

    metadata_path = wham_dir / "metadata.txt"
    with open(metadata_path , "w+") as f:
        f.writelines(metadata_lines)


    ## Run the WHAM executable
    wham_out_path = wham_dir / "analysis.txt"
    start = time.time()
    p = subprocess.Popen([wham_exec_path, str(min_center), str(max_center), str(n_bins),
                          str(wham_tol), str(kT), str(0), metadata_path, wham_out_path])
    p.wait()
    end = time.time()
    wham_time = end - start
    rc = p.returncode
    if rc != 0:
        raise RuntimeError(f"WHAM analysis failed with error code: {rc}")
    with open(run_dir / "summary.txt", "a") as f:
        f.write(f"WHAM time: {wham_time}\n")


    ## Read in the free energies (per R) and free energies (per bin)
    with open(wham_out_path, "r") as f:
        wham_lines = f.readlines()
    pmf_lines = wham_lines[:n_bins+1] # really free energies
    hist_fe_lines = wham_lines[n_bins+1:]


    ### pmf data (really free energies)
    assert(pmf_lines[0][:5] == "#Coor")
    header = pmf_lines[0][1:].split()
    assert(header == ["Coor", "Free", "+/-", "Prob", "+/-"])
    all_ex_ops = list()
    all_ex_fes = list()
    all_ex_probs = list()
    for line in pmf_lines[1:]:
        assert(line[0] != "#")
        tokens = line.split()

        op = float(tokens[0])
        all_ex_ops.append(op)

        fe = float(tokens[1])
        all_ex_fes.append(fe)

        prob = float(tokens[3])
        all_ex_probs.append(prob)
    all_ex_ops = onp.array(all_ex_ops)
    all_ex_fes = onp.array(all_ex_fes)
    all_ex_probs = onp.array(all_ex_probs)

    assert(hist_fe_lines[0][:7] == "#Window")
    header = hist_fe_lines[0][1:].split()
    assert(header == ["Window", "Free", "+/-"])
    bin_idxs = list()
    bin_fes = list()
    for line in hist_fe_lines[1:]:
        assert(line[0] == "#")
        tokens = line[1:].split()

        bin_idx = int(tokens[0])
        bin_idxs.append(bin_idx)

        bin_fe = float(tokens[1])
        bin_fes.append(bin_fe)
    bin_idx = onp.array(bin_idxs)
    bin_fes = onp.array(bin_fes)


    plt.plot(all_ex_ops, all_ex_fes)
    plt.xlabel("OP")
    plt.ylabel("Free Energy (kT)")
    plt.tight_layout()
    plt.savefig(run_dir / "fe.png")
    plt.clf()


def get_parser():
    parser = argparse.ArgumentParser(description="Do WHAM for a rigid spider")

    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--output-basedir', type=str, default="output/", help="Output base directory")

    parser.add_argument('--wham-tol', type=float,
                        # default=1e-5,
                        default=0.25,
                        help="Tolerance for free energy convergence.")
    parser.add_argument('--wham-basedir', type=str, help='Base directory for WHAM executable',
                        default="/n/home10/rkrueger/wham")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    params = {
        'log_morse_attr_eps': 4.05178597,
        'morse_attr_alpha': 1.31493759,
        'morse_r_cutoff': 12.,
        'morse_r_onset': 10.,
        'spider_attr_particle_radius': 1.07176809,
        'spider_base_particle_radius': 1.03204563,
        'spider_base_radius': 4.49771056,
        'spider_head_height': 10.0,
        'spider_head_particle_radius': 0.330227
    }

    run(args, params)
