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
import catalyst.icosahedron_ext_rigid_tagged.rigid_body as rigid_body
from jax import random, jit, vmap, lax
import jax.numpy as jnp


from catalyst.icosahedron_ext_rigid_tagged.complex import Complex, PENTAPOD_LEGS, BASE_LEGS, combined_body_to_injavis_lines




def run(args, sim_params, initial_separation_coeff=0.1, vertex_to_bind_idx=5):

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

    dt = 1e-3
    kT = 1.0
    gamma = 10
    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
    key = random.PRNGKey(0)

    displacement_fn, shift_fn = space.free()
    spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])

    min_head_radius = 0.1
    complex_ = Complex(
        initial_separation_coeff=initial_separation_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
        displacement_fn=displacement_fn, shift_fn=shift_fn,
        spider_base_radius=sim_params['spider_base_radius'], spider_head_height=sim_params['spider_head_height'],
        spider_base_particle_radius=sim_params['spider_base_particle_radius'],
        jnp.clip(params['spider_attr_particle_pos_norm'], 0.0, 1.0),
        params['spider_attr_site_radius'],
        jnp.max(jnp.array([min_head_radius, params['spider_head_particle_radius']])),
        spider_point_mass=1.0, spider_mass_err=1e-6,
        spider_bond_idxs=spider_bond_idxs
    )


    combined_body, base_energy_fn, leg_energy_fn = complex_.get_extracted_rb_info(
        morse_shell_center_spider_head_eps=jnp.exp(sim_params['log_morse_attr_eps']),
        morse_shell_center_spider_head_alpha=sim_params['morse_attr_alpha'],
        morse_r_onset=sim_params['morse_r_onset'],
        morse_r_cutoff=sim_params['morse_r_cutoff']
    )
    init_energy = base_energy_fn(combined_body)
    base_energy_fn = jit(base_energy_fn)

    mass = complex_.shape.mass(onp.array([0, 1]))


    # Do WHAM
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
        spider_body = R[-1]
        vertex_body = R[0]
        spider_body_pos = complex_.spider_info.get_body_frame_positions(spider_body)
        # head_pos = spider_body_pos[-1]

        attr_site_pos = spider_body_pos[5:10]
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


    plt.tight_layout()
    plt.savefig("all_hist.png")
    plt.clf()


    for t_idx in range(num_centers):
        center = bin_centers[t_idx]
        traj_ops = all_traj_order_params[t_idx]

        sns.kdeplot(traj_ops, label=f"{center}")


    plt.tight_layout()
    plt.savefig("all_kde.png")
    plt.clf()


    for t_idx in range(num_centers-1):
        if not all_traj_order_params[t_idx].max() > all_traj_order_params[t_idx+1].min():
            print(f"WARNING: no overlap between windows {t_idx} and {t_idx+1}")


    pdb.set_trace()


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

    init_head_height = 10.0
    init_log_head_eps = 4.0
    init_alpha = 1.0

    params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': init_head_height,
        'spider_base_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,
        'spider_attr_particle_pos_norm': 0.5,
        'spider_attr_site_radius': 0.3,

        # catalyst energy
        'log_morse_attr_eps': init_log_head_eps,
        'morse_attr_alpha': init_alpha,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0
    }

    run(args, params)
