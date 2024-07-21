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
from catalyst.icosahedron_tagged import utils


def plot_fe(wham_out_path, n_bins, savepath):
    ## Read in the free energies (per R) and free energies (per bin)
    with open(wham_out_path, "r") as f:
        wham_lines = f.readlines()
    pmf_lines = wham_lines[:n_bins+1] # really free energies
    hist_fe_lines = wham_lines[n_bins+1:]

    ### pmf data (really free energies)
    assert(pmf_lines[0][:5] == "#Coor")
    header = pmf_lines[0][1:].split()
    len_header = len(header)
    if len_header == 5:
        assert(header == ["Coor", "Free", "+/-", "Prob", "+/-"])
    elif len_header == 3:
        assert(header == ["Coor", "Free", "+/-"])
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

        if len_header == 5:
            prob = float(tokens[3])
            all_ex_probs.append(prob)
    all_ex_ops = onp.array(all_ex_ops)
    all_ex_fes = onp.array(all_ex_fes)
    all_ex_probs = onp.array(all_ex_probs)

    assert(hist_fe_lines[0][:7] == "#Window")
    header = hist_fe_lines[0][1:].split()
    len_header = len(header)
    if len_header == 3:
        assert(header == ["Window", "Free", "+/-"])
    elif len_header == 2:
        assert(header == ["Window", "Free"])
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
    plt.savefig(savepath)
    plt.clf()

    return



def run(args, sim_params):

    min_head_radius = args['min_head_radius']
    spider_leg_radius = args['spider_leg_radius']
    initial_separation_coeff = args['init_sep_coeff']
    n_bins = args['n_bins']
    op_name = args['op_name']

    use_split_point = args['use_split_point']
    split_point = args['split_point']

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

    sim_params_str = ""
    for k, v in sim_params.items():
        sim_params_str += f"{k}: {v}\n"
    with open(run_dir / "sim_params.txt", "w+") as f:
        f.write(sim_params_str)

    vertex_to_bind_idx = 5

    dt = args['dt']
    kT = 1.0
    gamma = 10
    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
    key = random.PRNGKey(0)

    displacement_fn, shift_fn = space.free()

    complex_ = Complex(
        initial_separation_coeff=initial_separation_coeff,
        vertex_to_bind_idx=vertex_to_bind_idx,
        displacement_fn=displacement_fn, shift_fn=shift_fn,
        spider_base_radius=sim_params['spider_base_radius'],
        spider_head_height=sim_params['spider_head_height'],
        spider_base_particle_radius=sim_params['spider_base_particle_radius'],
        spider_attr_particle_radius=sim_params['spider_attr_site_radius'],
        spider_head_particle_radius=jnp.max(jnp.array([min_head_radius, sim_params['spider_head_particle_radius']])),
        spider_point_mass=1.0, spider_mass_err=1e-6,
        bond_radius=spider_leg_radius,
        rel_attr_particle_pos=jnp.clip(sim_params['spider_attr_particle_pos_norm'], 0.0, 1.0)
    )

    if op_name == "attr":
        def get_new_vertex_com(R, dist):
            leg_rbs = R[-5:] # the spider
            spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
            attr_site_pos = spider_space_frame_pos[1::3]
            avg_attr_site_pos = jnp.mean(attr_site_pos, axis=0)

            a = space.distance(displacement_fn(avg_attr_site_pos, attr_site_pos[0]))
            b = jnp.sqrt(dist**2 - a**2) # pythag

            vertex_com = R[0].center
            avg_attr_site_to_vertex = displacement_fn(avg_attr_site_pos, vertex_com)
            dir_ = avg_attr_site_to_vertex / jnp.linalg.norm(avg_attr_site_to_vertex)
            new_vertex_pos = avg_attr_site_pos - dir_*b
            return new_vertex_pos
    elif op_name == "head":
        def get_new_vertex_com(R, dist):
            leg_rbs = R[-5:] # the spider
            spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
            head_site_pos = spider_space_frame_pos[0::3]
            avg_head_site_pos = jnp.mean(head_site_pos, axis=0)

            # Note: assumes all head site positions are about the same
            vertex_com = R[0].center
            head_site_to_vertex = displacement_fn(avg_head_site_pos, vertex_com)
            dir_ = head_site_to_vertex / jnp.linalg.norm(head_site_to_vertex)
            new_vertex_pos = avg_head_site_pos - dir_*dist
            return new_vertex_pos
    else:
        raise RuntimeError(f"Invalid op_name: {op_name}")

    @jit
    def get_init_body(R, dist):
        new_vertex_pos = get_new_vertex_com(R, dist)
        new_center = R.center.at[0].set(new_vertex_pos)
        return rigid_body.RigidBody(new_center, R.orientation)

    combined_body, base_energy_fn = complex_.get_extracted_rb_info(
        morse_attr_eps=jnp.exp(sim_params['log_morse_attr_eps']),
        morse_attr_alpha=sim_params['morse_attr_alpha'],
        morse_r_onset=sim_params['morse_r_onset'],
        morse_r_cutoff=sim_params['morse_r_cutoff'])
    init_energy = base_energy_fn(combined_energy_fn)
    base_energy_fn = jit(base_energy_fn)

    with open(run_dir / "energy.txt", 'w+') as of:
        of.write(f"Init base energy: {init_energy}\n")

    combined_shape_species = onp.array([0, 1, 1, 1, 1, 1])
    mass = complex_.shape.mass(combined_shape_species)


    # Dummy simulation
    run_dummy = False
    if run_dummy:
        init_fn, step_fn = simulate.nvt_langevin(base_energy_fn, shift_fn, dt,
                                                 kT, gamma=gamma_rb)
        step_fn = jit(step_fn)

        state = init_fn(key, combined_body, mass=mass)
        n_steps = args['n_sample_states_per_sim'] * args['sample_every']
        traj = list()
        sample_every = args['sample_every']
        n_vis_states = 0
        for i in tqdm(range(n_steps)):
            state = step_fn(state)
            if i % sample_every == 0:
                traj.append(state.position)
                n_vis_states += 1

        traj_injavis_lines = list()
        box_size = 30.0
        for i in tqdm(range(n_vis_states), desc="Generating injavis output"):
            s = traj[i]
            traj_injavis_lines += combined_body_to_injavis_lines(complex_, s, box_size=box_size)[0]

        with open(run_dir / "test_wham_combined_sim.pos", 'w+') as of:
            of.write('\n'.join(traj_injavis_lines))


    # Start stuff for WHAM

    min_center = args['min_center']
    max_center = args['max_center']
    k_bias = args['k_bias']

    if not use_split_point:
        bin_centers = list(onp.linspace(min_center, max_center, n_bins))
        num_centers = len(bin_centers)
        bin_centers = jnp.array(bin_centers)

        k_biases = list()

        for center in bin_centers:
            k_biases.append(k_bias)
        k_biases = jnp.array(k_biases)
    else:
        assert(split_point > min_center and split_point < max_center)
        k_bias_split_point = args['k_bias_split_point']
        bin_centers_lo = list(onp.linspace(min_center, split_point, n_bins))
        k_biases_lo = [k_bias_split_point for _ in range(n_bins)]

        bin_centers_hi = list(onp.linspace(split_point, max_center, n_bins))
        k_biases_hi = [k_bias for _ in range(n_bins)]

        bin_centers = jnp.array(bin_centers_lo + bin_centers_lo)
        k_biases = jnp.array(k_biases_lo + k_biases_lo)
        num_centers = len(bin_centers)
        n_bins *= 2


    if op_name == "attr":
        def order_param_fn(R):
            leg_rbs = R[-5:] # the spider
            spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
            attr_site_pos = spider_space_frame_pos[1::3]

            vertex_com = R[0].center
            disps = vmap(displacement_fn, (None, 0))(vertex_com, attr_site_pos)
            drs = vmap(space.distance)(disps)
            return jnp.mean(drs)
    elif op_name == "head":
        def order_param_fn(R):
            leg_rbs = R[-5:] # the spider
            spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
            head_site_pos = spider_space_frame_pos[0::3]

            vertex_com = R[0].center
            disps = vmap(displacement_fn, (None, 0))(vertex_com, head_site_pos)
            drs = vmap(space.distance)(disps)
            return jnp.mean(drs)
    else:
        raise RuntimeError(f"Invalid op_name: {op_name}")
    get_traj_order_params = vmap(order_param_fn)


    def _harmonic_bias(op, center_idx):
        center = bin_centers[center_idx]
        k_bias = k_biases[center_idx]
        return 1/2*k_bias * (center - op)**2

    def harmonic_bias(R, center_idx):
        op = order_param_fn(R)
        return _harmonic_bias(op, center_idx)



    # n_eq_steps = 5000
    n_eq_steps = args['n_eq_steps']
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


    R_eq_inits = list()
    for c_idx in jnp.arange(num_centers):
        dist = bin_centers[c_idx]
        c_body = get_init_body(combined_body, dist)
        R_eq_inits.append(c_body)
    R_eq_inits = utils.tree_stack(R_eq_inits)

    key, eq_key = random.split(key)
    eq_keys = random.split(eq_key, num_centers)
    start = time.time()
    # R_eq = vmap(eq_fn, (None, 0, 0))(combined_body, bin_centers, eq_keys)
    # R_eq = vmap(eq_fn, (None, 0, 0))(combined_body, jnp.arange(num_centers), eq_keys)
    R_eq = vmap(eq_fn, (0, 0, 0))(R_eq_inits, jnp.arange(num_centers), eq_keys)
    end = time.time()
    eq_time = end - start


    traj_injavis_lines = list()
    # box_size = 30.0
    box_size = args['box_size']
    dist_string = ""
    bad_dists = list()
    bad_dist_tol = 0.5
    last_good = None
    for i in tqdm(range(num_centers), desc="Generating injavis output"):
        center = bin_centers[i]
        s = R_eq[i]
        s_op = order_param_fn(s)
        dist_string += f"Target dist: {center}\n"
        dist_string += f"Eq dist: {s_op}\n"
        print(f"Target dist: {center}")
        print(f"Eq dist: {s_op}")

        if onp.abs(center - s_op) > bad_dist_tol:
            bad_dists.append(center)
            if last_good is not None:
                new_center = R_eq.center.at[i].set(R_eq.center[last_good])
                new_q = R_eq.orientation.vec.at[i].set(R_eq.orientation.vec[last_good])
                R_eq = rigid_body.RigidBody(center=new_center, orientation=rigid_body.Quaternion(new_q))
                dist_string += f"- Replacing with {last_good}\n"
                print(f"- Replacing with {last_good}")
        else:
             last_good = i

        traj_injavis_lines += combined_body_to_injavis_lines(complex_, s, box_size=box_size)[0]
    dist_string += f"\n\nBad Distances:\n"
    for bad_dist in bad_dists:
        dist_string += f"- {bad_dist}\n"

    with open(run_dir / "dist_info.txt", 'w+') as of:
        of.write(dist_string)

    with open(run_dir / "wham_eq_states.pos", 'w+') as of:
        of.write('\n'.join(traj_injavis_lines))


    # sample_every = 100
    sample_every = args['sample_every']
    # n_sample_states_per_sim = 50
    n_sample_states_per_sim = args['n_sample_states_per_sim']
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
    plt.savefig(run_dir / "all_hist.png")
    plt.clf()

    for t_idx in range(num_centers):
        center = bin_centers[t_idx]
        traj_ops = all_traj_order_params[t_idx]

        sns.kdeplot(traj_ops, label=f"{center}")

    # plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "all_kde.png")
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
        t_line = f"{fpath}\t{center}\t{k_biases[t_idx]}\n"
        metadata_lines.append(t_line)

    metadata_path = wham_dir / "metadata.txt"
    with open(metadata_path , "w+") as f:
        f.writelines(metadata_lines)


    ## Run the WHAM executable
    wham_out_path = wham_dir / "analysis.txt"
    start = time.time()
    command_string = f"{wham_exec_path} {min_center} {max_center} {n_bins} {wham_tol} {kT} 0 {metadata_path} {wham_out_path}"
    with open(run_dir / "wham_command.txt", "a") as f:
        f.write(f"{command_string}\n")
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

    plot_fe(wham_out_path, n_bins, run_dir / "fe.png")


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

    parser.add_argument('--init-sep-coeff', type=float, default=0.2,
                        help="Initial separation coefficient")
    parser.add_argument('--min-head-radius', type=float, default=0.1,
                        help="Tolerance for free energy convergence.")

    parser.add_argument('--min-center', type=float, default=2.5,
                        help="Minimum OP for WHAM.")
    parser.add_argument('--max-center', type=float, default=6.0,
                        help="Maximum OP for WHAM.")
    parser.add_argument('--k-bias', type=float, default=5e4,
                        help="Spring constant for umbrella sampling.")

    parser.add_argument('--n-eq-steps', type=int, default=5000,
                        help="Number of equilibration steps.")
    parser.add_argument('--box-size', type=float, default=50.0,
                        help="Length of side length of box.")
    parser.add_argument('--sample-every', type=int, default=100,
                        help="Sampling frequency.")
    parser.add_argument('--n-sample-states-per-sim', type=int, default=50,
                        help="Number of states to sample per sim. Determines # steps along with sample_every.")

    parser.add_argument('--n-bins', type=int, default=400,
                        help="Number of WHAM bins.")

    parser.add_argument('--dt', type=float, default=1e-3,
                        help="Timestep for simulation.")

    parser.add_argument('--spider-leg-radius', type=float, default=0.25, help="Spider leg radius")
    parser.add_argument('--op-name', type=str, help='Name of order parameter',
                        choices=["attr", "head"],
                        default="attr")

    parser.add_argument('--use-split-point', action='store_true')
    parser.add_argument('--split-point', type=float, default=4.0,
                        help="Point for splitting the centers.")
    parser.add_argument('--k-bias-split-point', type=float, default=50000.0,
                        help="Spring constant for centers between min_center and split_point.")

    return parser


if __name__ == "__main__":

    plot_fe("analysis-ext-smaller.txt", n_bins=100, savepath="ex_fe_ext.png")
    plot_fe("analysis-tagged-smaller.txt", n_bins=100, savepath="ex_fe_tagged.png")
    pdb.set_trace()


    parser = get_parser()
    args = vars(parser.parse_args())

    """
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
    """

    # ext-rigid-tagged-test-eps3-bigger-radius-start, iteration 350
    params = {
        "log_morse_attr_eps": 4.445757112690842,
        "morse_attr_alpha": 1.228711252063668,
        "morse_r_cutoff": 12.0,
        "morse_r_onset": 10.0,
        "spider_attr_particle_pos_norm": 0.31171913270018414,
        "spider_attr_site_radius": 1.4059036817138681,
        "spider_base_particle_radius": 1.0949878258735661,
        "spider_base_radius": 5.018836622251073,
        "spider_head_height": 9.462070953473482,
        "spider_head_particle_radius": 1.0
    }

    run(args, params)
