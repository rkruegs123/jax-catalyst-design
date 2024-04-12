import numpy as onp
import pdb
import decimal
from tqdm import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt

from jax_md import space, simulate
import catalyst.icosahedron_tagged.rigid_body as rigid_body
from jax import random, jit, vmap, lax
import jax.numpy as jnp

from catalyst.icosahedron_tagged.complex import Complex, combined_body_to_injavis_lines





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
# k_bias = 1000

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
