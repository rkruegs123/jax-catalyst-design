import numpy as onp
import pdb
import decimal
from tqdm import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt

from jax_md import space, simulate
import catalyst.icosahedron.rigid_body as rigid_body
from jax import random, jit, vmap, lax
import jax.numpy as jnp


from catalyst.icosahedron.complex_info import ComplexInfo, PENTAPOD_LEGS, BASE_LEGS, combined_body_to_injavis_lines


sim_params = {
    # catalyst shape
    'spider_base_radius': 5.0,
    'spider_head_height': 4.0,
    'spider_base_particle_radius': 0.5,
    'spider_head_particle_radius': 0.5,

    # catalyst energy
    'log_morse_shell_center_spider_head_eps': 9.21,
    'morse_shell_center_spider_head_alpha': 1.5,
    'morse_r_onset': 10.0,
    'morse_r_cutoff': 12.0
}


initial_separation_coeff = 0.1
vertex_to_bind_idx = 5

dt = 1e-3
kT = 1.0
gamma = 10
gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
key = random.PRNGKey(0)

displacement_fn, shift_fn = space.free()
spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])

complex_ = ComplexInfo(
    initial_separation_coeff=initial_separation_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
    displacement_fn=displacement_fn, shift_fn=shift_fn,
    spider_base_radius=sim_params['spider_base_radius'], spider_head_height=sim_params['spider_head_height'],
    spider_base_particle_radius=sim_params['spider_base_particle_radius'],
    spider_head_particle_radius=sim_params['spider_head_particle_radius'],
    spider_point_mass=1.0, spider_mass_err=1e-6,
    spider_bond_idxs=spider_bond_idxs
)


combined_body, base_energy_fn, leg_energy_fn = complex_.get_extracted_rb_info(
    morse_shell_center_spider_head_eps=jnp.exp(sim_params['log_morse_shell_center_spider_head_eps']),
    morse_shell_center_spider_head_alpha=sim_params['morse_shell_center_spider_head_alpha'],
    morse_r_onset=sim_params['morse_r_onset'],
    morse_r_cutoff=sim_params['morse_r_cutoff']
)
init_energy = base_energy_fn(init_body)
base_energy_fn = jit(base_energy_fn)



# Do WHAM
bin_centers = list(onp.linspace(2.5, 10.0, 20))
num_centers = len(bin_centers)
bin_centers= jnp.array(bin_centers)

k_biases = list()
for center in bin_centers:
    k_biases.append(5e3)
k_biases = jnp.array(k_biases)


# FIXME
def order_param_fn(R):
    spider_body = body[-1]
    vertex_body = body[0]
    spider_body_pos = self.spider_info.get_body_frame_positions(spider_body)
    head_pos = spider_body_pos[-1]

    dr = displacement_fn(head_pos, vertecx_body.center)
    r = space.distance(dr)
    return r
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
