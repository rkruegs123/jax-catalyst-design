import numpy as onp
import pdb
import decimal
from tqdm import tqdm

from jax_md import space, simulate
import catalyst.icosahedron_tagged.rigid_body as rigid_body
from jax import random, jit, vmap
import jax.numpy as jnp

from catalyst.icosahedron_tagged.complex import Complex




min_dist = 2.5
max_dist = 10.0
dist_diff = max_dist - min_dist
bin_width = 0.5

dec_difference = decimal.Decimal(str(max_dist)) - decimal.Decimal(str(min_dist))
assert(dec_difference % decimal.Decimal(str(bin_width)) == decimal.Decimal('0.0'))
num_centers = int(dec_difference // decimal.Decimal(str(bin_width)))
bin_centers = onp.array([min_dist+i*bin_width for i in range(num_centers)])

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


def combined_body_to_injavis_lines(
        body, box_size, shell_patch_radius=0.5, shell_vertex_color="43a5be",
        shell_patch_color="4fb06d", spider_head_color="ff0000", spider_base_color="1c1c1c"):

    vertex_body = body[0]
    vertex_body = rigid_body.RigidBody(
        center=jnp.expand_dims(vertex_body.center, 0),
        orientation=rigid_body.Quaternion(jnp.expand_dims(vertex_body.orientation.vec, 0)))
    spider_body = body[1:]

    _, spider_box_def, spider_type_defs, spider_pos = complex_.spider.body_to_injavis_lines(
        spider_body, box_size)
    _, shell_box_def, shell_type_defs, shell_pos = complex_.shell.body_to_injavis_lines(
        vertex_body, box_size, shell_patch_radius, vertex_to_bind=vertex_to_bind_idx)

    assert(spider_box_def == shell_box_def)
    box_def = spider_box_def
    type_defs = shell_type_defs + spider_type_defs
    positions = shell_pos + spider_pos
    all_lines = [box_def] + type_defs + positions + ["eof"]
    return all_lines, box_def, type_defs, positions



# Extract the spider and the single vertex from the rigid body
complex_rb = complex_.rigid_body
spider_body, shell_body = complex_.split_body(complex_rb)
vertex_to_bind = shell_body[vertex_to_bind_idx]
complex_shape = complex_.shape
complex_shape_species = complex_.shape_species


combined_center = jnp.concatenate([onp.array([vertex_to_bind.center]), spider_body.center])
combined_quat_vec = jnp.concatenate([
    onp.array([vertex_to_bind.orientation.vec]),
    spider_body.orientation.vec])
combined_body = rigid_body.RigidBody(combined_center, rigid_body.Quaternion(combined_quat_vec))
combined_shape_species = onp.array([0, 1, 1, 1, 1, 1])

spider_energy_fn = complex_.spider.get_energy_fn()

def base_energy_fn(body, **kwargs):
    vertex_body = body[0]
    spider_body = body[1:]
    spider_val = spider_energy_fn(spider_body, **kwargs)
    return spider_val


pdb.set_trace()


# Dummy simulation
gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
init_fn, step_fn = simulate.nvt_langevin(base_energy_fn, shift_fn, dt,
                                         kT, gamma=gamma_rb)
step_fn = jit(step_fn)
mass = complex_shape.mass(combined_shape_species)
state = init_fn(key, combined_body, mass=mass)
n_steps = 10000
traj = list()
for i in tqdm(range(n_steps)):
    state = step_fn(state)
    traj.append(state.position)


traj_injavis_lines = list()
n_vis_states = len(traj)
box_size = 30.0
vis_every = 500
for i in tqdm(range(n_vis_states), desc="Generating injavis output"):
    if i % vis_every == 0:
        s = traj[i]
        traj_injavis_lines += combined_body_to_injavis_lines(s, box_size=box_size)[0]

with open("test_combined_sim.pos", 'w+') as of:
    of.write('\n'.join(traj_injavis_lines))







