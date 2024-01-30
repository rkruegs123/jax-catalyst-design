import pdb
import matplotlib.pyplot as plt

from jax_md import space
import jax.numpy as jnp
from jax import jit, tree_util

from catalyst.octahedron.complex_getter import ComplexInfo, TETRAPOD_LEGS, BASE_LEGS
from catalyst.octahedron import utils
from catalyst.octahedron.utils import get_body_frame_positions, traj_to_pos_file


def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)

def run(params):
    displacement_fn, shift_fn = space.free()

    spider_bond_idxs = jnp.concatenate([TETRAPOD_LEGS, BASE_LEGS])

    complex_info = ComplexInfo(
        initial_separation_coeff=0.0, vertex_to_bind_idx=utils.vertex_to_bind_idx,
        displacement_fn=displacement_fn, shift_fn=shift_fn,
        spider_base_radius=params["spider_base_radius"],
        spider_head_height=params["spider_head_height"],
        spider_base_particle_radius=params["spider_base_particle_radius"],
        spider_head_particle_radius=params["spider_head_particle_radius"],
        spider_point_mass=1.0, spider_mass_err=1e-6,
        spider_bond_idxs=spider_bond_idxs, spider_leg_radius=1.0
    )

    energy_fn = complex_info.get_energy_fn(
        morse_shell_center_spider_head_eps=jnp.exp(params["log_morse_shell_center_spider_head_eps"]),
        morse_shell_center_spider_head_alpha=params["morse_shell_center_spider_head_alpha"]
    )
    energy_fn = jit(energy_fn)


    rigid_body = complex_info.rigid_body
    curr_energy = energy_fn(rigid_body)
    prev_energy = curr_energy + 1.0
    move_amount = 0.05
    trajectory = list()
    energies = list()
    max_iters = 100
    i = 0

    # Extraction stage
    while (curr_energy < prev_energy) and i < max_iters:
        trajectory.append(rigid_body)
        energies.append(energy_fn(rigid_body))

        spider_rb = rigid_body[-1]
        spider_head_pos = complex_info.spider_info.get_body_frame_positions(spider_rb)[-1]

        target_vertex_rb = rigid_body[utils.vertex_to_bind_idx]

        dr = displacement_fn(spider_head_pos, target_vertex_rb.center)
        new_target_vertex_center = shift_fn(target_vertex_rb.center, dr*move_amount)

        new_rb_center = rigid_body.center.at[utils.vertex_to_bind_idx].set(new_target_vertex_center)
        rigid_body = rigid_body.set(center=new_rb_center)

        print(i)
        prev_energy = curr_energy
        curr_energy = energy_fn(rigid_body)
        print(curr_energy)

        i += 1
    end_extraction = i-1

    # Detachment stage
    total_distance = 3.0
    increment = 0.1
    distance_moved = 0.0
    shell_rb = complex_info.shell_info.rigid_body
    shell_center = jnp.mean(shell_rb.center, axis=0)
    while distance_moved < total_distance:
        spider_rb = rigid_body[-1]

        dr_axis = displacement_fn(spider_rb.center, shell_center)
        dr_axis_norm = dr_axis / dr_axis.sum()

        dr = -dr_axis_norm * increment

        rigid_body_center = rigid_body.center.at[utils.vertex_to_bind_idx].set(shift_fn(rigid_body.center[utils.vertex_to_bind_idx], dr))
        rigid_body_center = rigid_body_center.at[-1].set(shift_fn(rigid_body_center[-1], dr))

        rigid_body = rigid_body.set(center=rigid_body_center)
        trajectory.append(rigid_body)
        energies.append(energy_fn(rigid_body))

        distance_moved += increment



    plt.plot(energies)
    plt.axvline(x=end_extraction, linestyle="--", label="End Extraction")
    plt.legend()
    plt.show()
    plt.clf()

    trajectory = tree_stack(trajectory)
    traj_to_pos_file(trajectory, complex_info, "path.pos", box_size=30.0)

    return energies, end_extraction



if __name__ == "__main__":

    # Low energy limit solution
    low_params = {
        "log_morse_shell_center_spider_head_eps": 6.022961752044741,
        "morse_r_cutoff": 9.151092213048766,
        "morse_r_onset": 8.771026317332652,
        "morse_shell_center_spider_head_alpha": 2.9744531045779055,
        "spider_base_particle_radius": 1.698872953390476,
        "spider_base_radius": 3.3987408062303737,
        "spider_head_height": 3.4576374094329965,
        "spider_head_particle_radius": 2.0178703029523812,
    }

    low_energies, low_end_extraction = run(low_params)

    pdb.set_trace()


    # Unoptimized
    unoptimized_params = {
        "spider_base_radius": 5.0,
        "spider_head_height": 5.0,
        "spider_base_particle_radius": 0.5,
        "spider_head_particle_radius": 0.5,
        "log_morse_shell_center_spider_head_eps": 8.0,
        "morse_shell_center_spider_head_alpha": 1.0,
        "morse_r_onset": 10.0,
        "morse_r_cutoff": 12.0,
    }

    unopt_energies, unopt_end_extraction = run(unoptimized_params)

    # High energy limit solution
    high_params = {
        "log_morse_shell_center_spider_head_eps": 8.04724810628451,
        "morse_r_cutoff": 10.952194181007227,
        "morse_r_onset": 9.634176788323536,
        "morse_shell_center_spider_head_alpha": 1.5980508665424513,
        "spider_base_particle_radius": 1.4432461247304358,
        "spider_base_radius": 3.581915778221133,
        "spider_head_height": 5.092513564314658,
        "spider_head_particle_radius": 0.3733964968469948,
    }

    hi_energies, hi_end_extraction = run(high_params)


    plt.plot(unopt_energies, color="blue", label="unopt")
    plt.axvline(x=unopt_end_extraction, linestyle="--", color="blue", label="unopt End Extraction")

    plt.plot(hi_energies, color="red", label="high")
    plt.axvline(x=hi_end_extraction, linestyle="--", color="red", label="high End Extraction")

    plt.plot(low_energies, color="green", label="low")
    plt.axvline(x=low_end_extraction, linestyle="--", color="green", label="low End Extraction")

    plt.legend()
    plt.show()
    plt.clf()
