import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp
import matplotlib.pyplot as plt

from jax import vmap, lax, jit, random
import jax.numpy as jnp
from jax_md import space, simulate, energy
# from jax_md import rigid_body

import catalyst.icosahedron_tagged.rigid_body as rigid_body
from catalyst.icosahedron_tagged.leg import Leg
from catalyst.icosahedron_tagged import utils

from jax.config import config
config.update('jax_enable_x64', True)




def get_target_positions(head_height, base_radius):
    target_positions = jnp.zeros((5, 3))
    x = head_height
    def scan_fn(target_positions, i):
        y = base_radius * jnp.cos(i * 2 * jnp.pi / 5)
        z = base_radius * jnp.sin(i * 2 * jnp.pi / 5)
        target_positions = target_positions.at[i, :].set(jnp.array([x, y, z]))
        return target_positions, i
    target_positions, _ = lax.scan(scan_fn, target_positions, jnp.arange(5))

    return target_positions

class Spider:
    def __init__(self, displacement_fn, shift_fn,
                 base_radius, head_height,
                 base_particle_radius,
                 attr_particle_pos_norm, attr_site_radius,
                 head_particle_radius,
                 point_mass=1.0, mass_err=1e-6, target_positions=None,
                 bond_alpha=2.0
    ):

        self.displacement_fn = displacement_fn
        self.shift_fn = shift_fn
        self.n_legs = 5
        self.base_radius = base_radius
        self.base_particle_radius = base_particle_radius
        self.head_particle_radius = head_particle_radius
        self.attr_site_radius = attr_site_radius
        self.head_height = head_height
        self.particle_radii = jnp.array([head_particle_radius, attr_site_radius, base_particle_radius])
        self.bond_alpha = bond_alpha

        self.legs = [Leg(base_radius, head_height, base_particle_radius,
                         attr_particle_pos_norm, attr_site_radius,
                         head_particle_radius, point_mass, mass_err) for _ in range(self.n_legs)]
        spider_rb = utils.tree_stack([leg.rigid_body for leg in self.legs])
        spider_shape = self.legs[0].shape


        d = vmap(displacement_fn, (0, None))

        leg_length = self.legs[0].leg_length

        if target_positions is None:
            target_positions = get_target_positions(head_height, base_radius)

        self.target_positions = target_positions
        self.target_position_dx = space.distance(displacement_fn(target_positions[0], target_positions[1]))


        start_base_pos = jnp.array([leg_length, 0.0, 0.0])
        start_head_pos = jnp.array([0.0, 0.0, 0.0])
        reoriented_vectors = d(target_positions, start_head_pos)
        norm = jnp.linalg.norm(reoriented_vectors, axis=1).reshape(-1, 1)
        reoriented_vectors /= norm

        orig_vec = displacement_fn(start_head_pos, start_base_pos)
        orig_vec /= jnp.linalg.norm(orig_vec)

        # Note: we normalize HERE
        crossed = vmap(jnp.cross, (None, 0))(orig_vec, reoriented_vectors)
        norm = jnp.linalg.norm(crossed, axis=1).reshape(-1, 1)
        crossed /= norm

        # dotted = vmap(jnp.dot, (0, None))(reoriented_vectors, orig_vec)
        dotted = vmap(jnp.dot, (None, 0))(orig_vec, reoriented_vectors)
        theta = jnp.arccos(dotted)
        cos_part = jnp.cos(theta / 2).reshape(-1, 1)
        mult = vmap(lambda v, s: s*v, (0, 0))
        sin_part = mult(crossed, jnp.sin(theta/2))
        orientation = jnp.concatenate([cos_part, sin_part], axis=1)

        orientation = rigid_body.Quaternion(orientation)
        spider_rb = spider_rb.set(orientation=orientation)

        self.rigid_body = spider_rb
        self.shape = spider_shape


    def get_energy_fn(self, add_bonds=True):
        # Morse attraction between heads
        morse_alpha = 4.0
        morse_eps = 100.0
        morse_eps_mat = morse_eps * jnp.array([[1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]]) # only heads attract
        pair_energy_morse = energy.morse_pair(self.displacement_fn, species=3, sigma=0.0,
                                              epsilon=morse_eps_mat, alpha=morse_alpha)

        # Soft sphere repulsion between everything except heads
        soft_sphere_eps = 1000.0
        soft_sphere_eps_mat = soft_sphere_eps * jnp.array([[0.0, 1.0, 1.0],
                                                           [1.0, 1.0, 1.0],
                                                           [1.0, 1.0, 1.0]])
        soft_sphere_sigma_mat = jnp.array([[self.head_particle_radius*2, self.head_particle_radius+self.attr_site_radius, self.base_particle_radius+self.head_particle_radius],
                                           [self.head_particle_radius+self.attr_site_radius, self.attr_site_radius*2, self.base_particle_radius+self.attr_site_radius],
                                           [self.head_particle_radius+self.base_particle_radius, self.attr_site_radius+self.base_particle_radius, self.base_particle_radius*2]])
        pair_energy_soft = energy.soft_sphere_pair(self.displacement_fn, species=3,
                                                   sigma=soft_sphere_sigma_mat,
                                                   epsilon=soft_sphere_eps_mat)

        # Weak repulsion between attractive sites
        weak_ss_eps = 10000.0
        weak_ss_eps_mat = weak_ss_eps * jnp.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]]
        )
        pair_energy_soft_weak = energy.soft_sphere_pair(self.displacement_fn, species=3,
                                                        sigma=1.2*self.attr_site_radius*2,
                                                        epsilon=weak_ss_eps_mat, alpha=2.0)

        pair_energy_fn = lambda R, **kwargs: pair_energy_morse(R, **kwargs) \
                         + pair_energy_soft(R, **kwargs)
                         # + pair_energy_soft_weak(R, **kwargs)

        if add_bonds:
            flattened_base_idxs = onp.arange(2, self.n_legs*3)[::3]
            bonds = jnp.array([[flattened_base_idxs[0], flattened_base_idxs[1]],
                               [flattened_base_idxs[2], flattened_base_idxs[3]],
                               [flattened_base_idxs[1], flattened_base_idxs[2]]
                               #[flattened_base_idxs[3], flattened_base_idxs[4]]
            ])
            pair_energy_bond = energy.simple_spring_bond(self.displacement_fn, bonds, length=self.target_position_dx, epsilon=1000.)
            pair_energy_fn_with_bonds = lambda R, **kwargs: pair_energy_fn(R, **kwargs) + pair_energy_bond(R, **kwargs)
            energy_fn = rigid_body.point_energy(pair_energy_fn_with_bonds, self.shape)
        else:
            energy_fn = rigid_body.point_energy(pair_energy_fn, self.shape)

        return energy_fn


    def body_to_injavis_lines(self, body, box_size, head_color="ff0000",
                              base_color="1c1c1c", attr_color="3d8c40"):


        leg0 = self.legs[0]

        all_positions = list()
        num_legs = body.center.shape[0]
        for leg_rb_idx in range(num_legs):
            leg_rb = body[leg_rb_idx]
            all_lines, box_def, type_defs, positions = leg0.body_to_injavis_lines(
                leg_rb, box_size, head_color, base_color, attr_color)

            # all_positions.append(positions)
            all_positions += positions

        all_lines = [box_def] + type_defs + all_positions + ["eof"]

        return all_lines, box_def, type_defs, all_positions

class TestSpider(unittest.TestCase):
    def test_no_error(self):
        displacement_fn, shift_fn = space.free()
        spider = Spider(displacement_fn, shift_fn,
                        base_radius=4.0, head_height=5.0,
                        base_particle_radius=0.5,
                        attr_particle_pos_norm=0.5, attr_site_radius=1.0,
                        head_particle_radius=0.25)

    def test_simulate(self):
        displacement_fn, shift_fn = space.free()

        base_particle_radius = 0.5
        attr_site_radius = 0.5
        head_particle_radius = 0.25
        base_radius = 4.0
        spider = Spider(displacement_fn, shift_fn,
                        base_radius=base_radius, head_height=5.0,
                        base_particle_radius=base_particle_radius,
                        attr_particle_pos_norm=0.5, attr_site_radius=attr_site_radius,
                        head_particle_radius=head_particle_radius)
        spider_rb = spider.rigid_body

        energy_fn = spider.get_energy_fn()
        energy_fn = jit(energy_fn)

        dt = 1e-3
        kT = 1.0
        gamma = 10.0
        gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
        # init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma_rb)
        step_fn = jit(step_fn)
        key = random.PRNGKey(0)
        state = init_fn(key, spider_rb, mass=spider.shape.mass())

        trajectory = list()
        energies = list()
        n_steps = 50000
        for _ in tqdm(range(n_steps)):
            state = step_fn(state)
            trajectory.append(state.position)
            energies.append(energy_fn(state.position))

        trajectory = utils.tree_stack(trajectory)

        n_vis_freq = 500
        vis_traj_idxs = jnp.arange(0, n_steps+1, n_vis_freq)
        n_vis_states = len(vis_traj_idxs)
        trajectory = trajectory[vis_traj_idxs]

        leg0 = spider.legs[0]
        spring_dist = spider.target_position_dx
        all_dists_b1 = list()
        all_dists_b2 = list()
        for i in tqdm(range(n_vis_states)):
            s = trajectory[i]

            l0_pos = leg0.get_body_frame_positions(s[0])
            l1_pos = leg0.get_body_frame_positions(s[1])
            curr_dist_b1 = space.distance(displacement_fn(l0_pos[2], l1_pos[2]))
            all_dists_b1.append(curr_dist_b1)

            l2_pos = leg0.get_body_frame_positions(s[2])
            l3_pos = leg0.get_body_frame_positions(s[3])
            curr_dist_b2 = space.distance(displacement_fn(l2_pos[2], l3_pos[2]))
            all_dists_b2.append(curr_dist_b2)

        plt.plot(all_dists_b1, label="Bond 1")
        plt.plot(all_dists_b2, label="Bond 2")
        plt.axhline(y=spring_dist, linestyle="--", color="red", label="Spring Length")
        plt.legend()
        plt.show()
        plt.clf()


        box_size = 30.0
        traj_injavis_lines = list()
        traj_path = "new_spider.pos"
        for i in tqdm(range(n_vis_states), desc="Generating injavis output"):
            s = trajectory[i]
            traj_injavis_lines += spider.body_to_injavis_lines(s, box_size=box_size)[0]
        with open(traj_path, 'w+') as of:
            of.write('\n'.join(traj_injavis_lines))





if __name__ == "__main__":
    unittest.main()
