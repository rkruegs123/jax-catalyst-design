import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp

from jax import vmap, lax, tree_util, jit, random
import jax.numpy as jnp
from jax_md import rigid_body, space, simulate, energy

from catalyst.icosahedron_new.leg import Leg

from jax.config import config
config.update('jax_enable_x64', True)




def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)

class Spider:
    def __init__(self, base_radius, head_height,
                 base_particle_radius,
                 attr_particle_pos_norm, attr_site_radius,
                 head_particle_radius,
                 point_mass=1.0, mass_err=1e-6):

        self.n_legs = 5 # FIXME: arbitrary -- will have to change

        self.legs = [Leg(base_radius, head_height, base_particle_radius,
                         attr_particle_pos_norm, attr_site_radius,
                         head_particle_radius, point_mass, mass_err) for _ in range(self.n_legs)]
        spider_rb = tree_stack([leg.rigid_body for leg in self.legs])
        spider_shape = rigid_body.concatenate_shapes(*[leg.shape for leg in self.legs])


        displacement_fn, shift_fn = space.free()
        d = vmap(displacement_fn, (None, 0))

        target_positions = jnp.zeros((5, 3))
        leg_length = self.legs[0].leg_length
        x = head_height
        def scan_fn(target_positions, i):
            y = base_radius * jnp.cos(i * 2 * jnp.pi / 5)
            z = base_radius * jnp.sin(i * 2 * jnp.pi / 5)
            target_positions = target_positions.at[i, :].set(jnp.array([x, y, z]))
            return target_positions, i
        target_positions, _ = lax.scan(scan_fn, target_positions, jnp.arange(5))

        start_base_pos = jnp.array([leg_length, 0.0, 0.0])
        start_head_pos = jnp.array([0.0, 0.0, 0.0])
        reoriented_vectors = d(start_head_pos, target_positions)
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
        spider = Spider(base_radius=4.0, head_height=5.0,
                        base_particle_radius=0.5,
                        attr_particle_pos_norm=0.5, attr_site_radius=1.0,
                        head_particle_radius=0.25)

    def test_simulate(self):

        base_particle_radius = 0.5
        attr_site_radius = 0.5
        head_particle_radius = 0.25
        base_radius = 4.0
        spider = Spider(base_radius=base_radius, head_height=5.0,
                        base_particle_radius=base_particle_radius,
                        attr_particle_pos_norm=0.5, attr_site_radius=attr_site_radius,
                        head_particle_radius=head_particle_radius)
        spider_rb = spider.rigid_body

        displacement_fn, shift_fn = space.free()
        morse_alpha = 4.0
        morse_eps = 100.0
        morse_eps_mat = morse_eps * jnp.array([[1.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]]) # only heads attract
        pair_energy_morse = energy.morse_pair(displacement_fn, species=3, sigma=0.0,
                                              epsilon=morse_eps_mat, alpha=morse_alpha)

        soft_sphere_eps = 1000.0
        soft_sphere_eps_mat = soft_sphere_eps * jnp.array([[0.0, 1.0, 1.0],
                                                           [1.0, 1.0, 1.0],
                                                           [1.0, 1.0, 1.0]])
        soft_sphere_sigma_mat = jnp.array([[head_particle_radius*2, head_particle_radius+attr_site_radius, base_particle_radius+head_particle_radius],
                                           [head_particle_radius+attr_site_radius, attr_site_radius*2, base_particle_radius+attr_site_radius],
                                           [head_particle_radius+base_particle_radius, attr_site_radius+base_particle_radius, base_particle_radius*2]])
        pair_energy_soft = energy.soft_sphere_pair(displacement_fn, species=3,
                                                   sigma=soft_sphere_sigma_mat,
                                                   epsilon=soft_sphere_eps_mat)


        weak_ss_eps = 10000.0
        weak_ss_eps_mat = weak_ss_eps * jnp.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]]
        )
        pair_energy_soft_weak = energy.soft_sphere_pair(displacement_fn, species=3,
                                                        sigma=1.2*attr_site_radius*2,
                                                        epsilon=weak_ss_eps_mat, alpha=2.0)


        pentagon_side_length = 2*base_radius*onp.sin(36.0)

        pair_energy_fn = lambda R, **kwargs: pair_energy_morse(R, **kwargs) \
                         + pair_energy_soft(R, **kwargs)
                         # + pair_energy_soft_weak(R, **kwargs)


        energy_fn = rigid_body.point_energy(pair_energy_fn, spider.shape)
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
        for _ in range(n_steps):
            state = step_fn(state)
            trajectory.append(state.position)
            energies.append(energy_fn(state.position))

        trajectory = tree_stack(trajectory)

        n_vis_states = 200
        vis_traj_idxs = jnp.arange(0, n_steps+1, n_vis_states)
        trajectory = trajectory[vis_traj_idxs]

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
