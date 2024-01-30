import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp

from jax import vmap, lax, tree_util
import jax.numpy as jnp
from jax_md import rigid_body, space, simulate

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

        self.n_legs = 3 # FIXME: arbitrary -- will have to change

        self.legs = [Leg(base_radius, head_height, base_particle_radius,
                         attr_particle_pos_norm, attr_site_radius,
                         head_particle_radius, point_mass, mass_err) for _ in range(self.n_legs)]
        spider_rb = tree_stack([leg.rigid_body for leg in self.legs])
        spider_shape = rigid_body.concatenate_shapes(*[leg.shape for leg in self.legs])

        pdb.set_trace()

    def body_to_injavis_line(self, body, box_size, head_color="ff0000",
                              base_color="1c1c1c", attr_color="3d8c40"):
        leg0 = self.legs[0]

        all_positions = list()
        for leg_rb in body:
            all_lines, box_def, type_defs, positions = leg0.body_to_injavis_lines(
                leg_rb, box_size, head_color, base_color, attr_color)

            all_positions.append(positions)

        all_lines = [box_def] + type_defs + all_positionns + ["eof"]

        return all_lines, box_def, type_defs, all_positions

class TestSpider(unittest.TestCase):
    def test_no_error(self):
        spider = Spider(base_radius=4.0, head_height=5.0,
                        base_particle_radius=0.5,
                        attr_particle_pos_norm=0.5, attr_site_radius=1.0,
                        head_particle_radius=0.25)

    def test_simulate(self):

        base_particle_radius = 0.5
        attr_site_radius = 1.0
        head_particle_radius = 0.25
        spider = Spider(base_radius=4.0, head_height=5.0,
                        base_particle_radius=base_particle_radius,
                        attr_particle_pos_norm=0.5, attr_site_radius=attr_site_radius,
                        head_particle_radius=head_particle_radius)

        displacement_fn, shift_fn = space.free()
        morse_alpha = 4.0
        morse_eps_mat = morse_eps * jnp.array([[10.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]]) # only heads attract
        pair_energy_morse = energy.morse_pair(displacement_fn, species=3, sigma=0.0,
                                              epsilon=morse_eps_mat, alpha=morse_alpha)

        soft_sphere_eps = 10000.0
        soft_sphere_eps_mat = soft_sphere_eps * jnp.array([[0.0, 1.0, 1.0],
                                                           [1.0, 1.0, 1.0],
                                                           [1.0, 1.0, 1.0]])
        soft_sphere_sigma_mat = jnp.array([[head_particle_radius*2, head_particle_radius+attr_site_radius, base_particle_radius+head_particle_radius],
                                           [head_particle_radius+attr_site_radius, attr_site_radius*2, base_particle_radius+attr_site_radius],
                                           [head_particle_radius+base_particle_radius, attr_site_radius+base_particle_radius, base_particle_radius*2]])
        pair_energy_soft = energy.soft_sphere_pair(displacement_fn, species=3,
                                                   sigma=soft_sphere_sigma_mat,
                                                   epsilon=soft_sphere_eps_mat)

        pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) \
                         + pair_energy_morse(R, **kwargs)
        energy_fn = rigid_body.point_energy(pair_energy_fn, spider.shape)

        dt=1e-4
        kT = 1.0
        init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, self.shift_fn, dt, kT)
        step_fn = jit(step_fn)
        key = random.PRNGKey(0)
        state = init_fn(key, spider.rigid_body, mass=spider.shape.mass())

        trajectory = list()
        for _ in range(1000):
            state = step_fn(state)
            trajectory.append(state.position)



if __name__ == "__main__":
    unittest.main()
