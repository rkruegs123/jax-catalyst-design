import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp

from jax import vmap, lax
import jax.numpy as jnp
# from jax_md import rigid_body

import catalyst.icosahedron_tagged.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)

class Leg:
    def __init__(self, base_radius, head_height,
                 base_particle_radius,
                 attr_particle_pos_norm, attr_site_radius,
                 head_particle_radius,
                 point_mass=1.0, mass_err=1e-6):
        self.base_radius = base_radius
        self.head_height = head_height
        self.attr_particle_pos_norm = attr_particle_pos_norm
        self.attr_site_radius = attr_site_radius
        self.base_particle_radius = base_particle_radius
        self.head_particle_radius = head_particle_radius

        self.mass_err = mass_err
        self.point_mass = point_mass

        self.leg_length = onp.sqrt(base_radius**2 + head_height**2)


        head_pos = [0.0, 0.0, 0.0]
        attr_pos = [self.leg_length * self.attr_particle_pos_norm, 0.0, 0.0]
        base_pos = [self.leg_length, 0.0, 0.0]
        leg_positions = jnp.array([head_pos, attr_pos, base_pos])

        n_positions = len(leg_positions)
        # masses = jnp.ones(n_positions) * self.point_mass + jnp.arange(n_positions) * self.mass_err
        masses = jnp.array([0.01, 0.5, 1.0]) + jnp.arange(n_positions) * self.mass_err

        leg_rb = rigid_body.RigidBody(
            center=jnp.array([0.0, 0.0, 0.0]),
            orientation=rigid_body.Quaternion(jnp.array([1.0, 0.0, 0.0, 0.0])))
        leg_species = jnp.array([0, 1, 2])
        leg_shape = rigid_body.point_union_shape(leg_positions, masses).set(point_species=leg_species)

        self.rigid_body = leg_rb
        self.shape = leg_shape

    def get_body_frame_positions(self, body):
        return rigid_body.transform(body, self.shape)

    def body_to_injavis_lines(self, body, box_size, head_color="ff0000",
                              base_color="1c1c1c", attr_color="3d8c40"):

        assert(len(body.center.shape) == 1)
        body_pos = self.get_body_frame_positions(body)

        assert(len(body_pos.shape) == 2)
        assert(body_pos.shape[0] == 3)
        assert(body_pos.shape[1] == 3)

        box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
        head_def = f"def H \"sphere {self.head_particle_radius*2} {head_color}\""
        attr_def = f"def A \"sphere {self.attr_site_radius*2} {attr_color}\""
        base_def = f"def B \"sphere {self.base_particle_radius*2} {base_color}\""

        head_pos = body_pos[0]
        head_line = f"H {head_pos[0]} {head_pos[1]} {head_pos[2]}"

        attr_pos = body_pos[1]
        attr_line = f"A {attr_pos[0]} {attr_pos[1]} {attr_pos[2]}"

        base_pos = body_pos[2]
        base_line = f"B {base_pos[0]} {base_pos[1]} {base_pos[2]}"

        all_lines = [box_def, head_def, attr_def, base_def, head_line, attr_line, base_line, "eof"]

        return all_lines, box_def, [head_def, attr_def, base_def], [head_line, attr_line, base_line]


class TestLeg(unittest.TestCase):
    def test_write_injavis(self):
        leg = Leg(base_radius=4.0, head_height=5.0,
                  base_particle_radius=0.5,
                  attr_particle_pos_norm=0.5, attr_site_radius=1.0,
                  head_particle_radius=0.25)
        leg_rb = leg.rigid_body
        injavis_lines, _, _, _ = leg.body_to_injavis_lines(leg_rb, box_size=15.0)
        with open('leg.pos', 'w+') as of:
            of.write('\n'.join(injavis_lines))


if __name__ == "__main__":
    unittest.main()
