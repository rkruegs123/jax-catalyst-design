import pdb
from pathlib import Path
import unittest
from tqdm import tqdm

from jax import vmap, lax
import jax.numpy as jnp
# from jax_md import rigid_body

import catalyst.icosahedron_ext_rigid.rigid_body as rigid_body
from catalyst.icosahedron_ext_rigid import utils

from jax.config import config
config.update('jax_enable_x64', True)


class Spider:
    def __init__(self, base_radius, head_height,
                 base_particle_radius,
                 attr_particle_pos_norm, attr_site_radius,
                 head_particle_radius,
                 point_mass=1.0, mass_err=1e-6):
        self.base_radius = base_radius
        self.head_height = head_height
        self.base_particle_radius = base_particle_radius
        self.head_particle_radius = head_particle_radius
        self.mass_err = mass_err
        self.point_mass = point_mass
        self.attr_site_radius = attr_site_radius
        self.attr_particle_pos_norm = attr_particle_pos_norm
        self.particle_radii = jnp.array([base_particle_radius, attr_site_radius, head_particle_radius])
        self.n_legs = 5

        self.load()

    def load(self):
        init_positions = self.get_positions()
        n_positions = init_positions.shape[0]
        spider_rigid_body = rigid_body.RigidBody(
            center=jnp.array([0.0, 0.0, 0.0]),
            orientation=rigid_body.Quaternion(jnp.array([1.0, 0.0, 0.0, 0.0])))

        masses = jnp.ones(n_positions) * self.point_mass + jnp.arange(n_positions) * self.mass_err
        spider_species = jnp.array([[0] * 5 + [1] * 5 + [2]]).flatten()
        spider_shape = rigid_body.point_union_shape(init_positions, masses).set(point_species=spider_species)

        self.rigid_body = spider_rigid_body
        self.shape = spider_shape

    def get_positions(self, z=0.0):
        base_pos = jnp.zeros((self.n_legs, 3))
        attr_site_pos = jnp.zeros((self.n_legs, 3))

        z_attr = z + self.head_height * self.attr_particle_pos_norm
        attr_plane_radius = self.base_radius * (1.0 - self.attr_particle_pos_norm)

        def scan_fn(carry, i):
            base_pos, attr_site_pos = carry
            x = self.base_radius * jnp.cos(i * 2 * jnp.pi / self.n_legs)
            y = self.base_radius * jnp.sin(i * 2 * jnp.pi / self.n_legs)
            base_pos = base_pos.at[i, :].set(jnp.array([x, y, z]))

            x_attr = attr_plane_radius * jnp.cos(i * 2 * jnp.pi / self.n_legs) # x * self.attr_radius / self.base_radius
            y_attr = attr_plane_radius * jnp.sin(i * 2 * jnp.pi / self.n_legs)

            attr_site_pos = attr_site_pos.at[i, :].set(jnp.array([x_attr, y_attr, z_attr]))
            return (base_pos, attr_site_pos), i

        (base_pos, attr_site_pos), _ = lax.scan(scan_fn, (base_pos, attr_site_pos), jnp.arange(self.n_legs))
        head_pos = jnp.array([[0., 0., 1 * (self.head_height + z)]])

        return jnp.array(jnp.concatenate([base_pos, attr_site_pos, head_pos]))

    def get_energy_fn(self):
        return lambda R, **kwargs: 0.0

    def get_body_frame_positions(self, body):
        return rigid_body.transform(body, self.shape)

    # note: body is only a single state, not a trajectory
    def body_to_injavis_lines(
            self, body, box_size,
            head_color="ff0000", attr_color="5eff33", base_color="1c1c1c"):

        assert(len(body.center.shape) == 1)
        body_pos = self.get_body_frame_positions(body)

        assert(len(body_pos.shape) == 2)
        assert(body_pos.shape[0] == 11)
        assert(body_pos.shape[1] == 3)

        box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
        head_def = f"def H \"sphere {self.head_particle_radius*2} {head_color}\""
        attr_def = f"def A \"sphere {self.attr_site_radius*2} {attr_color}\""
        base_def = f"def B \"sphere {self.base_particle_radius*2} {base_color}\""

        head_pos = body_pos[-1]
        head_line = f"H {head_pos[0]} {head_pos[1]} {head_pos[2]}"
        all_lines = [box_def, head_def, attr_def, base_def, head_line]

        for idx in range(self.n_legs):
            base_pos = body_pos[idx]
            base_line = f"B {base_pos[0]} {base_pos[1]} {base_pos[2]}"
            all_lines.append(base_line)

            attr_pos = body_pos[self.n_legs + idx]
            attr_line = f"A {attr_pos[0]} {attr_pos[1]} {attr_pos[2]}"
            all_lines.append(attr_line)

        # Return: all lines, box info, particle types, positions
        return all_lines, box_def, [head_def, attr_def, base_def], all_lines[4:]


class TestSpider(unittest.TestCase):
    def test_no_errors(self):
        spider = Spider(base_radius=3.0, head_height=4.0,
                        base_particle_radius=0.5,
                        attr_particle_pos_norm=0.5,
                        attr_site_radius=0.3,
                        head_particle_radius=0.5,
                        mass_err=1e-6)
        return

    def test_energy_fn_no_errors(self):
        spider = Spider(base_radius=3.0, head_height=4.0,
                        base_particle_radius=0.5,
                        attr_particle_pos_norm=0.5,
                        attr_site_radius=0.3,
                        head_particle_radius=0.5,
                        mass_err=1e-6)
        energy_fn = spider.get_energy_fn()

        init_energy = energy_fn(spider.rigid_body)
        print(f"Initial energy: {init_energy}")

    def test_injavis(self):
        box_size = 30.0
        spider = Spider(base_radius=3.0, head_height=4.0,
                        base_particle_radius=0.5,
                        attr_particle_pos_norm=0.5,
                        attr_site_radius=0.3,
                        head_particle_radius=0.5,
                        mass_err=1e-6)
        spider_rb = spider.rigid_body
        traj_injavis_lines = spider.body_to_injavis_lines(spider_rb, box_size=box_size)[0]
        traj_path = f"maybe_right_injavis.pos"
        with open(traj_path, 'w+') as of:
            of.write('\n'.join(traj_injavis_lines))


if __name__ == "__main__":
    unittest.main()
