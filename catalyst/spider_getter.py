import pdb
from pathlib import Path
import unittest
from tqdm import tqdm

from jax import vmap, lax
import jax.numpy as jnp
from jax_md import rigid_body # FIXME: switch to mod_rigid_body after initial testing


class SpiderInfo:
    def __init__(self, base_radius, head_height,
                 base_particle_radius, head_particle_radius,
                 point_mass=1.0, mass_err=1e-6):
        self.base_radius = base_radius
        self.head_height = head_height
        self.base_particle_radius = base_particle_radius
        self.head_particle_radius = head_particle_radius
        self.mass_err = mass_err
        self.point_mass = point_mass
        self.load()

    def load(self):
        init_positions = self.get_positions()
        n_positions = init_positions.shape[0]
        spider_rigid_body = rigid_body.RigidBody(
            center=jnp.array([0.0, 0.0, 0.0]),
            orientation=rigid_body.Quaternion(jnp.array([1.0, 0.0, 0.0, 0.0])))

        masses = jnp.ones(n_positions) * self.point_mass + jnp.arange(n_positions) * self.mass_err
        spider_species = jnp.array([[0] * 5 + [1]]).flatten()
        spider_shape = rigid_body.point_union_shape(init_positions, masses).set(point_species=spider_species)

        self.rigid_body = spider_rigid_body
        self.shape = spider_shape

    def get_positions(self, z=0.0):
        spider_pos = jnp.zeros((5, 3))

        def scan_fn(spider_pos, i):
            x = self.base_radius * jnp.cos(i * 2 * jnp.pi / 5)
            y = self.base_radius * jnp.sin(i * 2 * jnp.pi / 5)
            spider_pos = spider_pos.at[i, :].set(jnp.array([x, y, z]))
            return spider_pos, i

        spider_base, _ = lax.scan(scan_fn, spider_pos, jnp.arange(len(spider_pos)))
        spider_head = jnp.array([[0., 0., -1 * (self.head_height + z)]])

        return jnp.array(jnp.concatenate([spider_base, spider_head]))

    def get_energy_fn(self):
        return lambda R, **kwargs: 0.0

    def get_body_frame_positions(self):
        body_pos = vmap(rigid_body.transform, (0, None))(self.rigid_body, self.shape)
        return body_pos


class TestSpiderInfo(unittest.TestCase):
    def test_no_errors(self):
        spider_info = SpiderInfo(base_radius=3.0, head_height=4.0,
                                 base_particle_radius=0.5, head_particle_radius=0.5,
                                 mass_err=1e-6)
        return

    def test_energy_fn_no_errors(self):
        spider_info = SpiderInfo(base_radius=3.0, head_height=4.0,
                                 base_particle_radius=0.5, head_particle_radius=0.5,
                                 mass_err=1e-6)
        energy_fn = spider_info.get_energy_fn()

        init_energy = energy_fn(spider_info.rigid_body)
        print(f"Initial energy: {init_energy}")


if __name__ == "__main__":
    unittest.main()
