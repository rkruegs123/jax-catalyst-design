import pdb
from pathlib import Path
import unittest
from tqdm import tqdm

from jax import vmap, lax
import jax.numpy as jnp
from jax_md import rigid_body # FIXME: switch to mod_rigid_body after initial testing

from catalyst.spider_getter import SpiderInfo
from catalyst.shell_getter import ShellInfo

class ComplexInfo:
    def __init__(self,
                 # complex-specific arguments
                 initial_separation_coeff, vertex_to_bind_idx, displacement_fn,

                 # spider-specific arguments arguments
                 spider_base_radius, spider_head_height, spider_point_mass, spider_mass_err=1e-6
    ):
        self.vertex_to_bind_idx = vertex_to_bind_idx
        self.displacement_fn = displacement_fn

        self.spider_base_radius = spider_base_radius
        self.spider_head_height = spider_head_height
        self.spider_point_mass = spider_point_mass
        self.spider_mass_err = spider_mass_err

        self.load()

    def load(self):
        shell_info = ShellInfo() # note: won't change
        init_spider_info = SpiderInfo(
            self.spider_base_radius, self.spider_head_height,
            self.spider_point_mass, self.spider_mass_err)

        init_spider_center = init_spider_info.rigid_body.center

        vertex_to_bind = shell_info.rigid_body[self.vertex_to_bind_idx]
        disp_vector = self.displacement_fn(vertex_to_bind.center,
                                           jnp.mean(shell_info.rigid_body.center, axis=0))
        disp_vector /= jnp.linalg.norm(disp_vector)

        spider_center = vertex.center + disp_vector * shell_info.vertex_radius * initial_separation_coeff # shift spider away from vertex

        spider_rigid_body = rigid_body.RigidBody(
            center=jnp.array([spider_center]),
            orientation=rigid_body.Quaternion(jnp.array([vertex_to_bind.orientation.vec])))
        max_shell_species = jnp.max(shell_info.vertex_shape.point_species)
        spider_species = spider_info.shape.point_species + max_shell_species + 1
        spider_shape = spider_info.shape.set(point_species=spider_species)

        complex_shape = rigid_body.concatenate_shapes(shell_info.vertex_shape, spider_shape)
        complex_center = jnp.concatenate([shell_info.rigid_body.center, spider_rigid_body.center])
        complex_orientation = rigid_body.Quaternion(
            jnp.concatenate([shell_info.rigid_body.orientation.vec,
                             spider_rigid_body.orientation.vec]))

        complex_rigid_body = rigid_body.RigidBody(complex_center, complex_orientation)

        self.rigid_body = complex_rigid_body
        self.shape = complex_shape


    # FIXME: this is just a sketch
    def get_energy_fn(self):
        shell_energy_fn = self.shell_info.get_energy_fn()
        spider_energy_fn = self.spider_info.get_energy_fn()
        shell_spider_interaction_energy_fn = self.get_interaction_energy_fn()

        def complex_energy_fn(body: RigidBody):
            spider_body = body[-1]
            shell_body = body[:12]
            return shell_energy_fn(shell_body) + spider_energy_fn(spider_body) \
                + shell_spider_interaction_energy_fn(body)
