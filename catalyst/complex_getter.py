import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp

from jax import vmap, lax
import jax.numpy as jnp
# from jax_md import rigid_body, energy, space # FIXME: switch to mod_rigid_body after initial testing
from jax_md import energy, space

import catalyst.rigid_body as rigid_body
from catalyst.spider_getter import SpiderInfo
from catalyst.shell_getter import ShellInfo
from catalyst import utils

from jax.config import config
config.update('jax_enable_x64', True)


class ComplexInfo:
    def __init__(self,
                 # complex-specific arguments
                 initial_separation_coeff, vertex_to_bind_idx, displacement_fn,

                 # spider-specific arguments arguments
                 spider_base_radius, spider_head_height,
                 spider_base_particle_radius, spider_head_particle_radius,
                 spider_point_mass, spider_mass_err=1e-6,

                 # misc.
                 verbose=True
    ):
        self.initial_separation_coeff = initial_separation_coeff
        self.vertex_to_bind_idx = vertex_to_bind_idx
        self.displacement_fn = displacement_fn

        self.spider_base_radius = spider_base_radius
        self.spider_head_height = spider_head_height
        self.spider_base_particle_radius = spider_base_particle_radius
        self.spider_head_particle_radius = spider_head_particle_radius
        self.spider_point_mass = spider_point_mass
        self.spider_mass_err = spider_mass_err

        self.verbose = verbose

        self.load()

    def split_body(self, body):
        spider_body = body[-1]
        shell_body = body[:12]
        return spider_body, shell_body

    def load(self):
        self.shell_info = ShellInfo(self.displacement_fn, verbose=self.verbose) # note: won't change
        spider_info = SpiderInfo(
            self.spider_base_radius, self.spider_head_height,
            self.spider_base_particle_radius, self.spider_head_particle_radius,
            self.spider_point_mass, self.spider_mass_err)

        init_spider_center = spider_info.rigid_body.center

        vertex_to_bind = self.shell_info.rigid_body[self.vertex_to_bind_idx]
        disp_vector = self.displacement_fn(vertex_to_bind.center,
                                           jnp.mean(self.shell_info.rigid_body.center, axis=0))
        disp_vector /= jnp.linalg.norm(disp_vector)

        spider_center = vertex_to_bind.center + disp_vector * self.shell_info.vertex_radius * self.initial_separation_coeff # shift spider away from vertex

        spider_rigid_body = rigid_body.RigidBody(
            center=jnp.array([spider_center]),
            orientation=rigid_body.Quaternion(jnp.array([vertex_to_bind.orientation.vec])))
        spider_info.rigid_body = spider_rigid_body
        max_shell_species = self.shell_info.vertex_shape.point_species[-1] # assumes monotonicity
        spider_species = spider_info.shape.point_species + max_shell_species + 1
        spider_info.shape = spider_info.shape.set(point_species=spider_species)
        self.spider_info = spider_info
        self.n_point_species = spider_species[-1] + 1 # note: assumes monotonicity

        complex_shape = rigid_body.concatenate_shapes(self.shell_info.vertex_shape, spider_info.shape)
        complex_center = jnp.concatenate([self.shell_info.rigid_body.center, spider_rigid_body.center], dtype=jnp.float64)
        complex_orientation = rigid_body.Quaternion(
            jnp.concatenate([self.shell_info.rigid_body.orientation.vec,
                             spider_rigid_body.orientation.vec], dtype=jnp.float64))

        complex_rigid_body = rigid_body.RigidBody(complex_center, complex_orientation)

        self.rigid_body = complex_rigid_body
        self.shape = complex_shape
        self.shape_species = onp.array(list(onp.zeros(12)) + [1], dtype=onp.int32).flatten()


    def get_interaction_energy_fn(
            self, morse_shell_center_spider_base_eps, morse_shell_center_spider_base_alpha,
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            soft_eps, morse_r_onset, morse_r_cutoff

    ):
        spider_radii = jnp.array([self.spider_info.base_particle_radius,
                                  self.spider_info.head_particle_radius])
        # zero_interaction = jnp.zeros((self.n_point_species, self.n_point_species))
        zero_interaction = jnp.zeros((4, 4)) # FIXME: do we have to hardcode?

        morse_eps = zero_interaction.at[0, 2:-1].set(morse_shell_center_spider_base_eps)
        morse_eps = morse_eps.at[2:-1, 0].set(morse_shell_center_spider_base_eps) # symmetry
        morse_eps = morse_eps.at[0, -1].set(morse_shell_center_spider_head_eps)
        morse_eps = morse_eps.at[-1, 0].set(morse_shell_center_spider_head_eps)

        morse_alpha = zero_interaction.at[0, 2:-1].set(morse_shell_center_spider_base_alpha)
        morse_alpha = morse_alpha.at[2:-1, 0].set(morse_shell_center_spider_base_alpha)
        morse_alpha = morse_alpha.at[0, -1].set(morse_shell_center_spider_head_alpha)
        morse_alpha = morse_alpha.at[-1, 0].set(morse_shell_center_spider_head_alpha)

        soft_sphere_eps = zero_interaction.at[0, 2:].set(soft_eps) # icosahedral centers repel catalyst centers
        soft_sphere_eps = soft_sphere_eps.at[2:, 0].set(soft_eps) # symmetry

        soft_sphere_sigma = zero_interaction.at[0, 2:].set(self.shell_info.vertex_radius + spider_radii) # icosahedral centers repel catalyst centers
        soft_sphere_sigma = soft_sphere_sigma.at[2:, 0].set(self.shell_info.vertex_radius + spider_radii)
        soft_sphere_sigma = jnp.where(soft_sphere_sigma == 0.0, 1e-5, soft_sphere_sigma) # avoids nans


        pair_energy_soft = energy.soft_sphere_pair(
            self.displacement_fn,
            # species=self.n_point_species,
            species=4,
            sigma=soft_sphere_sigma, epsilon=soft_sphere_eps)
        pair_energy_morse = energy.morse_pair(
            self.displacement_fn,
            # species=self.n_point_species,
            species=4,
            sigma=0.0, epsilon=morse_eps, alpha=morse_alpha,
            r_onset=morse_r_onset, r_cutoff=morse_r_cutoff
        )
        pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
        energy_fn = rigid_body.point_energy(pair_energy_fn, self.shape, self.shape_species)
        return energy_fn
        # return lambda R, **kwargs: 0.0


    def get_energy_components_fn(self,
            # Shell-spider interaction energy parameters
            morse_shell_center_spider_base_eps, morse_shell_center_spider_base_alpha,
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,

            # Shell-shell interaction energy parameters
            morse_ii_eps=10.0, morse_ii_alpha=5.0,

            # Misc. parameters
            soft_eps=10000.0, morse_r_onset=10.0, morse_r_cutoff=12.0):

        shell_energy_fn = self.shell_info.get_energy_fn(
            morse_ii_eps, morse_ii_alpha, soft_eps,
            morse_r_onset, morse_r_cutoff)
        spider_energy_fn = self.spider_info.get_energy_fn()
        shell_spider_interaction_energy_fn = self.get_interaction_energy_fn(
            morse_shell_center_spider_base_eps, morse_shell_center_spider_base_alpha,
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            soft_eps, morse_r_onset, morse_r_cutoff
        )

        def energy_components_fn(body: rigid_body.RigidBody, **kwargs):
            spider_body, shell_body = self.split_body(body)
            shell_energy = shell_energy_fn(shell_body, **kwargs)
            spider_energy = spider_energy_fn(spider_body, **kwargs)
            interaction_energy = shell_spider_interaction_energy_fn(body, **kwargs)
            return shell_energy, spider_energy, interaction_energy
        return energy_components_fn
        
    def get_energy_fn(
            self,

            # Shell-spider interaction energy parameters
            morse_shell_center_spider_base_eps, morse_shell_center_spider_base_alpha,
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,

            # Shell-shell interaction energy parameters
            morse_ii_eps=10.0, morse_ii_alpha=5.0,

            # Misc. parameters
            soft_eps=10000.0, morse_r_onset=10.0, morse_r_cutoff=12.0

    ):
        energy_components_fn = self.get_energy_components_fn(
            morse_shell_center_spider_base_eps, morse_shell_center_spider_base_alpha,
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_ii_eps, morse_ii_alpha,
            soft_eps, morse_r_onset, morse_r_cutoff)

        def complex_energy_fn(body: rigid_body.RigidBody, **kwargs):
            shell_energy, spider_energy, interaction_energy = energy_components_fn(body, **kwargs)
            return shell_energy + spider_energy + interaction_energy

        return complex_energy_fn

    def get_body_frame_positions(self, body):
        raise NotImplementedError
        # return utils.get_body_frame_positions(body, self.shape)

    def body_to_injavis_lines(
            self, body, box_size,
            shell_patch_radius=0.5, shell_vertex_color="43a5be", shell_patch_color="4fb06d",
            spider_head_color="ff0000", spider_base_color="1c1c1c"):

        spider_body, shell_body = self.split_body(body)
        _, spider_box_def, spider_type_defs, spider_pos = self.spider_info.body_to_injavis_lines(
            spider_body, box_size, spider_head_color, spider_base_color)
        _, shell_box_def, shell_type_defs, shell_pos = self.shell_info.body_to_injavis_lines(
            shell_body, box_size, shell_patch_radius, shell_vertex_color, shell_patch_color)

        assert(spider_box_def == shell_box_def)
        box_def = spider_box_def
        type_defs = shell_type_defs + spider_type_defs
        positions = shell_pos + spider_pos
        all_lines = [box_def] + type_defs + positions + ["eof"]
        return all_lines, box_def, type_defs, positions



class TestComplexInfo(unittest.TestCase):
    def _test_init(self):
        displacement_fn, _ = space.free()
        complex_info = ComplexInfo(
            initial_separation_coeff=0.1, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn,
            spider_base_radius=5.0, spider_head_height=4.0,
            spider_base_particle_radius=0.5, spider_head_particle_radius=0.5,
            spider_point_mass=1.0, spider_mass_err=1e-6
        )

        body_pos = complex_info.get_body_frame_positions(complex_info.rigid_body)

        return

    def test_energy_fn(self):
        displacement_fn, _ = space.free()
        complex_info = ComplexInfo(
            initial_separation_coeff=0.1, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn,
            spider_base_radius=5.0, spider_head_height=5.0,
            spider_base_particle_radius=0.5, spider_head_particle_radius=0.5,
            spider_point_mass=1.0, spider_mass_err=1e-6
        )
        energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_base_eps=2.5, morse_shell_center_spider_base_alpha=1.0,
            morse_shell_center_spider_head_eps=jnp.exp(9.21), morse_shell_center_spider_head_alpha=1.5,
        )
        init_energy = energy_fn(complex_info.rigid_body)
        print(f"Initial energy: {init_energy}")

        energy_components_fn = complex_info.get_energy_components_fn(
            morse_shell_center_spider_base_eps=2.5, morse_shell_center_spider_base_alpha=1.0,
            morse_shell_center_spider_head_eps=jnp.exp(9.21), morse_shell_center_spider_head_alpha=1.5,
        )
        shell_energy, spider_energy, interaction_energy = energy_components_fn(complex_info.rigid_body)
        print(f"Initial shell energy: {shell_energy}")
        print(f"Initial spider energy: {spider_energy}")
        print(f"Initial interaction energy: {interaction_energy}")
        

if __name__ == "__main__":
    unittest.main()
