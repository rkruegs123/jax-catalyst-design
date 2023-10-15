import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp

from jax import vmap, lax
import jax.numpy as jnp
from jax_md import energy, space
from jax_md import rigid_body

from catalyst.icosahedron_cargo.spider_getter import SpiderInfo
from catalyst.icosahedron_cargo.shell_getter import ShellInfo
from catalyst.icosahedron_cargo import utils

from jax.config import config
config.update('jax_enable_x64', True)


# Define options for leg pairs. Note that indices are w.r.t. the spider body pos
# NO_LEGS = jnp.array([], dtype=jnp.int32)
PENTAPOD_LEGS = jnp.array([
    [0, 5],
    [1, 5],
    [2, 5],
    [3, 5],
    [4, 5]
], dtype=jnp.int32)
BASE_LEGS = jnp.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 0]
], dtype=jnp.int32)

class ComplexInfo:
    def __init__(self,
                 # complex-specific arguments
                 initial_separation_coeff, vertex_to_bind_idx,
                 displacement_fn, shift_fn,

                 # spider-specific arguments arguments
                 spider_base_radius, spider_head_height,
                 spider_base_particle_radius, spider_head_particle_radius,
                 spider_point_mass, spider_mass_err=1e-6,

                 # misc.
                 verbose=True,

                 # legs
                 spider_bond_idxs=None, spider_leg_radius=0.5

    ):
        self.initial_separation_coeff = initial_separation_coeff
        self.vertex_to_bind_idx = vertex_to_bind_idx
        self.displacement_fn = displacement_fn
        self.shift_fn = shift_fn

        self.spider_base_radius = spider_base_radius
        self.spider_head_height = spider_head_height
        self.spider_base_particle_radius = spider_base_particle_radius
        self.spider_head_particle_radius = spider_head_particle_radius
        self.spider_point_mass = spider_point_mass
        self.spider_mass_err = spider_mass_err

        self.verbose = verbose

        self.spider_bond_idxs = spider_bond_idxs
        self.spider_leg_radius = spider_leg_radius

        self.load()

    def split_body(self, body):
        cargo_body = body[-1]
        spider_body = body[-2]
        shell_body = body[:12]
        return spider_body, shell_body, cargo_body

    def load(self):
        self.shell_info = ShellInfo(self.displacement_fn, self.shift_fn,
                                    verbose=self.verbose) # note: won't change
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
        max_shell_species = self.shell_info.shape.point_species[-1] # assumes monotonicity
        spider_species = spider_info.shape.point_species + max_shell_species + 1
        spider_info.shape = spider_info.shape.set(point_species=spider_species)
        self.spider_info = spider_info
        self.spider_radii = jnp.array([self.spider_info.base_particle_radius,
                                       self.spider_info.head_particle_radius])
        self.n_point_species = spider_species[-1] + 1 # note: assumes monotonicity


        # Get cargo information. FIXME: should maybe take as argument
        num_cargo_particles = 1
        assert(num_cargo_particles == 1)

        # cargo_center = jnp.array([jnp.mean(self.shell_info.rigid_body.center, axis=0)])
        cargo_center = jnp.array([[0.0, 0.0, 0.0]])
        cargo_orientation_vec = jnp.array([[1., 0., 0., 0.]])
        cargo_species = spider_species[-1] + 1
        cargo_shape = rigid_body.point_union_shape(onp.array([[1.0, 1.0, 1.0]]), 0.1).set(
            point_species=jnp.array([cargo_species])
        )
        self.cargo_radius = 0.25

        complex_shape = rigid_body.concatenate_shapes(
            self.shell_info.shape, spider_info.shape, cargo_shape)
        complex_center = jnp.concatenate([self.shell_info.rigid_body.center,
                                          spider_rigid_body.center,
                                          cargo_center], dtype=jnp.float64)
        complex_orientation = rigid_body.Quaternion(
            jnp.concatenate([self.shell_info.rigid_body.orientation.vec,
                             spider_rigid_body.orientation.vec,
                             cargo_orientation_vec], dtype=jnp.float64))

        complex_rigid_body = rigid_body.RigidBody(complex_center, complex_orientation)

        self.rigid_body = complex_rigid_body
        self.shape = complex_shape
        self.shape_species = onp.array(list(onp.zeros(12)) + [1] + [2], dtype=onp.int32).flatten()


    def get_leg_energy_fn(self, soft_eps, bond_diameter, leg_alpha):

        def leg_energy_fn(body):
            spider_body, shell_body, _ = self.split_body(body)
            spider_body_pos = self.spider_info.get_body_frame_positions(spider_body)
            shell_body_pos = self.shell_info.get_body_frame_positions(shell_body)

            shell_vertex_centers = shell_body_pos[::6]
            spider_bond_positions = spider_body_pos[self.spider_bond_idxs]


            all_dists = utils.mapped_dist_point_to_line(
                spider_bond_positions, shell_vertex_centers,
                self.displacement_fn)

            bond_energy_sm = jnp.sum(
                energy.soft_sphere(all_dists,
                                   epsilon=soft_eps,
                                   sigma=bond_diameter,
                                   alpha=jnp.array(leg_alpha)))

            return bond_energy_sm

        return leg_energy_fn


    def get_head_shell_energy_fn(
            self, morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_r_onset, morse_r_cutoff):

        zero_interaction = jnp.zeros((5, 5)) # FIXME: do we have to hardcode?

        morse_eps = zero_interaction.at[0, -2].set(morse_shell_center_spider_head_eps)
        morse_eps = morse_eps.at[-2, 0].set(morse_shell_center_spider_head_eps)

        morse_alpha = zero_interaction.at[0, -2].set(morse_shell_center_spider_head_alpha)
        morse_alpha = morse_alpha.at[-2, 0].set(morse_shell_center_spider_head_alpha)

        morse_sigma = zero_interaction.at[0, -2].set(self.shell_info.vertex_radius + self.spider_radii[-1])
        morse_sigma = morse_sigma.at[-2, 0].set(self.shell_info.vertex_radius + self.spider_radii[-1])

        pair_energy_morse = energy.morse_pair(
            self.displacement_fn,
            # species=self.n_point_species,
            species=5,
            sigma=morse_sigma, epsilon=morse_eps, alpha=morse_alpha,
            r_onset=morse_r_onset, r_cutoff=morse_r_cutoff
        )

        return rigid_body.point_energy(pair_energy_morse, self.shape, self.shape_species)


    def get_interaction_energy_fn(
            self, morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_r_onset, morse_r_cutoff,
            soft_eps,
            ss_shell_center_spider_leg_alpha
    ):
        # zero_interaction = jnp.zeros((self.n_point_species, self.n_point_species))
        zero_interaction = jnp.zeros((5, 5)) # FIXME: do we have to hardcode?

        soft_sphere_eps = zero_interaction.at[0, 2].set(soft_eps) # icosahedral centers repel catalyst centers
        soft_sphere_eps = soft_sphere_eps.at[2, 0].set(soft_eps) # symmetry

        soft_sphere_eps = soft_sphere_eps.at[4, 0].set(soft_eps)
        soft_sphere_eps = soft_sphere_eps.at[4, 2].set(soft_eps)
        soft_sphere_eps = soft_sphere_eps.at[4, 3].set(soft_eps)
        soft_sphere_eps = soft_sphere_eps.at[0, 4].set(soft_eps)
        soft_sphere_eps = soft_sphere_eps.at[2, 4].set(soft_eps)
        soft_sphere_eps = soft_sphere_eps.at[3, 4].set(soft_eps)

        soft_sphere_sigma = zero_interaction.at[0, 2].set(self.shell_info.vertex_radius + self.spider_radii[0]) # icosahedral centers repel catalyst centers
        soft_sphere_sigma = soft_sphere_sigma.at[2, 0].set(self.shell_info.vertex_radius + self.spider_radii[0])

        soft_sphere_sigma = soft_sphere_sigma.at[0, 4].set(self.shell_info.vertex_radius + self.cargo_radius)
        soft_sphere_sigma = soft_sphere_sigma.at[4, 0].set(self.shell_info.vertex_radius + self.cargo_radius)
        soft_sphere_sigma = soft_sphere_sigma.at[2, 4].set(self.spider_radii[0] + self.cargo_radius)
        soft_sphere_sigma = soft_sphere_sigma.at[4, 2].set(self.spider_radii[0] + self.cargo_radius)
        soft_sphere_sigma = soft_sphere_sigma.at[3, 4].set(self.spider_radii[1] + self.cargo_radius)
        soft_sphere_sigma = soft_sphere_sigma.at[4, 3].set(self.spider_radii[1] + self.cargo_radius)


        soft_sphere_sigma = jnp.where(soft_sphere_sigma == 0.0, 1e-5, soft_sphere_sigma) # avoids nans


        pair_energy_soft = energy.soft_sphere_pair(
            self.displacement_fn,
            # species=self.n_point_species,
            species=5,
            sigma=soft_sphere_sigma, epsilon=soft_sphere_eps)

        soft_energy_fn = rigid_body.point_energy(pair_energy_soft, self.shape, self.shape_species)
        morse_energy_fn = self.get_head_shell_energy_fn(
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_r_onset, morse_r_cutoff)
        base_energy_fn = lambda body, **kwargs: soft_energy_fn(body, **kwargs) + morse_energy_fn(body, **kwargs)

        # Construct the leg energy function
        if self.spider_bond_idxs is not None:
            leg_energy_fn = self.get_leg_energy_fn(
                soft_eps, 2 * self.spider_leg_radius, ss_shell_center_spider_leg_alpha)

            energy_fn = lambda body: base_energy_fn(body) + leg_energy_fn(body)

            return energy_fn
        return base_energy_fn


    def get_energy_components_fn(
            self,

            # Shell-spider interaction energy parameters
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_r_onset=10.0, morse_r_cutoff=12.0,

            # Shell-shell interaction energy parameters
            morse_ii_eps=10.0, morse_ii_alpha=5.0,

            # Misc. parameters
            soft_eps=10000.0,

            # Leg parameters
            ss_shell_center_spider_leg_alpha=2.0
    ):

        shell_energy_fn = self.shell_info.get_energy_fn(
            morse_ii_eps, morse_ii_alpha, soft_eps)
        spider_energy_fn = self.spider_info.get_energy_fn()
        shell_spider_interaction_energy_fn = self.get_interaction_energy_fn(
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_r_onset, morse_r_cutoff,
            soft_eps,
            ss_shell_center_spider_leg_alpha
        )

        def energy_components_fn(body: rigid_body.RigidBody, **kwargs):
            spider_body, shell_body, _ = self.split_body(body)
            shell_energy = shell_energy_fn(shell_body, **kwargs)
            spider_energy = spider_energy_fn(spider_body, **kwargs)
            interaction_energy = shell_spider_interaction_energy_fn(body, **kwargs)
            return shell_energy, spider_energy, interaction_energy
        return energy_components_fn

    def get_energy_fn(
            self,

            # Shell-spider interaction energy parameters
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_r_onset=10.0, morse_r_cutoff=12.0,

            # Shell-shell interaction energy parameters
            morse_ii_eps=10.0, morse_ii_alpha=5.0,

            # Misc. parameters
            soft_eps=10000.0,

            # Leg parameters
            ss_shell_center_spider_leg_alpha=2.0

    ):
        energy_components_fn = self.get_energy_components_fn(
            morse_shell_center_spider_head_eps, morse_shell_center_spider_head_alpha,
            morse_r_onset, morse_r_cutoff,
            morse_ii_eps, morse_ii_alpha,
            soft_eps,
            ss_shell_center_spider_leg_alpha)

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

        spider_body, shell_body, cargo_body = self.split_body(body)
        _, spider_box_def, spider_type_defs, spider_pos = self.spider_info.body_to_injavis_lines(
            spider_body, box_size, spider_head_color, spider_base_color)
        _, shell_box_def, shell_type_defs, shell_pos = self.shell_info.body_to_injavis_lines(
            shell_body, box_size, shell_patch_radius, shell_vertex_color, shell_patch_color)

        self.cargo_radius
        cargo_color = "63a8bc"
        cargo_def = f"def C \"sphere {self.cargo_radius*2} {cargo_color}\""
        cargo_pos = body.center[-1]
        cargo_pos_line = f"C {cargo_pos[0]} {cargo_pos[1]} {cargo_pos[2]}"

        assert(spider_box_def == shell_box_def)
        box_def = spider_box_def
        type_defs = shell_type_defs + spider_type_defs + [cargo_def]
        positions = shell_pos + spider_pos + [cargo_pos_line]
        # type_defs = spider_type_defs + [cargo_def]
        # positions = spider_pos + [cargo_pos_line]
        all_lines = [box_def] + type_defs + positions + ["eof"]
        return all_lines, box_def, type_defs, positions



class TestComplexInfo(unittest.TestCase):

    def _test_init(self):
        displacement_fn, shift_fn = space.free()
        complex_info = ComplexInfo(
                initial_separation_coeff=0.1, vertex_to_bind_idx=5,
                displacement_fn=displacement_fn, shift_fn=shift_fn,
                spider_base_radius=5.0, spider_head_height=4.0,
                spider_base_particle_radius=0.5, spider_head_particle_radius=0.5,
                spider_point_mass=1.0, spider_mass_err=1e-6
        )

        body_pos = complex_info.get_body_frame_positions(complex_info.rigid_body)
        return

    def test_injavis(self):
        displacement_fn, shift_fn = space.free()
        complex_info = ComplexInfo(
                initial_separation_coeff=0.1, vertex_to_bind_idx=5,
                displacement_fn=displacement_fn, shift_fn=shift_fn,
                spider_base_radius=5.0, spider_head_height=4.0,
                spider_base_particle_radius=0.5, spider_head_particle_radius=0.5,
                spider_point_mass=1.0, spider_mass_err=1e-6
        )

        body = complex_info.rigid_body
        complex_lines, _, _, _ = complex_info.body_to_injavis_lines(body, box_size=30.0)
        with open('complex_state.pos', 'w+') as of:
            of.write('\n'.join(complex_lines))

    def test_energy_fn(self):

        displacement_fn, shift_fn = space.free()
        complex_info = ComplexInfo(
                initial_separation_coeff=0.1, vertex_to_bind_idx=5,
                displacement_fn=displacement_fn, shift_fn=shift_fn,
                spider_base_radius=5.0, spider_head_height=4.0,
                spider_base_particle_radius=0.5, spider_head_particle_radius=0.5,
                spider_point_mass=1.0, spider_mass_err=1e-6
        )

        energy_fn = self.complex_info.get_energy_fn(
            morse_shell_center_spider_head_eps=jnp.exp(9.21), morse_shell_center_spider_head_alpha=1.5,
        )
        init_energy = energy_fn(self.complex_info.rigid_body)
        print(f"Initial energy: {init_energy}")

        energy_components_fn = self.complex_info.get_energy_components_fn(
            morse_shell_center_spider_head_eps=jnp.exp(9.21), morse_shell_center_spider_head_alpha=1.5,
        )
        shell_energy, spider_energy, interaction_energy = energy_components_fn(self.complex_info.rigid_body)
        print(f"Initial shell energy: {shell_energy}")
        print(f"Initial spider energy: {spider_energy}")
        print(f"Initial interaction energy: {interaction_energy}")


if __name__ == "__main__":
    unittest.main()
