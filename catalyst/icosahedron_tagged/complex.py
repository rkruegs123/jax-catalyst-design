import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp
import matplotlib.pyplot as plt
import time

from jax import vmap, lax, jit, random
import jax.numpy as jnp
from jax_md import energy, space, simulate
# from jax_md import rigid_body

import catalyst.icosahedron_tagged.rigid_body as rigid_body
from catalyst.icosahedron_tagged.spider import Spider
from catalyst.icosahedron_tagged.shell import Shell
from catalyst.icosahedron_tagged import utils

from jax.config import config
config.update('jax_enable_x64', True)



class Complex:
    def __init__(self,
                 # complex-specific arguments
                 initial_separation_coeff, vertex_to_bind_idx,
                 displacement_fn, shift_fn,

                 # spider-specific arguments arguments
                 spider_base_radius, spider_head_height,
                 spider_base_particle_radius,
                 spider_attr_particle_radius, spider_head_particle_radius,
                 spider_point_mass, spider_mass_err=1e-6,

                 # misc.
                 verbose=True,

                 # legs
                 bond_radius=0.25, bond_alpha=2.0,
                 rel_attr_particle_pos=0.5,
                 add_spider_bonds=True,
                 opt_leg_springs=False,
                 leg_spring_eps=None,

                 head_particle_eps=100000.0
    ):
        self.n_legs = 5
        self.add_spider_bonds = add_spider_bonds
        self.opt_leg_springs = opt_leg_springs
        self.leg_spring_eps = leg_spring_eps
        self.head_particle_eps = head_particle_eps

        self.initial_separation_coeff = initial_separation_coeff
        self.vertex_to_bind_idx = vertex_to_bind_idx
        self.displacement_fn = displacement_fn
        self.shift_fn = shift_fn

        self.spider_base_radius = spider_base_radius
        self.spider_head_height = spider_head_height
        self.spider_base_particle_radius = spider_base_particle_radius
        self.spider_attr_particle_radius = spider_attr_particle_radius
        self.spider_head_particle_radius = spider_head_particle_radius
        self.spider_point_mass = spider_point_mass
        self.spider_mass_err = spider_mass_err

        self.verbose = verbose

        self.bond_radius = bond_radius
        self.bond_alpha = bond_alpha

        self.rel_attr_particle_pos = rel_attr_particle_pos

        self.load()

    def split_body(self, body):
        spider_body = body[-5:]
        shell_body = body[:12]
        return spider_body, shell_body

    def load(self):
        self.shell = Shell(self.displacement_fn, self.shift_fn,
                           verbose=self.verbose) # note: won't change

        vertex_to_bind = self.shell.rigid_body[self.vertex_to_bind_idx]
        vertex_mask = [0]*6*12
        vertex_mask[self.vertex_to_bind_idx*6] = 1
        vertex_mask += [0, 1, 0]*5
        vertex_mask = jnp.array(vertex_mask)
        self.mask = vertex_mask



        # Get different target positions
        target_positions = jnp.zeros((5, 3))
        z = self.spider_head_height
        def scan_fn(target_positions, i):
            y = self.spider_base_radius * jnp.cos(i * 2 * jnp.pi / 5)
            x = self.spider_base_radius * jnp.sin(i * 2 * jnp.pi / 5)
            target_positions = target_positions.at[i, :].set(jnp.array([x, y, z]))
            return target_positions, i
        target_positions, _ = lax.scan(scan_fn, target_positions, jnp.arange(5))

        new_target_positions = rigid_body.quaternion_rotate(vertex_to_bind.orientation, target_positions)

        spider = Spider(
            self.displacement_fn, self.shift_fn,
            self.spider_base_radius, self.spider_head_height,
            self.spider_base_particle_radius,
            self.rel_attr_particle_pos, self.spider_attr_particle_radius,
            self.spider_head_particle_radius,
            self.spider_point_mass, self.spider_mass_err,
            target_positions=new_target_positions
        )
        leg_length = spider.legs[0].leg_length

        assert(spider.n_legs == self.n_legs)


        # Next

        init_spider_center = spider.rigid_body.center
        disp_vector = self.displacement_fn(vertex_to_bind.center,
                                           jnp.mean(self.shell.rigid_body.center, axis=0))
        disp_vector /= jnp.linalg.norm(disp_vector)

        leg_center = vertex_to_bind.center + disp_vector * self.shell.vertex_radius * self.initial_separation_coeff # shift spider away from vertex
        spider_center = jnp.full((5, 3), leg_center)



        # FIXME: may have to change this
        # spider_orientation = vertex_to_bind.orientation * spider.rigid_body.orientation
        spider_orientation = spider.rigid_body.orientation

        spider_rigid_body = rigid_body.RigidBody(
            center=spider_center,
            orientation=spider_orientation)
        spider.rigid_body = spider_rigid_body


        # Get a shape that will only be used for the interaction energy function
        max_shell_species = self.shell.shape.point_species[-1] # assumes monotonicity
        spider_species = spider.shape.point_species + max_shell_species + 1
        # spider.shape = spider.shape.set(point_species=spider_species)
        spider_in_complex_shape = spider.shape.set(point_species=spider_species)
        self.spider_in_complex_shape = spider_in_complex_shape
        self.spider = spider
        self.spider_radii = jnp.array([self.spider.head_particle_radius,
                                       self.spider.attr_site_radius,
                                       self.spider.base_particle_radius])
        self.n_point_species = spider_species[-1] + 1 # note: assumes monotonicity

        complex_shape = rigid_body.concatenate_shapes(self.shell.shape, spider_in_complex_shape)
        complex_center = jnp.concatenate([self.shell.rigid_body.center, spider_rigid_body.center], dtype=jnp.float64)
        complex_orientation = rigid_body.Quaternion(
            jnp.concatenate([self.shell.rigid_body.orientation.vec,
                             spider_rigid_body.orientation.vec], dtype=jnp.float64))

        complex_rigid_body = rigid_body.RigidBody(complex_center, complex_orientation)

        self.rigid_body = complex_rigid_body
        self.shape = complex_shape
        self.shape_species = onp.array(list(onp.zeros(12)) + [1]*spider.n_legs, dtype=onp.int32).flatten()


    def get_rep_bond_energy_fn(self, soft_eps, bond_radius, bond_alpha):

        def single_leg_rep(l_idx, all_leg_space_frame_pos, all_vertex_space_frame_pos):
            leg_start_idx = 3*l_idx
            leg_end_idx = leg_start_idx+3

            leg_space_frame = all_leg_space_frame_pos[leg_start_idx:leg_end_idx]

            leg_bond_idxs = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
            leg_bond_positions = leg_space_frame[leg_bond_idxs]

            all_dists = utils.mapped_dist_point_to_line(
                leg_bond_positions, all_vertex_space_frame_pos, self.displacement_fn)

            bond_energy_sm = jnp.sum(
                energy.soft_sphere(all_dists,
                                   epsilon=soft_eps,
                                   sigma=bond_radius + self.shell.vertex_radius,
                                   alpha=jnp.array(bond_alpha)))

            return bond_energy_sm

        def base_bond_rep(bond_spider_idxs, all_leg_space_frame_pos, all_vertex_space_frame_pos):
            l_idx1, l_idx2 = bond_spider_idxs

            leg1_base_pos = all_leg_space_frame_pos[3*l_idx1+2]
            leg2_base_pos = all_leg_space_frame_pos[3*l_idx2+2]

            bond_positions = jnp.array([[leg1_base_pos, leg2_base_pos]])

            all_dists = utils.mapped_dist_point_to_line(
                bond_positions, all_vertex_space_frame_pos, self.displacement_fn)

            bond_energy_sm = jnp.sum(
                energy.soft_sphere(all_dists,
                                   epsilon=soft_eps,
                                   sigma=bond_radius + self.shell.vertex_radius,
                                   alpha=jnp.array(bond_alpha)))

            return bond_energy_sm


        def rep_bond_energy_fn(body):
            spider_body, shell_body = self.split_body(body)

            spider_space_frame_pos = vmap(self.spider.legs[0].get_body_frame_positions)(spider_body).reshape(-1, 3)
            shell_body_pos = self.shell.get_body_frame_positions(shell_body)
            shell_vertex_centers = shell_body_pos[::6]

            """
            all_leg_rep_vals = vmap(single_leg_rep, (0, None, None))(
                jnp.arange(5), # note: really self.spider.n_legs
                spider_space_frame_pos, shell_vertex_centers)
            return all_leg_rep_vals.sum()
            """
            rep_val = single_leg_rep(0, spider_space_frame_pos, shell_vertex_centers)
            rep_val += single_leg_rep(1, spider_space_frame_pos, shell_vertex_centers)
            rep_val += single_leg_rep(2, spider_space_frame_pos, shell_vertex_centers)
            rep_val += single_leg_rep(3, spider_space_frame_pos, shell_vertex_centers)
            rep_val += single_leg_rep(4, spider_space_frame_pos, shell_vertex_centers)
            if self.add_spider_bonds:
                rep_val += base_bond_rep(jnp.array([0, 1]), spider_space_frame_pos, shell_vertex_centers)
                rep_val += base_bond_rep(jnp.array([1, 2]), spider_space_frame_pos, shell_vertex_centers)
                # rep_val += base_bond_rep(jnp.array([2, 3]), spider_space_frame_pos, shell_vertex_centers)
                rep_val += base_bond_rep(jnp.array([3, 4]), spider_space_frame_pos, shell_vertex_centers)

                if self.opt_leg_springs:
                    rep_val += base_bond_rep(jnp.array([2, 3]), spider_space_frame_pos, shell_vertex_centers)
                    rep_val += base_bond_rep(jnp.array([4, 0]), spider_space_frame_pos, shell_vertex_centers)

            return rep_val

        return rep_bond_energy_fn

    def get_energy_fn(
            self,

            # Shell-shell interaction energy parameters
            morse_ii_eps=10.0, morse_ii_alpha=5.0,

            # Shell-attr interaction parameters
            morse_attr_eps=350.0, morse_attr_alpha=2.0, morse_r_onset=12.0, morse_r_cutoff=14.0,

            # Misc. parameters
            soft_eps=10000.0,

    ):

        shell_energy_fn = self.shell.get_energy_fn(
            morse_ii_eps, morse_ii_alpha, soft_eps)

        spider_energy_fn = self.spider.get_energy_fn(
            add_bonds=self.add_spider_bonds,
            opt_leg_springs=self.opt_leg_springs,
            leg_spring_eps=self.leg_spring_eps,
            head_particle_eps=self.head_particle_eps
        )

	## soft sphere repulsion between spider and shell

        # vertex center, vertex patch, spider head, spider attr, spider base
        zero_interaction = jnp.zeros((5, 5))
        spider_pt_species = jnp.array([2, 3, 4])
        soft_sphere_eps = zero_interaction.at[0, spider_pt_species].set(soft_eps)
        soft_sphere_eps = soft_sphere_eps.at[spider_pt_species, 0].set(soft_eps)

        soft_sphere_sigma = zero_interaction.at[0, spider_pt_species].set(self.shell.vertex_radius + self.spider.particle_radii)
        soft_sphere_sigma = soft_sphere_sigma.at[spider_pt_species, 0].set(self.shell.vertex_radius + self.spider.particle_radii)
        sigma = jnp.where(soft_sphere_sigma == 0.0, 1e-5, soft_sphere_sigma) # avoids nans
        pair_energy_soft = energy.soft_sphere_pair(
            self.displacement_fn,
            # species=self.n_point_species
            species=5,
            sigma=sigma, epsilon=soft_sphere_eps)
        soft_energy_fn = rigid_body.point_energy(pair_energy_soft, self.shape, self.shape_species)


        ## attraction between attr and shell vertex
        morse_eps = zero_interaction.at[0, 3].set(morse_attr_eps)
        morse_eps = morse_eps.at[3, 0].set(morse_attr_eps)

        morse_alpha = zero_interaction.at[0, 3].set(morse_attr_alpha)
        morse_alpha = morse_alpha.at[3, 0].set(morse_attr_alpha)

        pair_energy_morse = energy.morse_pair(
            self.displacement_fn,
            # species=self.n_point_species,
            species=5,
            sigma=sigma, epsilon=morse_eps, alpha=morse_alpha,
            r_onset=morse_r_onset, r_cutoff=morse_r_cutoff,
            # per_particle=True,
            per_particle=False,
        )

        self.tagged_shape_species = onp.array([0, 1, 1, 1, 1, 1])
        # morse_energy_fn = rigid_body.point_energy(pair_energy_morse, self.shape, self.shape_species)
        morse_energy_fn = rigid_body.point_energy(pair_energy_morse, self.shape, self.tagged_shape_species)

        rep_bond_energy_fn = self.get_rep_bond_energy_fn(soft_eps, self.bond_radius, self.bond_alpha)

        def pointwise_interaction_energy_fn(body: rigid_body.RigidBody, **kwargs):
            spider_body, shell_body = self.split_body(body)
            bind_body_flat = body[self.vertex_to_bind_idx]
            combined_body_center = jnp.concatenate([bind_body_flat.center.reshape(1, -1),
                                                    spider_body.center])
            combined_body_qvec = jnp.concatenate([bind_body_flat.orientation.vec.reshape(1, -1),
                                                  spider_body.orientation.vec])
            combined_body = rigid_body.RigidBody(combined_body_center, rigid_body.Quaternion(combined_body_qvec))
            pointwise_morse = morse_energy_fn(combined_body, **kwargs)
            # pointwise_morse = 0.0

            # pointwise_morse_vec = morse_energy_fn(body, **kwargs) # Mask out everything but the vertex to bind and the spider attractive sites

            # pointwise_morse = pointwise_morse_vec[self.vertex_to_bind_idx*6]*2 #* self.mask#jnp.dot(pointwise_morse_vec, self.mask)
            # pointwise_morse = 0.0
            pointwise_interaction_energy = soft_energy_fn(body, **kwargs) + pointwise_morse # morse_energy_fn(body, **kwargs)

            return pointwise_interaction_energy


        def interaction_energy_fn(body: rigid_body.RigidBody, **kwargs):

            pointwise_interaction_energy = pointwise_interaction_energy_fn(body, **kwargs)
            bond_interaction_energy = rep_bond_energy_fn(body)

            return pointwise_interaction_energy + bond_interaction_energy


        def energy_fn(body: rigid_body.RigidBody, **kwargs):
            spider_body, shell_body = self.split_body(body)
            shell_energy = shell_energy_fn(shell_body, **kwargs)
            spider_energy = spider_energy_fn(spider_body, **kwargs)
            interaction_energy = interaction_energy_fn(body, **kwargs)
            return shell_energy + spider_energy + interaction_energy

        return energy_fn, pointwise_interaction_energy_fn


    def body_to_injavis_lines(
            self, body, box_size,
            shell_patch_radius=0.5, shell_vertex_color="43a5be", shell_patch_color="4fb06d",
            spider_head_color="ff0000", spider_base_color="1c1c1c"):

        spider_body, shell_body = self.split_body(body)

        _, spider_box_def, spider_type_defs, spider_pos = self.spider.body_to_injavis_lines(
            spider_body, box_size)
        _, shell_box_def, shell_type_defs, shell_pos = self.shell.body_to_injavis_lines(
            shell_body, box_size, shell_patch_radius, vertex_to_bind=self.vertex_to_bind_idx)

        assert(spider_box_def == shell_box_def)
        box_def = spider_box_def
        type_defs = shell_type_defs + spider_type_defs
        positions = shell_pos + spider_pos
        all_lines = [box_def] + type_defs + positions + ["eof"]
        return all_lines, box_def, type_defs, positions


    # The below is for the single vertex to bind and the spider
    def get_extracted_rb_info(
            self,

            # Shell-shell interaction energy parameters
            morse_ii_eps=10.0, morse_ii_alpha=5.0,

            # Shell-attr interaction parameters
            morse_attr_eps=350.0, morse_attr_alpha=2.0, morse_r_onset=12.0, morse_r_cutoff=14.0,

            # Misc. parameters
            soft_eps=10000.0,
    ):
        # Returns both a body and an energy function

        ## Construct the body
        spider_body, shell_body = self.split_body(self.rigid_body)
        vertex_to_bind = shell_body[self.vertex_to_bind_idx]

        combined_center = jnp.concatenate([onp.array([vertex_to_bind.center]), spider_body.center])
        combined_quat_vec = jnp.concatenate([
            onp.array([vertex_to_bind.orientation.vec]),
            spider_body.orientation.vec])
        combined_body = rigid_body.RigidBody(combined_center, rigid_body.Quaternion(combined_quat_vec))
        combined_shape_species = onp.array([0, 1, 1, 1, 1, 1])


        ## Construct the energy function

        ### Get the contribution from the spider
        spider_energy_fn = self.spider.get_energy_fn(
            add_bonds=self.add_spider_bonds,
            opt_leg_springs=self.opt_leg_springs,
            leg_spring_eps=self.leg_spring_eps,
            head_particle_eps=self.head_particle_eps
        )

        ### Note: no contribution from just the vertex

        ### Construct the pointwise interaction energy

        zero_interaction = jnp.zeros((5, 5))
        spider_pt_species = jnp.array([2, 3, 4])
        soft_sphere_eps = zero_interaction.at[0, spider_pt_species].set(soft_eps)
        soft_sphere_eps = soft_sphere_eps.at[spider_pt_species, 0].set(soft_eps)

        soft_sphere_sigma = zero_interaction.at[0, spider_pt_species].set(self.shell.vertex_radius + self.spider.particle_radii)
        soft_sphere_sigma = soft_sphere_sigma.at[spider_pt_species, 0].set(self.shell.vertex_radius + self.spider.particle_radii)
        sigma = jnp.where(soft_sphere_sigma == 0.0, 1e-5, soft_sphere_sigma) # avoids nans
        pair_energy_soft = energy.soft_sphere_pair(
            self.displacement_fn,
            # species=self.n_point_species
            species=5,
            sigma=sigma, epsilon=soft_sphere_eps)
        soft_energy_fn = rigid_body.point_energy(pair_energy_soft, self.shape, combined_shape_species)

        morse_eps = zero_interaction.at[0, 3].set(morse_attr_eps)
        morse_eps = morse_eps.at[3, 0].set(morse_attr_eps)

        morse_alpha = zero_interaction.at[0, 3].set(morse_attr_alpha)
        morse_alpha = morse_alpha.at[3, 0].set(morse_attr_alpha)

        pair_energy_morse = energy.morse_pair(
            self.displacement_fn,
            species=5,
            sigma=sigma, epsilon=morse_eps, alpha=morse_alpha,
            r_onset=morse_r_onset, r_cutoff=morse_r_cutoff,
            per_particle=False,
        )

        morse_energy_fn = rigid_body.point_energy(pair_energy_morse, self.shape, combined_shape_species)

        ### Construct the bond repulsion
        bond_radius = self.bond_radius
        bond_alpha = self.bond_alpha


        def single_leg_rep(l_idx, all_leg_space_frame_pos, vertex_to_bind_space_frame_pos):
            leg_start_idx = 3*l_idx
            leg_end_idx = leg_start_idx+3

            leg_space_frame = all_leg_space_frame_pos[leg_start_idx:leg_end_idx]

            leg_bond_idxs = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
            leg_bond_positions = leg_space_frame[leg_bond_idxs]

            all_dists = utils.mapped_dist_point_to_line(
                leg_bond_positions, jnp.array([vertex_to_bind_space_frame_pos]),
                self.displacement_fn)

            bond_energy_sm = jnp.sum(
                energy.soft_sphere(all_dists,
                                   epsilon=soft_eps,
                                   sigma=bond_radius + self.shell.vertex_radius,
                                   alpha=jnp.array(bond_alpha)))

            return bond_energy_sm

        def base_bond_rep(bond_spider_idxs, all_leg_space_frame_pos, vertex_to_bind_space_frame_pos):
            l_idx1, l_idx2 = bond_spider_idxs

            leg1_base_pos = all_leg_space_frame_pos[3*l_idx1+2]
            leg2_base_pos = all_leg_space_frame_pos[3*l_idx2+2]

            bond_positions = jnp.array([[leg1_base_pos, leg2_base_pos]])

            all_dists = utils.mapped_dist_point_to_line(
                bond_positions, jnp.array([vertex_to_bind_space_frame_pos]), self.displacement_fn)

            bond_energy_sm = jnp.sum(
                energy.soft_sphere(all_dists,
                                   epsilon=soft_eps,
                                   sigma=bond_radius + self.shell.vertex_radius,
                                   alpha=jnp.array(bond_alpha)))

            return bond_energy_sm

        def rep_bond_energy_fn(body):
            spider_body = body[-5:]
            vertex_to_bind = body[0]
            # spider_body, shell_body = self.split_body(body)

            spider_space_frame_pos = vmap(self.spider.legs[0].get_body_frame_positions)(spider_body).reshape(-1, 3)
            # shell_body_pos = self.shell.get_body_frame_positions(shell_body)
            vertex_to_bind_space_pos = self.shell.get_body_frame_positions(
                rigid_body.RigidBody(center=jnp.array([vertex_to_bind.center]),
                                     orientation=rigid_body.Quaternion(vec=jnp.array([vertex_to_bind.orientation.vec]))))
            # vertex_to_bind_center = jnp.array([vertex_to_bind_space_pos[0]])
            vertex_to_bind_center = vertex_to_bind_space_pos[0]

            rep_val = single_leg_rep(0, spider_space_frame_pos, vertex_to_bind_center)
            rep_val += single_leg_rep(1, spider_space_frame_pos, vertex_to_bind_center)
            rep_val += single_leg_rep(2, spider_space_frame_pos, vertex_to_bind_center)
            rep_val += single_leg_rep(3, spider_space_frame_pos, vertex_to_bind_center)
            rep_val += single_leg_rep(4, spider_space_frame_pos, vertex_to_bind_center)
            if self.add_spider_bonds:
                rep_val += base_bond_rep(jnp.array([0, 1]), spider_space_frame_pos, vertex_to_bind_center)
                rep_val += base_bond_rep(jnp.array([1, 2]), spider_space_frame_pos, vertex_to_bind_center)
                rep_val += base_bond_rep(jnp.array([3, 4]), spider_space_frame_pos, vertex_to_bind_center)

                if self.opt_leg_springs:
                    rep_val += base_bond_rep(jnp.array([2, 3]), spider_space_frame_pos, shell_vertex_centers)
                    rep_val += base_bond_rep(jnp.array([4, 0]), spider_space_frame_pos, shell_vertex_centers)
            return rep_val

        def pointwise_interaction_energy_fn(body: rigid_body.RigidBody, **kwargs):
            pointwise_morse = morse_energy_fn(body, **kwargs)
            pointwise_interaction_energy = soft_energy_fn(body, **kwargs) + pointwise_morse
            return pointwise_interaction_energy

        def interaction_energy_fn(body: rigid_body.RigidBody, **kwargs):

            pointwise_interaction_energy = pointwise_interaction_energy_fn(body, **kwargs)
            bond_interaction_energy = rep_bond_energy_fn(body)

            return pointwise_interaction_energy + bond_interaction_energy


        def energy_fn(body: rigid_body.RigidBody, **kwargs):
            spider_body = body[-5:]
            spider_energy = spider_energy_fn(spider_body, **kwargs)
            interaction_energy = interaction_energy_fn(body, **kwargs)
            return spider_energy + interaction_energy

        return combined_body, energy_fn

def combined_body_to_injavis_lines(
        complex_, body, box_size, shell_patch_radius=0.5, shell_vertex_color="43a5be",
        shell_patch_color="4fb06d", spider_head_color="ff0000", spider_base_color="1c1c1c"):

    vertex_body = body[0]
    vertex_body = rigid_body.RigidBody(
        center=jnp.expand_dims(vertex_body.center, 0),
        orientation=rigid_body.Quaternion(jnp.expand_dims(vertex_body.orientation.vec, 0)))
    spider_body = body[1:]

    _, spider_box_def, spider_type_defs, spider_pos = complex_.spider.body_to_injavis_lines(
        spider_body, box_size)
    _, shell_box_def, shell_type_defs, shell_pos = complex_.shell.body_to_injavis_lines(
        vertex_body, box_size, shell_patch_radius, vertex_to_bind=complex_.vertex_to_bind_idx)

    assert(spider_box_def == shell_box_def)
    box_def = spider_box_def
    type_defs = shell_type_defs + spider_type_defs
    positions = shell_pos + spider_pos
    all_lines = [box_def] + type_defs + positions + ["eof"]
    return all_lines, box_def, type_defs, positions


class TestComplex(unittest.TestCase):

    def test_init(self):
        displacement_fn, shift_fn = space.free()
        complex_ = Complex(
            initial_separation_coeff=5.5, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=5.0, spider_head_height=10.0,
            spider_base_particle_radius=0.5, spider_attr_particle_radius=0.5,
            spider_head_particle_radius=0.5,
            spider_point_mass=1.0, spider_mass_err=1e-6
        )


        energy_fn = complex_.get_energy_fn()
        #energy_fn = jit(energy_fn)
        eng = energy_fn(complex_.rigid_body)
        #print('initial energy: ', eng)
        #pdb.set_trace()

        box_size = 30.0
        rb = complex_.rigid_body
        init_injavis_lines = complex_.body_to_injavis_lines(rb, box_size=box_size)[0]
        with open("init.pos", 'w+') as of:
            of.write('\n'.join(init_injavis_lines))

        return

    def test_simulate(self):

        displacement_fn, shift_fn = space.free()
        complex_ = Complex(
            initial_separation_coeff=5.5, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=5.0, spider_head_height=10.0,
            spider_base_particle_radius=0.5, spider_attr_particle_radius=0.5,
            spider_head_particle_radius=0.25,
            spider_point_mass=1.0, spider_mass_err=1e-6,
            head_particle_eps=100000.0
            # head_particle_eps=10000.0
        )

        energy_fn, _ = complex_.get_energy_fn()
        energy_fn = jit(energy_fn)

        dt = 1e-4
        kT = 1.0
        gamma = 10.0
        gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma_rb)
        step_fn = jit(step_fn)
        key = random.PRNGKey(0)
        mass = complex_.shape.mass(complex_.shape_species)
        state = init_fn(key, complex_.rigid_body, mass=mass)

        trajectory = list()
        # n_steps = 25000 # 50000
        n_steps = 2500
        energies = list()
        for _ in tqdm(range(n_steps)):
            state = step_fn(state)
            trajectory.append(state.position)
            energies.append(energy_fn(state.position))

        plt.plot(energies)
        plt.show()
        plt.clf()

        trajectory = utils.tree_stack(trajectory)
        n_vis_freq = 250
        vis_traj_idxs = jnp.arange(0, n_steps+1, n_vis_freq)
        n_vis_states = len(vis_traj_idxs)
        trajectory = trajectory[vis_traj_idxs]

        box_size = 30.0
        traj_injavis_lines = list()
        traj_path = "new_complex.pos"
        for i in tqdm(range(n_vis_states), desc="Generating injavis output"):
            s = trajectory[i]
            traj_injavis_lines += complex_.body_to_injavis_lines(s, box_size=box_size)[0]
        with open(traj_path, 'w+') as of:
            of.write('\n'.join(traj_injavis_lines))


    def test_sim_extracted(self):

        displacement_fn, shift_fn = space.free()


        # ext-rigid-tagged-test-eps3-bigger-radius-start, iteration 350
        """
        sim_params = {
            "log_morse_attr_eps": 4.445757112690842,
            "morse_attr_alpha": 1.228711252063668,
            "morse_r_cutoff": 12.0,
            "morse_r_onset": 10.0,
            "spider_attr_particle_pos_norm": 0.31171913270018414,
            "spider_attr_site_radius": 1.4059036817138681,
            "spider_base_particle_radius": 1.0949878258735661,
            "spider_base_radius": 5.018836622251073,
            "spider_head_height": 9.462070953473482,
            "spider_head_particle_radius": 1.0
        }
        """

        # ext-rigid-tagged-test-eps3-bigger-radius-start-rc0.001,
        sim_params = {
            "log_morse_attr_eps": 4.286391530030283,
            "morse_attr_alpha": 1.4193355362346702,
            "morse_r_cutoff": 12.0,
            "morse_r_onset": 10.0,
            "spider_attr_particle_pos_norm": 0.3632382047051499,
            "spider_attr_site_radius": 1.4752831792315242,
            "spider_base_particle_radius": 1.4979135216810637,
            "spider_base_radius": 4.642459866397608,
            "spider_head_height": 9.355803312442202,
            "spider_head_particle_radius": 1.0
        }

        init_sep_coeff = 5.5
        vertex_to_bind_idx = 5
        spider_leg_radius = 0.25
        min_head_radius = 0.1

        complex_ = Complex(
            initial_separation_coeff=init_sep_coeff,
            vertex_to_bind_idx=vertex_to_bind_idx,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=sim_params['spider_base_radius'],
            spider_head_height=sim_params['spider_head_height'],
            spider_base_particle_radius=sim_params['spider_base_particle_radius'],
            spider_attr_particle_radius=sim_params['spider_attr_site_radius'],
            spider_head_particle_radius=jnp.max(jnp.array([min_head_radius, sim_params['spider_head_particle_radius']])),
            spider_point_mass=1.0, spider_mass_err=1e-6,
            bond_radius=spider_leg_radius,
            rel_attr_particle_pos=jnp.clip(sim_params['spider_attr_particle_pos_norm'], 0.0, 1.0),
            # head_particle_eps=100000.0
            head_particle_eps=10000.0
        )


        combined_body, base_energy_fn = complex_.get_extracted_rb_info(
            morse_attr_eps=jnp.exp(sim_params['log_morse_attr_eps']),
            morse_attr_alpha=sim_params['morse_attr_alpha'],
            morse_r_onset=sim_params['morse_r_onset'],
            morse_r_cutoff=sim_params['morse_r_cutoff'])
        init_energy = base_energy_fn(combined_body)
        base_energy_fn = jit(base_energy_fn)

        op_name = "attr"
        # op_name = "head"
        if op_name == "attr":
            @jit
            def order_param_fn(R):
                leg_rbs = R[-5:] # the spider
                spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
                attr_site_pos = spider_space_frame_pos[1::3]

                vertex_com = R[0].center
                disps = vmap(displacement_fn, (None, 0))(vertex_com, attr_site_pos)
                drs = vmap(space.distance)(disps)
                return jnp.mean(drs)

            def get_new_vertex_com(R, dist):
                leg_rbs = R[-5:] # the spider
                spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
                attr_site_pos = spider_space_frame_pos[1::3]
                avg_attr_site_pos = jnp.mean(attr_site_pos, axis=0)

                a = space.distance(displacement_fn(avg_attr_site_pos, attr_site_pos[0]))
                b = onp.sqrt(dist**2 - a**2) # pythag

                vertex_com = R[0].center
                avg_attr_site_to_vertex = displacement_fn(avg_attr_site_pos, vertex_com)
                dir_ = avg_attr_site_to_vertex / jnp.linalg.norm(avg_attr_site_to_vertex)
                new_vertex_pos = avg_attr_site_pos - dir_*b
                return new_vertex_pos
        elif op_name == "head":
            @jit
            def order_param_fn(R):
                leg_rbs = R[-5:] # the spider
                spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
                head_site_pos = spider_space_frame_pos[0::3]

                vertex_com = R[0].center
                disps = vmap(displacement_fn, (None, 0))(vertex_com, head_site_pos)
                drs = vmap(space.distance)(disps)
                return jnp.mean(drs)

            def get_new_vertex_com(R, dist):
                leg_rbs = R[-5:] # the spider
                spider_space_frame_pos = vmap(complex_.spider.legs[0].get_body_frame_positions)(leg_rbs).reshape(-1, 3)
                head_site_pos = spider_space_frame_pos[0::3]
                avg_head_site_pos = jnp.mean(head_site_pos, axis=0)

                # Note: assumes all head site positions are about the same
                vertex_com = R[0].center
                head_site_to_vertex = displacement_fn(avg_head_site_pos, vertex_com)
                dir_ = head_site_to_vertex / jnp.linalg.norm(head_site_to_vertex)
                new_vertex_pos = avg_head_site_pos - dir_*dist
                return new_vertex_pos
        else:
            raise RuntimeError(f"Invalid op_name: {op_name}")

        def get_init_body(R, dist):
            new_vertex_pos = get_new_vertex_com(R, dist)
            new_center = R.center.at[0].set(new_vertex_pos)
            return rigid_body.RigidBody(new_center, R.orientation)

        # k_bias = 500000
        # k_bias = 0.0
        k_bias = 10000.
        # k_bias = 50000
        # target_op = 5.0
        target_op = 4.0
        init_body = get_init_body(combined_body, target_op)
        box_size = 30.0
        init_body_injavis_lines = combined_body_to_injavis_lines(complex_, init_body, box_size=box_size)[0]
        with open("init_body.pos", 'w+') as of:
            of.write('\n'.join(init_body_injavis_lines))

        def _harmonic_bias(op):
            return 1/2*k_bias * (target_op - op)**2

        def harmonic_bias(R):
            op = order_param_fn(R)
            return _harmonic_bias(op)

        def energy_fn(R):
            bias_val = harmonic_bias(R)
            base_val = base_energy_fn(R)
            return bias_val + base_val
        energy_fn = jit(energy_fn)


        # dt = 1e-4
        dt = 1e-3
        # dt = 1e-5
        # dt = 1e-6
        kT = 1.0
        gamma = 10.0
        gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma_rb)
        step_fn = jit(step_fn)
        key = random.PRNGKey(0)
        mass = complex_.shape.mass(onp.array([0, 1, 1, 1, 1, 1]))
        state = init_fn(key, init_body, mass=mass)


        n_steps = 10000
        sample_every = 250
        trajectory = list()
        ops = list()
        energies = list()
        for i in tqdm(range(n_steps)):
            state = step_fn(state)
            if i % sample_every == 0:
                trajectory.append(state.position)
                energies.append(energy_fn(state.position))
                ops.append(order_param_fn(state.position))

        plt.plot(ops)
        plt.axhline(y=target_op, linestyle="--", color="red")
        plt.show()
        plt.close()

        plt.plot(energies)
        plt.show()
        plt.close()

        traj_injavis_lines = list()
        n_vis_states = len(trajectory)

        for i in tqdm(range(n_vis_states), desc="Generating injavis output"):
            s = trajectory[i]
            traj_injavis_lines += combined_body_to_injavis_lines(complex_, s, box_size=box_size)[0]

        with open("test_combined_sim.pos", 'w+') as of:
            of.write('\n'.join(traj_injavis_lines))







if __name__ == "__main__":
    unittest.main()
