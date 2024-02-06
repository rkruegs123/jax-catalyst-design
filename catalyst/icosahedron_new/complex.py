import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import numpy as onp
import matplotlib.pyplot as plt

from jax import vmap, lax, jit, random
import jax.numpy as jnp
from jax_md import energy, space, rigid_body, simulate

from catalyst.icosahedron_new.spider import Spider
from catalyst.icosahedron_new.shell import Shell
from catalyst.icosahedron_new import utils

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

class Complex:
    def __init__(self,
                 # complex-specific arguments
                 initial_separation_coeff, vertex_to_bind_idx,
                 displacement_fn, shift_fn,

                 # spider-specific arguments arguments
                 spider_base_radius, spider_head_height,
                 spider_base_particle_radius, spider_attr_particle_radius, spider_head_particle_radius,
                 spider_point_mass, spider_mass_err=1e-6,

                 # misc.
                 verbose=True,

                 # legs
                 spider_bond_idxs=None, spider_leg_radius=0.5

    ):
        self.n_legs = 5

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

        self.spider_bond_idxs = spider_bond_idxs
        self.spider_leg_radius = spider_leg_radius

        self.load()

    def split_body(self, body):
        spider_body = body[-5:]
        shell_body = body[:12]
        return spider_body, shell_body

    def load(self):
        self.shell = Shell(self.displacement_fn, self.shift_fn,
                           verbose=self.verbose) # note: won't change

        vertex_to_bind = self.shell.rigid_body[self.vertex_to_bind_idx]

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
            0.5, self.spider_attr_particle_radius,
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

    def get_energy_fn(
            self,

            # Shell-shell interaction energy parameters
            morse_ii_eps=10.0, morse_ii_alpha=5.0,

            # Misc. parameters
            soft_eps=10000.0,

    ):

        shell_energy_fn = self.shell.get_energy_fn(
            morse_ii_eps, morse_ii_alpha, soft_eps)

        spider_energy_fn = self.spider.get_energy_fn()


        def energy_fn(body: rigid_body.RigidBody, **kwargs):
            spider_body, shell_body = self.split_body(body)
            shell_energy = shell_energy_fn(shell_body, **kwargs)
            spider_energy = spider_energy_fn(spider_body, **kwargs)
            return shell_energy + spider_energy

        # FIXME: have to add interaction energy. Could add a repulsive interaction between the head and the vertices that has a cutoff slightly lesss than the head height to mitigate off target of a single attractive site wanting to bind the vertex itself. Also, weak attraction between base sites and vertices. Of course attraction between attractive sites and vertices. Maybe or maybe not weak repulsion between attractive sites.

        return energy_fn


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
            spider_point_mass=1.0, spider_mass_err=1e-6
        )

        energy_fn = complex_.get_energy_fn()
        energy_fn = jit(energy_fn)

        dt = 1e-3
        kT = 1.0
        gamma = 10.0
        gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma_rb)
        step_fn = jit(step_fn)
        key = random.PRNGKey(0)
        mass = complex_.shape.mass(complex_.shape_species)
        state = init_fn(key, complex_.rigid_body, mass=mass)

        trajectory = list()
        n_steps = 50000
        energies = list()
        for _ in tqdm(range(n_steps)):
            state = step_fn(state)
            trajectory.append(state.position)
            energies.append(energy_fn(state.position))

        plt.plot(energies)
        plt.show()
        plt.clf()

        trajectory = utils.tree_stack(trajectory)
        n_vis_freq = 1000
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


if __name__ == "__main__":
    unittest.main()
