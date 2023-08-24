import pdb
from pathlib import Path
import unittest
from tqdm import tqdm

from jax import vmap
import jax.numpy as jnp
# from jax_md import rigid_body, energy, space
from jax_md import energy, space

from catalyst.icosahedron import utils
import catalyst.icosahedron.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)



class ShellInfo:
    def __init__(self, displacement_fn, obj_basedir="obj/", verbose=True):
        self.displacement_fn = displacement_fn
        self.obj_dir = Path(obj_basedir) / "icosahedron"
        assert(obj_dir.exists())
        self.set_path_names()
        self.vertex_radius = 2.0

        self.verbose = verbose

        self.load() # populate self.rigid_body and self.vertex_shape

    def run_minimization(self):
        raise NotImplementedError

    def set_path_names(self):
        self.icosahedron_rb_center_path = self.obj_dir / "icosahedron_rb_center.npy"
        self.icosahedron_rb_orientation_vec_path = self.obj_dir / "icosahedron_rb_orientation_vec.npy"
        self.vertex_shape_points_path = self.obj_dir / "vertex_shape_points.npy"
        self.vertex_shape_masses_path = self.obj_dir / "vertex_shape_masses.npy"
        self.vertex_shape_point_count_path = self.obj_dir / "vertex_shape_point_count.npy"
        self.vertex_shape_point_offset_path = self.obj_dir / "vertex_shape_point_offset.npy"
        self.vertex_shape_point_species_path = self.obj_dir / "vertex_shape_point_species.npy"
        self.vertex_shape_point_radius_path = self.obj_dir / "vertex_shape_point_radius.npy"

    def load_from_file(self):
        if self.verbose:
            print(f"Loading minimized icosahedron rigid body and vertex shape from data directory: {self.obj_dir}")
        icosahedron_rigid_body_center = jnp.load(self.icosahedron_rb_center_path).astype(jnp.float64)
        icosahedron_rigid_body_orientation_vec = jnp.load(self.icosahedron_rb_orientation_vec_path).astype(jnp.float64)
        icosahedron_rigid_body = rigid_body.RigidBody(
            center=icosahedron_rigid_body_center,
            orientation=rigid_body.Quaternion(vec=icosahedron_rigid_body_orientation_vec))

        vertex_shape_points = jnp.load(self.vertex_shape_points_path)
        vertex_shape_masses = jnp.load(self.vertex_shape_masses_path)
        vertex_shape_point_count = jnp.load(self.vertex_shape_point_count_path)
        vertex_shape_point_offset = jnp.load(self.vertex_shape_point_offset_path)
        vertex_shape_point_species = jnp.load(self.vertex_shape_point_species_path)
        vertex_shape_point_radius = jnp.load(self.vertex_shape_point_radius_path)

        vertex_shape = rigid_body.RigidPointUnion(
            points=vertex_shape_points,
            masses=vertex_shape_masses,
            point_count=vertex_shape_point_count,
            point_offset=vertex_shape_point_offset,
            point_species=vertex_shape_point_species,
            point_radius=vertex_shape_point_radius
        )

        self.rigid_body = icosahedron_rigid_body
        self.shape = vertex_shape
        self.shape_species = None
        return

    def load(self):

        icosahedron_paths_exist = self.icosahedron_rb_center_path.exists() \
                                  and self.icosahedron_rb_orientation_vec_path.exists()
        vertex_shape_paths_exist = self.vertex_shape_points_path.exists() \
                                   and self.vertex_shape_masses_path.exists() \
                                   and self.vertex_shape_point_count_path.exists() \
                                   and self.vertex_shape_point_offset_path.exists() \
                                   and self.vertex_shape_point_species_path.exists() \
                                   and self.vertex_shape_point_radius_path.exists()

        if icosahedron_paths_exist and vertex_shape_paths_exist:
            self.load_from_file()
        else:
            self.run_minimization()


    def get_body_frame_positions(self, body):
        return utils.get_body_frame_positions(body, self.shape).reshape(-1, 3)

    def get_energy_fn(self, morse_ii_eps=10.0, morse_ii_alpha=5.0, soft_eps=10000.0,
                      morse_r_onset=10.0, morse_r_cutoff=12.0
    ):

        n_point_species = 2 # hardcoded for clarity

        zero_interaction = jnp.zeros((n_point_species, n_point_species))

        # icosahedral patches attract eachother
        morse_eps = zero_interaction.at[1, 1].set(morse_ii_eps)
        morse_alpha = zero_interaction.at[1, 1].set(morse_ii_alpha)

        # icosahedral centers repel each other
        soft_sphere_eps = zero_interaction.at[0, 0].set(soft_eps)

        soft_sphere_sigma = zero_interaction.at[0, 0].set(self.vertex_radius*2)
        soft_sphere_sigma = jnp.where(soft_sphere_sigma == 0.0, 1e-5, soft_sphere_sigma) # avoids nans

        pair_energy_soft = energy.soft_sphere_pair(
            self.displacement_fn, species=n_point_species,
            sigma=soft_sphere_sigma, epsilon=soft_sphere_eps)
        pair_energy_morse = energy.morse_pair(
            self.displacement_fn, species=n_point_species,
            sigma=0.0, epsilon=morse_eps, alpha=morse_alpha,
            r_onset=morse_r_onset, r_cutoff=morse_r_cutoff)
        pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)

        # will accept body where body.center has dimensions (12, 3)
        # and body.orientation.vec has dimensions (12, 4)
        energy_fn = rigid_body.point_energy(
            pair_energy_fn,
            self.shape,
            # jnp.zeros(12) # FIXME: check if we need this
        )

        return energy_fn

    # note: body is only a single state, not a trajectory
    def body_to_injavis_lines(
            self, body, box_size,
            patch_radius=0.5,
            vertex_color="43a5be", patch_color="4fb06d"):

        assert(len(body.center.shape) == 2)
        body_pos = self.get_body_frame_positions(body)

        assert(len(body_pos.shape) == 2)
        assert(body_pos.shape[0] % 6 == 0)
        n_vertices = body_pos.shape[0] // 6
        if n_vertices != 12:
            print(f"WARNING: writing shell body with only {n_vertices} vertices")

        # assert(body_pos.shape[0] == 6 * 12)
        assert(body_pos.shape[1] == 3)

        box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
        vertex_def = f"def V \"sphere {self.vertex_radius*2} {vertex_color}\""
        patch_def = f"def P \"sphere {patch_radius*2} {patch_color}\""

        position_lines = list()
        for num_vertex in range(n_vertices):
            vertex_start_idx = num_vertex*6

            # vertex center
            vertex_center_pos = body_pos[vertex_start_idx]
            vertex_line = f"V {vertex_center_pos[0]} {vertex_center_pos[1]} {vertex_center_pos[2]}"
            position_lines.append(vertex_line)

            for num_patch in range(5):
                patch_pos = body_pos[vertex_start_idx+num_patch+1]
                patch_line = f"P {patch_pos[0]} {patch_pos[1]} {patch_pos[2]}"
                position_lines.append(patch_line)

        # Return: all lines, box info, particle types, positions
        all_lines = [box_def, vertex_def, patch_def] + position_lines + ["eof"]
        return all_lines, box_def, [vertex_def, patch_def], position_lines


class TestShellInfo(unittest.TestCase):
    displacement_fn, shift_fn = space.free()

    def test_final_configuration(self):
        shell_info = ShellInfo(self.displacement_fn)
        body_pos = shell_info.get_body_frame_positions(shell_info.rigid_body)

        num_vertices = 12
        self.assertEqual(num_vertices, body_pos.shape[0])
        self.assertEqual(body_pos.shape[1], 6)
        self.assertEqual(body_pos.shape[2], 3)
        self.assertEqual(len(body_pos.shape), 3)

        tol = 0.1

        for v in tqdm(range(num_vertices)):
            v_patch_positions = body_pos[v][1:]
            remaining_patch_positions = jnp.concatenate([body_pos[:v, 1:], body_pos[v+1:, 1:]])
            remaining_patch_positions = remaining_patch_positions.reshape(-1, 3)

            for v_patch_idx in range(5):
                v_patch_pos = v_patch_positions[v_patch_idx]
                distances = jnp.linalg.norm(remaining_patch_positions - v_patch_pos, axis=1)
                within_tol = jnp.where(distances < tol, 1, 0)
                self.assertEqual(within_tol.sum(), 1)

    def test_energy_fn_no_errors(self):
        shell_info = ShellInfo(self.displacement_fn)
        energy_fn = shell_info.get_energy_fn()
        init_energy = energy_fn(shell_info.rigid_body)
        print(f"Initial energy: {init_energy}")




if __name__ == "__main__":
    unittest.main()
