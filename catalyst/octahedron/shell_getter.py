import pdb
from pathlib import Path
import unittest
from tqdm import tqdm

from jax import vmap
import jax.numpy as jnp
from jax_md import energy, space

from catalyst.icosahedron import utils
import catalyst.icosahedron.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)


class ShellInfo:
    def __init__(self, displacement_fn, obj_basedir="obj/", verbose=True):
        self.displacement_fn = displacement_fn
        self.obj_dir = Path(obj_basedir) / "octahedron"
        assert(self.obj_dir.exists())
        self.set_path_names()
        self.vertex_radius = 2.0

        self.verbose = verbose

        self.load() # populate self.rigid_body and self.vertex_shape


    def set_path_names(self):
        self.rb_center_path = self.obj_dir / "rb_center.npy"
        self.rb_orientation_vec_path = self.obj_dir / "rb_orientation_vec.npy"
        self.vertex_shape_points_path = self.obj_dir / "vertex_shape_points.npy"
        self.vertex_shape_masses_path = self.obj_dir / "vertex_shape_masses.npy"
        self.vertex_shape_point_count_path = self.obj_dir / "vertex_shape_point_count.npy"
        self.vertex_shape_point_offset_path = self.obj_dir / "vertex_shape_point_offset.npy"
        self.vertex_shape_point_species_path = self.obj_dir / "vertex_shape_point_species.npy"
        self.vertex_shape_point_radius_path = self.obj_dir / "vertex_shape_point_radius.npy"

    def load(self):
        rb_paths_exist = self.rb_center_path.exists() \
                                  and self.rb_orientation_vec_path.exists()
        vertex_shape_paths_exist = self.vertex_shape_points_path.exists() \
                                   and self.vertex_shape_masses_path.exists() \
                                   and self.vertex_shape_point_count_path.exists() \
                                   and self.vertex_shape_point_offset_path.exists() \
                                   and self.vertex_shape_point_species_path.exists() \
                                   and self.vertex_shape_point_radius_path.exists()


        if rb_paths_exist and vertex_shape_paths_exist:
            self.load_from_file()
        else:
            self.run_minimization()

        # raise NotImplementedError

    def get_vertex_shape(self, vertex_coords):
        # Get the vertex shape (i.e. the coordinates of a vertex for defining a rigid body)

        anchor = vertex_coords[0]
        d = vmap(self.displacement_fn, (0, None))

        # Compute all pairwise distances
        dists = space.distance(d(vertex_coords, anchor))

        # Mask the diagonal
        self_distance_tolerance = 1e-5
        large_mask_distance = 100.0
        dists = jnp.where(dists < self_distance_tolerance, large_mask_distance, dists) # mask the diagonal

        # Find nearest neighbors
        # note: we use min because the distances to the nearest neighbors are all the same (they should be 1 diameter away)
        # note: this step is not differentiable, but that's fine: we keep the octahedron fixed for the optimization
        nbr_ids = jnp.where(dists == jnp.min(dists))[0]
        nbr_coords = vertex_coords[nbr_ids]

        # Compute displacements to neighbors to determine patch positions
        vec = d(nbr_coords, anchor)
        norm = jnp.linalg.norm(vec, axis=1).reshape(-1, 1)
        vec /= norm
        patch_pos = anchor - self.vertex_radius * vec

        # Collect shape in an array and return
        shape_coords = jnp.concatenate([jnp.array([anchor]), patch_pos]) - anchor
        return shape_coords

    def get_unminimized_shell(self, vertex_mass=1.0, patch_mass=1e-8):

        d = vmap(self.displacement_fn, (0, None))

        # Compute the coordinates of the vertices (i.e. no patches)
        vertex_coords = 2.0 / jnp.sqrt(2.0) * self.vertex_radius \
                        * jnp.array([[1.0, 0.0, 0.0],
                                     [-1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, -1.0, 0.0],
                                     [0.0, 0.0, 1.0],
                                     [0.0, 0.0, -1.0]])

        # Compute the vertex shape positions
        # note: first position is vertex, rest are patches
        vertex_rb_positions = self.get_vertex_shape(vertex_coords)
        num_patches = vertex_rb_positions.shape[0] - 1 #don't count vertex particle
        assert(num_patches == 4)

        # Set the species
        species = jnp.zeros(num_patches + 1, dtype=jnp.int32)
        species = species.at[1:].set(1) # first particle is the vertex, rest are patches

        # Get the masses
        # note: patch mass should be zero, but zeros cause nans in the gradient
        patch_mass = jnp.ones(num_patches)*patch_mass
        mass = jnp.concatenate((jnp.array([vertex_mass]), patch_mass), axis=0)

        # Set the shape
        vertex_shape = rigid_body.point_union_shape(vertex_rb_positions, mass).set(
            point_species=species)
        self.shape = vertex_shape


        """
        Now we orient rigid body particles (vertex + patches) within the rigid body.
        We don't orient the rotation about the z axis (where the z axis points
        toward the center of the icosahedron). We correct this by running a short
        simulation with just the icosahedron.

        We reference this stack overflow link to handle reorientation with
        quaternions:
        https://math.stackexchange.com/questions/60511/quaternion-for-an-object-that-to-point-in-a-direction
        """

        # Get vectors that point towards the center
        central_point = jnp.mean(vertex_coords, axis=0) # center of the icosahedron
        reoriented_vectors = d(vertex_coords, central_point)
        norm = jnp.linalg.norm(reoriented_vectors, axis=1).reshape(-1, 1)
        reoriented_vectors /= norm

        # Now we have to compute a quaternion such that we rotate the current directions towards the center (i.e. the reoriented vectors)
        orig_vec = self.displacement_fn(vertex_shape.points[0], jnp.mean(vertex_shape.points[1:], axis=0))
        orig_vec /= jnp.linalg.norm(orig_vec)
        crossed = vmap(jnp.cross, (None, 0))(orig_vec, reoriented_vectors)
        dotted = vmap(jnp.dot, (0, None))(reoriented_vectors, orig_vec)

        theta = jnp.arccos(dotted)
        cos_part = jnp.cos(theta / 2).reshape(-1, 1)
        mult = vmap(lambda v, s: s*v, (0, 0))
        sin_part = mult(crossed, jnp.sin(theta/2))
        orientation = jnp.concatenate([cos_part, sin_part], axis=1)
        norm = jnp.linalg.norm(orientation, axis=1).reshape(-1, 1)
        orientation /= norm
        orientation = rigid_body.Quaternion(orientation)
        return rigid_body.RigidBody(vertex_coords, orientation)


    def run_minimization(self, vertex_mass=1.0, patch_mass=1e-8):
        unmimized_rb = self.get_unminimized_shell(vertex_mass=1.0, patch_mass=1e-8) # FIXME
        return
        # raise NotImplementedError

    def load_from_file(self):
        if self.verbose:
            print(f"Loading minimized octahedron rigid body and vertex shape from data directory: {self.obj_dir}")
        rigid_body_center = jnp.load(self.rb_center_path).astype(jnp.float64)
        rigid_body_orientation_vec = jnp.load(self.rb_orientation_vec_path).astype(jnp.float64)
        octahedron_rigid_body = rigid_body.RigidBody(
            center=rigid_body_center,
            orientation=rigid_body.Quaternion(vec=rigid_body_orientation_vec))

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

        self.rigid_body = octahedron_rigid_body
        self.shape = vertex_shape
        self.shape_species = None

        return


    def get_body_frame_positions(self, body):
        return utils.get_body_frame_positions(body, self.shape).reshape(-1, 3)

    # note: body is only a single state, not a trajectory
    def body_to_injavis_lines(
            self, body, box_size,
            patch_radius=0.5,
            vertex_color="43a5be", patch_color="4fb06d"):

        assert(len(body.center.shape) == 2)
        body_pos = self.get_body_frame_positions(body)

        assert(len(body_pos.shape) == 2)
        assert(body_pos.shape[0] % 5 == 0)
        n_vertices = body_pos.shape[0] // 5
        if n_vertices != 6:
            print(f"WARNING: writing shell body with only {n_vertices} vertices")

        assert(body_pos.shape[1] == 3)

        box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
        vertex_def = f"def V \"sphere {self.vertex_radius*2} {vertex_color}\""
        patch_def = f"def P \"sphere {patch_radius*2} {patch_color}\""

        position_lines = list()
        for num_vertex in range(n_vertices):
            vertex_start_idx = num_vertex*5

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

    def test_load(self):
        displacement_fn, shift_fn = space.free()
        shell_info = ShellInfo(displacement_fn)

    def test_write_unminimized_injavis(self):
        displacement_fn, shift_fn = space.free()
        shell_info = ShellInfo(displacement_fn)
        unminimized_shell_rb = shell_info.get_unminimized_shell(vertex_mass=1.0, patch_mass=1e-8)
        injavis_lines, _, _, _ = shell_info.body_to_injavis_lines(unminimized_shell_rb, box_size=15.0)
        with open('unminimized_octahedron.pos', 'w+') as of:
            of.write('\n'.join(injavis_lines))


if __name__ == "__main__":
    unittest.main()
