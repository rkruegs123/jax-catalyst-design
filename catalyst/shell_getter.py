import pdb
from pathlib import Path
import unittest
from tqdm import tqdm

from jax import vmap
import jax.numpy as jnp
from jax_md import rigid_body


class ShellInfo:
    def __init__(self, obj_dir="obj/"):
        self.obj_dir = Path(obj_dir)
        self.vertex_radius = 2.0

        self.load() # populate self.rigid_body and self.vertex_shape

    def run_minimization(self):
        raise NotImplementedError

    def load_from_file(self):
        raise NotImplementedError

    def load(self):
        icosahedron_rb_center_path = self.obj_dir / "icosahedron_rb_center.npy"
        icosahedron_rb_orientation_vec_path = self.obj_dir / "icosahedron_rb_orientation_vec.npy"
        vertex_shape_points_path = self.obj_dir / "vertex_shape_points.npy"
        vertex_shape_masses_path = self.obj_dir / "vertex_shape_masses.npy"
        vertex_shape_point_count_path = self.obj_dir / "vertex_shape_point_count.npy"
        vertex_shape_point_offset_path = self.obj_dir / "vertex_shape_point_offset.npy"
        vertex_shape_point_species_path = self.obj_dir / "vertex_shape_point_species.npy"
        vertex_shape_point_radius_path = self.obj_dir / "vertex_shape_point_radius.npy"

        icosahedron_paths_exist = icosahedron_rb_center_path.exists() \
                                  and icosahedron_rb_orientation_vec_path.exists()
        vertex_shape_paths_exist = vertex_shape_points_path.exists() \
                                   and vertex_shape_masses_path.exists() \
                                   and vertex_shape_point_count_path.exists() \
                                   and vertex_shape_point_offset_path.exists() \
                                   and vertex_shape_point_species_path.exists() \
                                   and vertex_shape_point_radius_path.exists()

        if icosahedron_paths_exist and vertex_shape_paths_exist:
            print(f"Loading minimized icosahedron rigid body and vertex shape from data directory: {self.obj_dir}")
            icosahedron_rigid_body_center = jnp.load(icosahedron_rb_center_path)
            icosahedron_rigid_body_orientation_vec = jnp.load(icosahedron_rb_orientation_vec_path)
            icosahedron_rigid_body = rigid_body.RigidBody(
                center=icosahedron_rigid_body_center,
                orientation=rigid_body.Quaternion(vec=icosahedron_rigid_body_orientation_vec))

            vertex_shape_points = jnp.load(vertex_shape_points_path)
            vertex_shape_masses = jnp.load(vertex_shape_masses_path)
            vertex_shape_point_count = jnp.load(vertex_shape_point_count_path)
            vertex_shape_point_offset = jnp.load(vertex_shape_point_offset_path)
            vertex_shape_point_species = jnp.load(vertex_shape_point_species_path)
            vertex_shape_point_radius = jnp.load(vertex_shape_point_radius_path)

            vertex_shape = rigid_body.RigidPointUnion(
                points=vertex_shape_points,
                masses=vertex_shape_masses,
                point_count=vertex_shape_point_count,
                point_offset=vertex_shape_point_offset,
                point_species=vertex_shape_point_species,
                point_radius=vertex_shape_point_radius
            )

            self.rigid_body = icosahedron_rigid_body
            self.vertex_shape = vertex_shape
            return
        self.run_minimization()


    def get_body_frame_positions(self):
        body_pos = vmap(rigid_body.transform, (0, None))(self.rigid_body, self.vertex_shape) # 12x6x3
        return body_pos


class TestShellInfo(unittest.TestCase):
    def test_final_configuration(self):
        shell_info = ShellInfo()
        body_pos = shell_info.get_body_frame_positions()

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


if __name__ == "__main__":
    unittest.main()
