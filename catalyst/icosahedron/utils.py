from tqdm import tqdm

from jax import vmap, tree_util
import jax.numpy as jnp

from jax_md import space
# from jax_md import rigid_body

import catalyst.icosahedron.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)

def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)

def get_body_frame_positions(rb, shape):
    body_pos = vmap(rigid_body.transform, (0, None))(rb, shape)
    return body_pos

def traj_to_pos_file(traj, complex_info, traj_path, box_size=30.0):
    assert(len(traj.center.shape) == 3)
    n_states = traj.center.shape[0]

    traj_injavis_lines = list()
    for i in tqdm(range(n_states), desc="Generating injavis output"):
        s = traj[i]
        traj_injavis_lines += complex_info.body_to_injavis_lines(s, box_size=box_size)[0]
    with open(traj_path, 'w+') as of:
        of.write('\n'.join(traj_injavis_lines))


# https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def dist_point_to_line_segment(line_points, point, displacement_fn):
    line_p1 = jnp.squeeze(line_points[0])
    line_p2 = jnp.squeeze(line_points[1])

    disp_line = displacement_fn(line_p2, line_p1)
    norm = space.distance(disp_line)
    u = jnp.dot(displacement_fn(point, line_p1), disp_line) / norm**2
    u = jnp.where(u > 1, 1, u)
    u = jnp.where(u < 0, 0, u)
    pt = line_p1 + u * disp_line
    d_pt = displacement_fn(pt, point)

    return space.distance(d_pt)
mapped_dist_point_to_line = vmap(vmap(dist_point_to_line_segment, (0, None, None)), (None, 0, None))
