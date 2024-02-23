from tqdm import tqdm

from jax import vmap, tree_util
import jax.numpy as jnp

from jax_md import space
# from jax_md import rigid_body

import catalyst.icosahedron_tagged.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)


def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)


def get_body_frame_positions(rb, shape):
    body_pos = vmap(rigid_body.transform, (0, None))(rb, shape)
    return body_pos


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
