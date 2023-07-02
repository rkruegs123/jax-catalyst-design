from jax import vmap

from jax_md import rigid_body


def get_body_frame_positions(rb, shape):
    body_pos = vmap(rigid_body.transform, (0, None))(rb, shape)
    return body_pos
