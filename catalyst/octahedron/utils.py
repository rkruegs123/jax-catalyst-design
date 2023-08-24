from tqdm import tqdm

from jax import vmap
import jax.numpy as jnp

from jax_md import space
from jax_md import rigid_body

# import catalyst.octahedron.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)


def get_body_frame_positions(rb, shape):
    body_pos = vmap(rigid_body.transform, (0, None))(rb, shape)
    return body_pos
