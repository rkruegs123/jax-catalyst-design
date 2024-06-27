import pdb
import unittest

from jax import vmap, random
import jax.numpy as jnp

from jax_md import space

from catalyst.icosahedron_ext_rigid_tagged.complex import Complex, PENTAPOD_LEGS, BASE_LEGS

from jax.config import config
config.update('jax_enable_x64', True)


def get_loss_fn(displacement_fn, vertex_to_bind
):


    d = vmap(displacement_fn, (0, None))

    def abduction_loss(body):
        shell_body = body[:-1]
        disps = d(shell_body.center, body[vertex_to_bind].center)
        dists = space.distance(disps)
        vertex_far_from_icos = -jnp.sum(dists)
        return vertex_far_from_icos

    def loss_terms_fn(body, params, complex_info):
        abduction_term = abduction_loss(body)
        return abduction_term

    def loss_fn(body, params, complex_info):
        t1 = loss_terms_fn(body, params, complex_info)
        return t1

    return loss_fn, loss_terms_fn
