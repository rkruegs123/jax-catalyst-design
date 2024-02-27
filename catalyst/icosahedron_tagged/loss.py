import pdb
import unittest

from jax import vmap, random
import jax.numpy as jnp
from jax_md import space


from jax.config import config
config.update('jax_enable_x64', True)


def get_loss_fn(displacement_fn, vertex_to_bind):

    d = vmap(displacement_fn, (0, None))
    def abduction_loss(body):
        shell_body = body[:-1]
        disps = d(shell_body.center, body[vertex_to_bind].center)
        dists = space.distance(disps)
        vertex_far_from_icos = -jnp.sum(dists)
        return vertex_far_from_icos

    def release_term(body, params, complex_):
        init_rb = complex_.rigid_body
        init_spider_rb, init_shell_rb = self.split_body(body)

        init_vtx_to_bind = init_shell_rb[complex_.vertex_to_bind_idx]
        init_leg0 = init_spider_rb[0]
        init_spider_head = init_leg0.get_body_frame_positions(init_spider_rb[0])[0]

        disp_vector = displacement_fn(init_vtx_to_bind, init_spider_head)
        disp_vector_norm = disp_vector / space.distance(disp_vector)

        dist_to_attr_site = init_leg0.leg_length * init_leg0.attr_particle_pos_norm
        

    def loss_terms_fn(body, params, complex_info):
        abduction_term = abduction_loss(body)
        return abduction_term

    def loss_fn(body, params, complex_info):
        t1 = loss_terms_fn(body, params, complex_info)
        return t1

    return loss_fn, loss_terms_fn


