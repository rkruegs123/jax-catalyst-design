import pdb
import unittest

from jax import vmap, random
import jax.numpy as jnp
from jax_md import space


from jax.config import config
config.update('jax_enable_x64', True)


def get_loss_fn(
        displacement_fn, vertex_to_bind,
        use_abduction=True,
        use_remaining_shell_vertices_loss=False, remaining_shell_vertices_loss_coeff=1.0
):

    # assert(use_abduction or use_remaining_shell_vertices_loss)
    if (not use_abduction) and (not use_remaining_shell_vertices_loss):
        loss_fn = lambda body, params, complex_: 0.0
        loss_terms_fn = lambda body, params, complex_: (0.0, 0.0)
        return loss_fn, loss_terms_fn

    d = vmap(displacement_fn, (0, None))
    def abduction_loss(body):
        shell_body = body[:12]
        disps = d(shell_body.center, body[vertex_to_bind].center)
        dists = space.distance(disps)
        vertex_far_from_icos = -jnp.sum(dists)
        return vertex_far_from_icos


    def remaining_shell_vertices_loss(body, params, complex_):
        head_remaining_shell_energy_fn = complex_.get_remaining_shell_morse_energy_fn(
            morse_attr_eps=jnp.exp(params['log_morse_attr_eps']),
            morse_attr_alpha=params['morse_attr_alpha'],
            morse_r_onset=params["morse_r_onset"],
            morse_r_cutoff=params["morse_r_cutoff"])
        return head_remaining_shell_energy_fn(body)**2 * remaining_shell_vertices_loss_coeff


    use_abduction_bit = int(use_abduction)
    use_remaining_shell_vertices_bit = int(use_remaining_shell_vertices_loss)

    def loss_terms_fn(body, params, complex_):
        abduction_term = abduction_loss(body)*use_abduction_bit
        remaining_energy_term = remaining_shell_vertices_loss(body, params, complex_)*use_remaining_shell_vertices_bit
        return abduction_term, remaining_energy_term

    def loss_fn(body, params, complex_info):
        t1, t2 = loss_terms_fn(body, params, complex_info)
        return t1 + t2

    return loss_fn, loss_terms_fn
