import pdb
import unittest

from jax import vmap, random
import jax.numpy as jnp

from jax_md import space

from catalyst.dodecahedron.complex_getter import ComplexInfo

from jax.config import config
config.update('jax_enable_x64', True)


def get_loss_fn(
        displacement_fn, vertex_to_bind, use_abduction=True,
        use_stable_shell=False, min_com_dist=2.00, max_com_dist=3.20, stable_shell_k=20.0,
        use_remaining_shell_vertices_loss=False, remaining_shell_vertices_loss_coeff=1.0
):

    if not use_abduction and not use_stable_shell and not use_remaining_shell_vertices_loss:
        loss_fn = lambda body, params, complex_info: 0.0
        loss_terms_fn = lambda body, params, complex_info: (0.0, 0.0, 0.0)
        return loss_fn, loss_terms_fn

    d = vmap(displacement_fn, (0, None))

    def abduction_loss(body):
        shell_body = body[:-1]
        disps = d(shell_body.center, body[vertex_to_bind].center)
        dists = space.distance(disps)
        vertex_far_from_shell = -jnp.sum(dists)
        return vertex_far_from_shell

    spring_potential = lambda dr, r0, k: k*(dr-r0)**2
    wide_spring = lambda dr: jnp.where(dr < min_com_dist,
                                       spring_potential(dr, min_com_dist, stable_shell_k),
                                       jnp.where(dr > max_com_dist,
                                                 spring_potential(dr, max_com_dist, stable_shell_k),
                                                 0.0))
    mapped_displacement = vmap(displacement_fn, (0, None))
    def stable_shell_loss(body):
        shell_body = body[:-1]
        remaining_vertices = jnp.concatenate(
            [shell_body.center[:vertex_to_bind], shell_body.center[vertex_to_bind+1:]],
            axis=0)
        remaining_com = jnp.mean(remaining_vertices, axis=0)
        com_dists = space.distance(mapped_displacement(remaining_vertices, remaining_com))
        return wide_spring(com_dists).sum()

    def remaining_shell_vertices_loss(body, params, complex_info):
        head_remaining_shell_energy_fn = complex_info.get_head_remaining_shell_energy_fn(
            jnp.exp(params["log_morse_shell_center_spider_head_eps"]),
            params["morse_shell_center_spider_head_alpha"],
            params["morse_r_onset"], params["morse_r_cutoff"])
        return head_remaining_shell_energy_fn(body)**2 * remaining_shell_vertices_loss_coeff


    use_abduction_bit = int(use_abduction)
    use_stable_shell_bit = int(use_stable_shell)
    use_remaining_shell_vertices_bit = int(use_remaining_shell_vertices_loss)


    def loss_terms_fn(body, params, complex_info):
        abduction_term = abduction_loss(body)*use_abduction_bit
        stable_shell_term = stable_shell_loss(body)*use_stable_shell_bit
        remaining_energy_term = remaining_shell_vertices_loss(body, params, complex_info)*use_remaining_shell_vertices_bit
        norm = body[:-1].center.shape[0] - 1
        return abduction_term / norm, stable_shell_term / norm, remaining_energy_term / norm

    def loss_fn(body, params, complex_info):
        t1, t2, t3 = loss_terms_fn(body, params, complex_info)
        return t1 + t2 + t3

    return loss_fn, loss_terms_fn
