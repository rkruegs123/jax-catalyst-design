import pdb
import unittest

from jax import vmap, random
import jax.numpy as jnp

from jax_md import space

from catalyst.complex_getter import ComplexInfo

from jax.config import config
config.update('jax_enable_x64', True)


def get_loss_fn(
        displacement_fn, vertex_to_bind, use_abduction=True,
        use_stable_shell=False, min_com_dist=3.4, max_com_dist=4.25, stable_shell_k=20.0):

    if not use_abduction and not use_stable_shell:
        raise RuntimeError(f"At least one term must be included in the loss function")

    d = vmap(displacement_fn, (0, None))

    def abduction_loss(body):
        shell_body = body[:-1]
        disps = d(shell_body.center, body[vertex_to_bind].center)
        dists = space.distance(disps)
        vertex_far_from_icos = -jnp.sum(dists)
        return vertex_far_from_icos

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

    use_abduction_bit = int(use_abduction)
    use_stable_shell_bit = int(use_stable_shell)

    def loss_fn(body):
        unnormalized_loss = abduction_loss(body)*use_abduction_bit \
                            + stable_shell_loss(body)*use_stable_shell_bit
        norm = body[:-1].center.shape[0] - 1
        return unnormalized_loss / norm

    return loss_fn


class TestLoss(unittest.TestCase):

    def test_loss_fn(self):
        displacement_fn, shift_fn = space.free()
        complex_info = ComplexInfo(
            initial_separation_coeff=0.1, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn,
            spider_base_radius=5.0, spider_head_height=4.0,
            spider_base_particle_radius=0.5, spider_head_particle_radius=0.5,
            spider_point_mass=1.0, spider_mass_err=1e-6
        )
        loss_fn = get_loss_fn(displacement_fn, complex_info.vertex_to_bind_idx,
                              use_abduction=True, use_stable_shell=False)
        init_loss = loss_fn(complex_info.rigid_body)
        print(f"Initial loss: {init_loss}")


if __name__ == "__main__":
    unittest.main()
