import pdb
import unittest

from jax import vmap, random
import jax.numpy as jnp

from jax_md import space

from catalyst.icosahedron_ext_rigid.complex import Complex, PENTAPOD_LEGS, BASE_LEGS

from jax.config import config
config.update('jax_enable_x64', True)


def get_loss_fn(
        displacement_fn, vertex_to_bind, use_abduction=True,
        use_stable_shell=False, min_com_dist=3.4, max_com_dist=4.25, stable_shell_k=20.0,
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

    def remaining_shell_vertices_loss(body, params, complex_info):
        head_remaining_shell_energy_fn = complex_info.get_head_remaining_shell_energy_fn(
            jnp.exp(params["log_morse_attr_eps"]),
            params["morse_attr_alpha"],
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


class TestLoss(unittest.TestCase):

    sim_params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': 12.0,
        'spider_base_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,
        'spider_attr_particle_pos_norm': 0.2,
        'spider_attr_site_radius': 0.3,

        # catalyst energy
        'log_morse_attr_eps': 6.0,
        'morse_attr_alpha': 1.0,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0
    }

    def test_loss_fn(self):
        displacement_fn, shift_fn = space.free()
        spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])
        complex_info = Complex(
            initial_separation_coeff=0.1, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=self.sim_params["spider_base_radius"],
            spider_head_height=self.sim_params["spider_head_height"],
            spider_base_particle_radius=self.sim_params["spider_base_particle_radius"],
            spider_head_particle_radius=self.sim_params["spider_head_particle_radius"],
            spider_attr_particle_pos_norm=self.sim_params['spider_attr_particle_pos_norm'],
            spider_attr_site_radius=self.sim_params['spider_attr_site_radius'],
            spider_point_mass=1.0, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs, spider_leg_radius=0.25
        )
        loss_fn, _ = get_loss_fn(displacement_fn, complex_info.vertex_to_bind_idx,
                              use_abduction=True, use_stable_shell=False)
        init_loss = loss_fn(complex_info.rigid_body, self.sim_params, complex_info)
        print(f"Initial loss: {init_loss}")


if __name__ == "__main__":
    unittest.main()
