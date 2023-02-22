import pdb
import numpy as onp
import functools
from typing import Optional, Tuple, Dict, Callable, List, Union
import matplotlib.pyplot as plt

import jax.numpy as np

import jax.debug
from jax import random, grad, value_and_grad, remat, jacfwd
from jax import jit
from jax import vmap, lax
from jax import ops
from jax.config import config
config.update('jax_enable_x64', True)

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition # , rigid_body
from jax_md import dataclasses
from jax_md import util

import mod_rigid_body as rigid_body

from common import displacement_fn, shift_fn, d, d_prod

# https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def dist_point_to_line_segment(line_points, point):
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


bond_pairs = jnp.array([
    [-6, -1],
    [-5, -1],
    [-4, -1],
    [-3, -1],
    [-2, -1]
], dtype=jnp.int32)
vertices = jnp.arange(0, 6*12, 6, dtype=jnp.int32)
test_idx = onp.array([0, 1])
def get_leg_energy_fn(soft_sphere_eps, bond_diameter, shape, shape_species):
    def leg_energy_fn(body):
        position, _ = rigid_body.union_to_points(body, shape, shape_species)

        # We want to compute 12 * 5 distances. For each bond, distance to each vertex
        # return dist_point_to_line_segment(position[0], position[1], position[3])
        all_dists_fn = vmap(vmap(dist_point_to_line_segment, (0, None)), (None, 0))
        all_dists = all_dists_fn(position[bond_pairs], position[vertices])
        # return jnp.sum(all_dists)

        # Get the soft sphere attraction for each
        bond_energy_sm = jnp.sum(
            energy.soft_sphere(all_dists,
                               epsilon=soft_sphere_eps,
                               sigma=bond_diameter,
                               alpha=2))
        return bond_energy_sm

    return leg_energy_fn


if __name__ == "__main__":
    #pt_1 = jnp.array([0, 0, 0])
    #pt_2 = jnp.array([1, 0, 0])
    #line_pts = jnp.array([pt_1, pt_2])
    #point_for_dist = jnp.array([[1.5, 0.5, 0],
    #                            [0.1, 0.5, 0], 
    #                            [0, 0, 0.5]])

    point_for_dist = jnp.array([-2.16241278, -0.58582545,  3.10412613])
    pt_1 = jnp.array([-5.57976037, -1.48088655,  8.11028434])
    pt_2 = jnp.array([-1.94783784,  4.29471522,  4.45723335])

    print(dist_point_to_line_segment(jnp.array([pt_1, pt_2]), point_for_dist))
    print(dist_point_to_line_segment(jnp.array([pt_2, pt_1]), point_for_dist))
    #print(vmap(dist_point_to_line_segment, in_axes=(None, 0))(line_pts, point_for_dist))
    # print(grad(dist_point_to_line_segment, 2)(pt_1, pt_2, point_for_dist))
