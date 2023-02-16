import pdb
import numpy as onp
import functools
from typing import Optional, Tuple, Dict, Callable, List, Union
import matplotlib.pyplot as plt

import jax.numpy as np

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

#https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def dist_point_to_line_segment(line_p1, line_p2, point):
  disp_line = displacement_fn(line_p2, line_p1)
  norm = space.distance(disp_line)
  u = (displacement_fn(point, line_p1) * disp_line) / norm
  u = jnp.where(u > 1, 1, u)
  u = jnp.where(u < 0, 0, u)
  pt = line_p1 + u * disp_line
  d_pt = displacement_fn(pt, point)
  return space.distance(d_pt)



if __name__ == "__main__":
  pt_1 = jnp.array([0, 0, 0])
  pt_2 = jnp.array([1, 0, 0])
  point_for_dist = jnp.array([[0.5, 0.5, 0],
                              [1.0, 0.2, 1.1]])
  print(vmap(dist_point_to_line_segment, in_axes=(None, None, 0))(pt_1, pt_2, point_for_dist))
  #print(grad(dist_point_to_line_segment, 2)(pt_1, pt_2, point_for_dist))
