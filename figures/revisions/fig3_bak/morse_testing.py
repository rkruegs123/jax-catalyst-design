import numpy as onp
import pdb
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc

from jax_md import energy


rc('text', usetex=True)
plt.rcParams.update({'font.size': 24})



# Initial, diffusive
init_params = {
    "spider_base_radius": 5.0,
    "spider_head_height": 5.0,
    "spider_base_particle_radius": 1.0,
    "spider_head_particle_radius": 1.0,
    "log_morse_shell_center_spider_head_eps": 3.0,
    "morse_shell_center_spider_head_alpha": 1.5,
    "morse_r_onset": 10.0,
    "morse_r_cutoff": 12.0
}

# production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 3000
opt_params = {
    "log_morse_shell_center_spider_head_eps": 7.2612795505931995,
    "morse_r_cutoff": 10.285149443604414,
    "morse_r_onset": 9.736256086615986,
    "morse_shell_center_spider_head_alpha": 1.910963444404297,
    "spider_base_particle_radius": 1.0066685401207762,
    "spider_base_radius": 4.7113923231913954,
    "spider_head_height": 5.250512398201004,
    "spider_head_particle_radius": 1.1670004853096905
}


shell_vertex_radius = 2.0
init_sigma = shell_vertex_radius + init_params['spider_head_particle_radius']
opt_sigma = shell_vertex_radius + opt_params['spider_head_particle_radius']


min_dist = onp.max([init_sigma, opt_sigma]) - 0.35
# test_distances = onp.linspace(onp.max([init_sigma, opt_sigma]) - 0.35, 12.0, 250)
test_distances = onp.linspace(min_dist, 6.0, 250)
print(min_dist)

init_morse_fn = lambda r: energy.multiplicative_isotropic_cutoff(
    energy.morse,
    r_onset=init_params['morse_r_onset'],
    r_cutoff=init_params['morse_r_cutoff'])(r, sigma=init_sigma, epsilon=onp.exp(init_params['log_morse_shell_center_spider_head_eps']), alpha=init_params['morse_shell_center_spider_head_alpha'])

opt_morse_fn = lambda r: energy.multiplicative_isotropic_cutoff(
    energy.morse,
    r_onset=opt_params['morse_r_onset'],
    r_cutoff=opt_params['morse_r_cutoff'])(r, sigma=opt_sigma, epsilon=onp.exp(opt_params['log_morse_shell_center_spider_head_eps']), alpha=opt_params['morse_shell_center_spider_head_alpha'])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
ax1.plot(test_distances, init_morse_fn(test_distances), label=f"Initial")
ax1.set_xlabel("Distance")
ax1.set_ylabel("Energy (kT)")
ax1.set_title("Initial")

ax2.plot(test_distances, opt_morse_fn(test_distances), label=f"Optimized")
ax2.set_xlabel("Distance")
ax2.set_ylabel("Energy (kT)")
ax2.set_title("Optimized")

# ax.legend()
plt.show()
plt.close()



fig, ax = plt.subplots(figsize=(16, 12))

ax.plot(test_distances, init_morse_fn(test_distances), label=f"Initial")
ax.plot(test_distances, opt_morse_fn(test_distances), label=f"Optimized")

ax.set_xlabel("Distance")
ax.set_ylabel("Energy (kT)")
ax.legend()
plt.show()
