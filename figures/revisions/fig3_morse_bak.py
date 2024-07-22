import numpy as onp
import pdb
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc

from jax_md import energy


rc('text', usetex=True)
plt.rcParams.update({'font.size': 24})


mode = "diffusive"
# mode = "explosive"
if mode == "diffusive":
    basedir = Path("figures/revisions/data/fig2/diffusive-run/")
    lo_iter = 250
    hi_iter = 450
else:
    raise RuntimeError(f"Invalid mode: {mode}")

params_per_iter_fpath = basedir / "params_per_iter.txt"

with open(params_per_iter_fpath, "r") as f:
    unprocessed_lines = f.readlines()

processed_lines =  list()
for line in unprocessed_lines:
    if line.strip():
        processed_lines.append(line)


def get_params(iter_idx):
    n_params = 8
    start_idx = (1+n_params)*iter_idx
    end_idx = start_idx + (1+n_params)
    params_lines = processed_lines[start_idx:end_idx][1:]
    params = dict()
    for p_line in params_lines:
        tokens = p_line.strip()[2:].split(": ")
        params[tokens[0]] = float(tokens[1])

    return params

lo_params = get_params(lo_iter)
hi_params = get_params(hi_iter)


shell_vertex_radius = 2.0
lo_sigma = shell_vertex_radius + lo_params['spider_head_particle_radius']
hi_sigma = shell_vertex_radius + hi_params['spider_head_particle_radius']

test_distances = onp.linspace(onp.max([lo_sigma, hi_sigma]) - 0.35, 7.0, 100)

lo_morse_fn = lambda r: energy.multiplicative_isotropic_cutoff(
    energy.morse,
    r_onset=lo_params['morse_r_onset'],
    r_cutoff=lo_params['morse_r_cutoff'])(r, sigma=lo_sigma, epsilon=onp.exp(lo_params['log_morse_shell_center_spider_head_eps']), alpha=lo_params['morse_shell_center_spider_head_alpha'])

hi_morse_fn = lambda r: energy.multiplicative_isotropic_cutoff(
    energy.morse,
    r_onset=hi_params['morse_r_onset'],
    r_cutoff=hi_params['morse_r_cutoff'])(r, sigma=hi_sigma, epsilon=onp.exp(hi_params['log_morse_shell_center_spider_head_eps']), alpha=hi_params['morse_shell_center_spider_head_alpha'])


fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(test_distances, lo_morse_fn(test_distances), label=f"Iteration {lo_iter}")
ax.plot(test_distances, hi_morse_fn(test_distances), label=f"Iteration {hi_iter}")
ax.set_xlabel("Distance")
ax.set_ylabel("Energy (kT)")
ax.legend()
plt.show()
