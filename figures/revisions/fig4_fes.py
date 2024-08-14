import numpy as onp
import pdb
from tqdm import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib import rc

from jax import jit, vmap
import jax.numpy as jnp


rc('text', usetex=True)
plt.rcParams.update({'font.size': 32})


def read_fes(wham_out_path, n_bins):
    ## Read in the free energies (per R) and free energies (per bin)
    with open(wham_out_path, "r") as f:
        wham_lines = f.readlines()
    pmf_lines = wham_lines[:n_bins+1] # really free energies
    hist_fe_lines = wham_lines[n_bins+1:]

    ### pmf data (really free energies)
    assert(pmf_lines[0][:5] == "#Coor")
    header = pmf_lines[0][1:].split()
    len_header = len(header)
    if len_header == 5:
        assert(header == ["Coor", "Free", "+/-", "Prob", "+/-"])
    elif len_header == 3:
        assert(header == ["Coor", "Free", "+/-"])
    all_ex_ops = list()
    all_ex_fes = list()
    all_ex_probs = list()
    for line in pmf_lines[1:]:
        assert(line[0] != "#")
        tokens = line.split()

        op = float(tokens[0])
        all_ex_ops.append(op)

        fe = float(tokens[1])
        all_ex_fes.append(fe)

        if len_header == 5:
            prob = float(tokens[3])
            all_ex_probs.append(prob)
    all_ex_ops = onp.array(all_ex_ops)
    all_ex_fes = onp.array(all_ex_fes)
    all_ex_probs = onp.array(all_ex_probs)

    assert(hist_fe_lines[0][:7] == "#Window")
    header = hist_fe_lines[0][1:].split()
    len_header = len(header)
    if len_header == 3:
        assert(header == ["Window", "Free", "+/-"])
    elif len_header == 2:
        assert(header == ["Window", "Free"])
    bin_idxs = list()
    bin_fes = list()
    for line in hist_fe_lines[1:]:
        assert(line[0] == "#")
        tokens = line[1:].split()

        bin_idx = int(tokens[0])
        bin_idxs.append(bin_idx)

        bin_fe = float(tokens[1])
        bin_fes.append(bin_fe)
    bin_idx = onp.array(bin_idxs)
    bin_fes = onp.array(bin_fes)



    return all_ex_ops, all_ex_fes


basedir = Path(f"figures/revisions/data/fig4/")
output_basedir = Path("figures/revisions/output/fig4/")
assert(output_basedir.exists())

# for figsize_x, figsize_y in [(12, 10), (14, 10)]:
for figsize_x, figsize_y in [(14, 11.5)]:

    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    colors = ["#005AB5", "#DC3220"]
    modes = ["init-fe", "opt-fe"]
    labels = ["Initial", "Optimized"]
    for color, mode, label in zip(colors, modes, labels):

        ops, fes = read_fes(basedir / f"{mode}/wham/analysis.txt", 500)

        # ax.plot(ops_op1, fes_op1, label=r"$r_{v,\bar{a}}$")
        ax.plot(ops, fes, label=label, color=color, linewidth=3)

    save_fname = f"fes_{figsize_x}_{figsize_y}.pdf"
    save_fpath = str(output_basedir / save_fname)

    ax.set_xlabel("Order Parameter")
    ax.set_ylabel("Free Energy (kT)")
    # y_ticks = [0, 100, 200, 300, 400]
    y_ticks = [0, 100, 200, 300]
    ax.set_yticks(y_ticks)

    ax.legend(ncol=2, bbox_to_anchor=(0.75, 1.125), prop={'size': 28})

    plt.tight_layout()

    plt.savefig(save_fpath)
    plt.clf()

    # plt.show()
    # plt.close()
