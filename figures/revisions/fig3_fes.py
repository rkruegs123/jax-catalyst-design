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


op1_basedir = Path(f"figures/revisions/data/fig3/wham-op1")
op2_basedir = Path(f"figures/revisions/data/fig3/wham-op2")

for mode in ["rigid", "flexible-23", "flexible-all"]:

    ops_op1, fes_op1 = read_fes(op1_basedir / f"{mode}/wham/analysis.txt", 500)
    ops_op2, fes_op2 = read_fes(op2_basedir / f"{mode}/wham/analysis.txt", 500)

    if mode == "rigid":
        n_skip = 20
        ops_op2 = ops_op2[n_skip:]
        fes_op2 = fes_op2[n_skip:]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(ops_op1, fes_op1, label=r"$r_{v,\bar{a}}$")
    ax.plot(ops_op2, fes_op2, label=r"$r_{v,h}$")
    ax.set_xlabel("Order Parameter")
    ax.set_ylabel("Free Energy (kT)")
    # plt.tight_layout()
    # plt.savefig(savepath)
    # plt.clf()

    # y_ticks = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    y_ticks = [0, 100, 200, 300, 400]
    ax.set_yticks(y_ticks)
    ax.legend()

    plt.show()
    plt.close()
