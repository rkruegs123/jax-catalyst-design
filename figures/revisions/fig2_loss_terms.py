import matplotlib.pyplot as plt
import pdb
import numpy as np
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc


rc('text', usetex=True)

plt.rcParams.update({'font.size': 24})

output_basedir = Path("figures/revisions/output/fig2/")
assert(output_basedir.exists())

mode = "diffusive"
# mode = "explosive"
if mode == "diffusive":
    terms_fpath = "figures/revisions/data/fig2/diffusive-run/loss_terms.txt"
    loss_fpath = "figures/revisions/data/fig2/diffusive-run/loss.txt"
    max_iter = 3000
    # max_iter = 500
elif mode == "explosive":
    terms_fpath = "figures/revisions/data/fig2/explosive-run/loss_terms.txt"
    loss_fpath = "figures/revisions/data/fig2/explosive-run/loss.txt"
    max_iter = 3000
else:
    raise RuntimeError(f"Invalid mode: {mode}")

with open(terms_fpath, "r") as f:
    unprocessed_lines = f.readlines()

lines = list()
for l in unprocessed_lines:
    if l.strip():
        lines.append(l.strip())

lines_per_iter = 13
assert(len(lines) % lines_per_iter == 0)
n_iters = len(lines) // lines_per_iter

all_abduction_losses = list()
all_stability_losses = list()
all_energy_losses = list()
for i in range(n_iters):
    iter_lines = lines[i*lines_per_iter:(i+1)*lines_per_iter]
    avg_term_lines = iter_lines[-3:]
    terms = [float(tl.strip().split(':')[-1].strip()) for tl in avg_term_lines]
    all_abduction_losses.append(terms[0])
    all_stability_losses.append(terms[1])
    all_energy_losses.append(terms[2])


with open(loss_fpath, "r") as f:
    unprocessed_loss_lines = f.readlines()
losses = [float(l.strip()) for l in unprocessed_loss_lines if l.strip()]

legend_text_size = 22

if mode == "diffusive":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[1.5, 1], sharex=True)

    ax1.plot(losses[:max_iter], color="black")
    # ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Loss")

    ax2.plot(all_abduction_losses[:max_iter], label="Extraction Term", color="purple")
    # ax2.plot(all_stability_losses, label="Stability")
    ax2.plot(all_energy_losses[:max_iter], label="Remaining Energy Term", color="orange")
    # ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, shadow=True, prop={'size': 16})
    # ax2.legend(loc="upper right", prop={'size': legend_text_size})
    # ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.925), prop={'size': legend_text_size})
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), prop={'size': legend_text_size})
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss Term")
elif mode == "explosive":

    # fig, axes = plt.subplots(nrows=2, sharex=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[1.5, 1], sharex=True)

    axes[0].set_ylabel("Total Loss")
    # axes[0].yaxis.set_label_coords(0.00, 0.0, transform=fig.transFigure)
    axes[0].yaxis.set_label_coords(-0.075, 1.0)

    ax = axes[0]
    divider = make_axes_locatable(ax)
    ax2 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax2)

    ax.plot(losses[:max_iter], color="black")
    ax.set_ylim(-12, 10)
    ax.set_yticks([-10, -5, 0, 5])
    ax.spines['top'].set_visible(False)
    ax2.plot(losses[:max_iter], color="black")
    ax2.set_ylim(50, 1000)
    ax2.tick_params(bottom=False, labelbottom=False)
    ax2.spines['bottom'].set_visible(False)


    # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



    # create bottom subplot as usual
    axes[1].plot(all_abduction_losses[:max_iter], label="Extraction Term", color="purple")
    # axes[1].plot(all_stability_losses, label="Stability")
    axes[1].plot(np.log(all_energy_losses)[:max_iter], label="Log Remaining Energy Term", color="orange")
    # axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, shadow=True, prop={'size': 16})
    axes[1].legend(loc="upper right", prop={'size': legend_text_size})
    # axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss Term")




plt.tight_layout()

if mode == "diffusive":
    fname = "diffusive_loss.svg"
elif mode == "explosive":
    fname = "explosive_loss.svg"
fpath = str(output_basedir / fname)

# plt.show()
plt.savefig(fpath)
