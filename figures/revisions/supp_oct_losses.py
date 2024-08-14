import matplotlib.pyplot as plt
import pdb
import numpy as np
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc


rc('text', usetex=True)

plt.rcParams.update({'font.size': 24})

output_basedir = Path("figures/revisions/output/supp/")
assert(output_basedir.exists())

supp_data_basedir = Path("figures/revisions/data/supp")

# mode = "diffusive"
mode = "explosive"
if mode == "diffusive":
    loss_fpath = supp_data_basedir / "octahedron-diffusive" / "loss.txt"
    max_iter = 4000
elif mode == "explosive":
    loss_fpath = supp_data_basedir / "octahedron-explosive" / "loss.txt"
    max_iter = 4000
else:
    raise RuntimeError(f"Invalid mode: {mode}")


losses = np.loadtxt(loss_fpath)
losses = losses[:max_iter+1]

legend_text_size = 22

if mode == "diffusive":
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(losses, color="black")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
elif mode == "explosive":

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.yaxis.set_label_coords(-0.075, 1.0)

    divider = make_axes_locatable(ax)
    ax2 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax2)

    ax.plot(losses, color="black")
    ax.set_ylim(-12, 10)
    ax.set_yticks([-10, -5, 0, 5])
    ax.spines['top'].set_visible(False)
    ax2.plot(losses, color="black")
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





plt.tight_layout()

if mode == "diffusive":
    fname = "octa_diffusive_loss.svg"
elif mode == "explosive":
    fname = "octa_explosive_loss.svg"
fpath = str(output_basedir / fname)

plt.show()
# plt.savefig(fpath)
