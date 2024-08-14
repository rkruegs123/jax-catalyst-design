from pathlib import Path
import numpy as onp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pdb

rc('text', usetex=True)
plt.rcParams.update({'font.size': 32})


opt_basedir = Path("figures/revisions/data/fig4/opt-run")

loss_path = opt_basedir / "loss.txt"
losses = onp.loadtxt(loss_path)

release_loss_path = opt_basedir / "release_loss.txt"
release_losses = onp.loadtxt(release_loss_path)

remaining_energy_loss_path = opt_basedir / "remaining_energy_loss.txt"
remaining_energy_losses = onp.loadtxt(remaining_energy_loss_path)

extraction_loss_path = opt_basedir / "extract_loss.txt"
extraction_losses = onp.loadtxt(extraction_loss_path)

activated_extraction_loss_path = opt_basedir / "activated_extract_loss.txt"
activated_extraction_losses = onp.loadtxt(activated_extraction_loss_path)



"""
n_iters = len(losses)

loss_terms_path = opt_basedir / "loss_terms.txt"
with open(loss_terms_path, "r") as f:
    loss_terms_lines = f.readlines()
lines_per_iter = 14
iter_loss_terms = [loss_terms_lines[i*lines_per_iter:(i+1)*lines_per_iter] for i in range(n_iters)]

extraction_losses = list()
remaining_energy_losses = list()
for i in range(n_iters):
    iter_lines = iter_loss_terms[i]
    if not iter_lines:
        break

    extraction_loss_line = iter_lines[11]
    extraction_loss = float(extraction_loss_line.strip().split(':')[1].strip())
    extraction_losses.append(extraction_loss)

    remaining_energy_loss_line = iter_lines[12]
    remaining_energy_loss = float(remaining_energy_loss_line.strip().split(':')[1].strip())
    remaining_energy_losses.append(remaining_energy_loss)
"""

output_basedir = Path("figures/revisions/output/fig4/")
assert(output_basedir.exists())
# for figsize_x, figsize_y in [(12, 10), (12, 12), (10, 12)]:
for figsize_x, figsize_y in [(14, 12)]:
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    ax.plot(release_losses, label="Release", color="#332288")
    ax.plot(remaining_energy_losses, label="Remaining Energy", color="#44AA99")
    ax.plot(losses, label="Total", color="#DDCC77")
    # ax.plot(extraction_losses, label="Extraction Term", color="#AA4499")
    ax.plot(activated_extraction_losses, label="Extraction", color="#AA4499")


    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss Value")
    ax.legend(ncol=2, bbox_to_anchor=(0.9, 1.3), prop={'size': 28}, title="Loss Term")
    plt.tight_layout()

    # plt.show()
    # plt.close()

    plt.savefig(output_basedir / "loss_terms.pdf")
    plt.close()
