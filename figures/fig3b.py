import matplotlib.pyplot as plt
import pdb
import numpy as np
from brokenaxes import brokenaxes
from matplotlib import rc


rc('text', usetex=True)
plt.rcParams.update({'font.size': 32})

abduction_loss_fpath = "figures/data/fig3/abduction_loss.txt"
diffusive_loss_fpath = "figures/data/fig3/diffusive_loss.txt"

with open(abduction_loss_fpath, "r") as f:
    abduction_loss_lines = f.readlines()
abduction_losses = [float(l.strip()) for l in abduction_loss_lines if l.strip()]

with open(diffusive_loss_fpath, "r") as f:
    diffusive_loss_lines = f.readlines()
diffusive_losses = [float(l.strip()) for l in diffusive_loss_lines if l.strip()]

max_iter = min(len(abduction_losses), len(diffusive_losses))
# max_iter = 2500
abduction_losses = abduction_losses[:max_iter]
diffusive_losses = diffusive_losses[:max_iter]


"""
# plt.plot(abduction_losses)
plt.plot(diffusive_losses)
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
# plt.legend()
plt.tight_layout()
plt.show()


plt.clf()
"""









np.random.seed(19680801)

pts = np.random.rand(30)*.2
# Now let's make two outlier points which are far away from everything.
pts[[3, 14]] += .8

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax1) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

# plot the same data on both axes
ax1.plot(abduction_losses, color="red", label="Strong Interaction")
ax2.plot(abduction_losses, color="red")
ax1.plot(diffusive_losses, color="blue", label="Weak Interaction")
ax2.plot(diffusive_losses, color="blue",)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(0, 110)  # outliers only
ax2.set_ylim(-12, -3)  # most of the data

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()


d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.set_yticks([0, 50, 100])
ax2.set_yticks([-4, -8, -12])
# ax2.set_xticks([0, 500, 1000, 1500, 2000])
ax2.set_xticks([0, 1000, 2000])
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Total Loss")
ax2.yaxis.set_label_coords(-0.075, 1.0)



legend = ax1.legend(title=r'\underline{Initialization}', prop={'size': 24})
legend.get_title().set_fontsize('24')

# plt.show()
plt.savefig("fig3b.png")
