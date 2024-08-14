from pathlib import Path
import matplotlib.pyplot as plt
import pdb
import numpy as onp
from matplotlib import rc


font = {'size': 28}
rc('font', **font)
rc('text', usetex=True)

def get_iter_params(params_iter_path, i, lines_per_iter=11):
    with open(params_iter_path, "r") as f:
        param_iter_lines_lines = f.readlines()
    iter_lines = param_iter_lines_lines[i*lines_per_iter:(i+1)*lines_per_iter]
    assert(not iter_lines[0].strip())
    assert(not iter_lines[-1].strip())
    assert(iter_lines[1][:9] == "Iteration")
    iter_lines = iter_lines[2:-1]

    iter_params = dict()
    for l in iter_lines:
        tokens = l.split(':')
        key = tokens[0][2:]
        val = float(tokens[1].strip())
        iter_params[key] = val

    processed_iter_params = dict()
    for k, v in iter_params.items():
        if k == "spider_base_radius":
            processed_iter_params["spider_base_diameter"] = v*2
        elif k == "spider_head_height":
            processed_iter_params["spider_leg_length"] = onp.sqrt(iter_params["spider_base_radius"]**2 + iter_params["spider_head_height"]**2) # pythag
        else:
            processed_iter_params[k] = v
    return processed_iter_params

label_mapper = {
    "log_morse_shell_center_spider_head_eps": r'$\log (\epsilon_H)$',
    "morse_r_cutoff": r'$r_{cut}$',
    "morse_r_onset": r'$r_{on}$',
    "morse_shell_center_spider_head_alpha": r'$\alpha_{H}$',
    "spider_base_particle_radius": r'$r_{BP}$',
    "spider_base_diameter": r'$d$',
    "spider_leg_length": r'$\ell$',
    "spider_head_particle_radius": r'$r_H$'
}

label_keys = list(label_mapper.keys())
xs = onp.arange(len(label_keys))
labels = list(label_mapper.values())

supp_data_basedir = Path("figures/revisions/data/supp/")

diffusive_prefix = "production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0"
explosive_prefix = "production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0"
all_plots_info = [
    (supp_data_basedir / "diffusive-keys", "Low Energy Limit, Random Seeds", f"{diffusive_prefix}-k"),
    (supp_data_basedir / "diffusive-perturb", f'Low Energy Limit, Perturbed $\log (\epsilon_h)$', f"{diffusive_prefix}-perturb"),
    (supp_data_basedir / "explosive-keys", "High Energy Limit, Random Seeds", f"{explosive_prefix}-k"),
    (supp_data_basedir / "explosive-perturb", f'High Energy Limit, Perturbed $\log (\epsilon_h)$', f"{explosive_prefix}-perturb")
]


output_basedir = Path("figures/revisions/output/supp/")
assert(output_basedir.exists())

for curr_basedir, title, prefix in all_plots_info:

    all_vals = dict()
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["red", "blue", "green", "purple", "orange"]
    for k in range(1, 6):
        key_dir = curr_basedir / f"{prefix}{k}"
        key_losses = onp.loadtxt(key_dir / "loss.txt")
        min_loss_iter = onp.argmin(key_losses)
        key_params = get_iter_params(key_dir / "params_per_iter.txt", min_loss_iter)

        vals = list()
        for label_key in label_keys:
            vals.append(key_params[label_key])
        ax.scatter(xs, vals, color=colors[k-1], s=100)

        for k, v in key_params.items():
            if k not in all_vals:
                all_vals[k] = list()
            all_vals[k].append(v)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Parameter Value")

    ax.set_title(title)

    plt.tight_layout()

    plt.show()
    # plt.savefig(output_basedir / FIXME) # FIXME: need to set the right title
    plt.close()
