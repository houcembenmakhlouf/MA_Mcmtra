import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sn
import pandas as pd


def plot_classwise_prob(probs, classes, exp_path=None, title=None):
    """plots and saves the classwise probabilites.
    Args:
        probs (torch.Tensor): probs calculated with the Solver.get_classwise_prob() method. It has a shape of (num_classes, num_classes).
        classes (list of str): classes names
        exp_name (str, optional): The plot will be saved using this name. Defaults to None.
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    sn.reset_orig()

    probs = np.array(probs)

    fig, axs = plt.subplots(nrows=math.ceil(len(classes) / 2), ncols=2, figsize=(8, 18))
    plt.yticks(np.arange(0, 1))
    for i, class_name in enumerate(classes):
        barlist = axs[i // 2, i % 2].bar(np.arange(0, len(classes)), probs[i, :])
        barlist[i].set_color("r")
        axs[i // 2, i % 2].set_title(f"class {class_name}")
        axs[i // 2, i % 2].set_ylim([0, 100])
        axs[i // 2, i % 2].set_xlabel("Labels")
        axs[i // 2, i % 2].set_ylabel("Probability")

        formatter = mticker.ScalarFormatter()
        axs[i // 2, i % 2].xaxis.set_major_formatter(formatter)
        axs[i // 2, i % 2].xaxis.set_major_locator(
            mticker.FixedLocator(np.arange(0, len(classes) + 1, 1))
        )
        axs[i // 2, i % 2].yaxis.set_major_formatter(formatter)
        axs[i // 2, i % 2].yaxis.set_major_locator(
            mticker.FixedLocator(np.arange(0, 100 + 1, 20))
        )

    if title is not None:
        fig.suptitle(title, fontsize=20)
        if exp_path is not None:
            save_path = exp_path / f"classwise_prob_{title.replace(' ','_')}.png"
    else:
        if exp_path is not None:
            save_path = exp_path / f"classwise_prob.png"

    fig.tight_layout(pad=3.0)
    if exp_path is not None:
        fig.savefig(save_path, dpi=fig.dpi)

    plt.show(block=False)

    return fig


def plot_conf_matrix(conf_matrix, classes, exp_path=None, title=None):

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    sn.reset_orig()

    df_cm = pd.DataFrame(conf_matrix, classes, classes)
    df_cm = df_cm.round(1)
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes()
    sn.set(font_scale=1.3)  # for label size
    sn.heatmap(
        df_cm, annot=True, annot_kws={"size": 18}, ax=ax, cmap="YlGnBu", fmt="g"
    )  # font size
    if title is not None:
        ax.set_title(f"{title}", fontsize=18)
        if exp_path is not None:
            save_path = exp_path / f"conf_matrix_{title}.png"
    else:
        if exp_path is not None:
            save_path = exp_path / f"conf_matrix.png"

    ax.set_xlabel("predicted labels", fontsize=18)
    ax.set_ylabel("ground truth labels", fontsize=18)
    ax.tick_params(axis="x", labelsize="medium")
    ax.tick_params(axis="y", labelsize="medium")
    fig.tight_layout()
    if exp_path is not None:
        fig.savefig(save_path, dpi=fig.dpi)
    plt.show(block=False)

    return fig
