from os.path import join
import numpy as np
import pylab as plt
from run import *


def plot_loop(filename, **kwargs):
    fig, ax = plt.subplots(1, figsize=(6, 4))

    fw = np.loadtxt(join(data_path, "text", f"FW-{label}.txt"))
    bw = np.loadtxt(join(data_path, "text", f"BW-{label}.txt"))

    ax.plot(fw[:, 0], fw[:, 1], lw=1, label="fw1", color='r', **kwargs)
    ax.plot(bw[:, 0], bw[:, 1], lw=1, label="bw1", color='b', **kwargs)

    ax.plot(fw[:, 0], fw[:, 2], lw=1, label="fw2", color='g', **kwargs)
    ax.plot(bw[:, 0], bw[:, 2], lw=1, label="bw2", color='orange', **kwargs)

    ax.set_xlabel("coupling")
    ax.set_ylabel("R")
    ax.legend(loc='lower right')
    ax.margins(x=0.01)
    ax.set_ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig(join(data_path, "fig", f"{label}.png"), dpi=150)
    plt.close()


def plot_phase_correlation(file_name, g, **kwargs):

    omega = np.loadtxt(join(data_path, "text", f"omega{label}.txt"))
    data = np.loadtxt(join(data_path, "text",
                           "{:s}-{:.6f}.txt".format(label, g)))
    n = data.shape[0]
    fig, ax = plt.subplots(ncols=n, figsize=(n*2.5, 3.5))
    for i in range(n):
        ax[i].scatter(omega, data[i, :], color='k', s=10)
        ax[i].set_xlabel(r"$\omega_i$")
        ax[i].set_ylabel(r"$\alpha_i \lambda$")
    plt.tight_layout()
    filename = join(data_path, "fig", "c-{:s}-{:.3f}.png".format(label, g))
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":

    for label in network_labels:
        plot_loop(label, marker='o', ms=3)

    for label in network_labels:
        for g in np.arange(gi, gf, dg):
            plot_phase_correlation(label, g)
