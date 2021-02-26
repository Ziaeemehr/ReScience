from os.path import join
import numpy as np
import pylab as plt
from run import *


def plot_loop(filename, **kwargs):
    fig, ax = plt.subplots(1, figsize=(6, 4))

    fw = np.loadtxt(join(data_path, "text", f"FW-{label}.txt"))
    bw = np.loadtxt(join(data_path, "text", f"BW-{label}.txt"))

    ax.plot(fw[:, 0], fw[:, 1], lw=1, label="FW", color='r', **kwargs)
    ax.plot(bw[:, 0], bw[:, 1], lw=1, label="BW", color='b', **kwargs)
    ax.set_xlabel("coupling")
    ax.set_ylabel("R")
    ax.legend(loc='lower right')
    ax.margins(x=0.01)
    ax.set_ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig(join(data_path, "fig", "H_loop.png"), dpi=150)
    plt.close()


def plot_phase_correlation(ax, file_name, g, n, **kwargs):

    omega = np.loadtxt(join(data_path, "text", f"omega{label}.txt"))
    data = np.loadtxt(join(data_path, "text",
                           "{:s}-{:.6f}-{:d}.txt".format(label, g, n)))
    ax.scatter(omega, data, color='k', s=10)
    ax.set_xlabel(r"$\omega_i$")
    ax.set_ylabel(r"$\alpha_i \lambda$")


if __name__ == "__main__":

    for label in network_labels:
        plot_loop(label, marker='o', ms=3)

    ns = [12500, 23750]
    for label in network_labels:
        for g in np.arange(gi, gf, dg):
            fig, ax = plt.subplots(ncols=2, figsize=(6, 3.5))
            for n in range(2):
                plot_phase_correlation(ax[n], label, g, ns[n])
            plt.tight_layout()
            filename = join(data_path, "fig", "c-{:s}-{:.3f}.png".format(label, g))
            plt.savefig(filename)
            plt.close()
    