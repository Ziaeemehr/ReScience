from os.path import join
import numpy as np 
import pylab as plt 
from run import *


def plot_loop(filename, **kwargs):
    fig, ax = plt.subplots(1, figsize=(6,4))

    fw = np.loadtxt(join(data_path, f"FW-{label}.txt"))
    bw = np.loadtxt(join(data_path, f"BW-{label}.txt"))

    ax.plot(fw[:, 0], fw[:, 1], lw=1, label="FW", color='r', **kwargs)
    ax.plot(bw[:, 0], bw[:, 1], lw=1, label="BW", color='b', **kwargs)
    ax.set_xlabel("coupling")
    ax.set_ylabel("R")
    ax.legend(loc='lower right')
    ax.margins(x=0.01)
    ax.set_ylim(0,1.01)
    plt.tight_layout()
    plt.savefig(join(data_path, "H_loop.png"), dpi=150)
    plt.close()


if __name__ == "__main__":

    for label in network_labels:
        plot_loop(label, marker='o', ms=3)
