import os
import igraph
import numpy as np
import pylab as plt
import networkx as nx
from os import system
from time import time
from os.path import join
from threading import Thread
from joblib import Parallel, delayed
from lib import generate_random_graph, display_time
plt.switch_backend('agg')
# ---------------------------------------------------------#


# preparing the directories------------------------------------------
directories = ["../data", "../data/text", "../data/fig"]
for d in directories:
    if not os.path.exists(d):
        os.makedirs(d)
# -------------------------------------------------------------------


def run_command(arg):
    command = "{0} {1} {2} {3} {4} {5} {6} \
    {7} {8} {9} {10} {11}".format(*arg)
    system("./prog " + command)
# ---------------------------------------------------------#


def batch_run():
    arg = []
    for label in network_labels:
        arg.append([N,
                    dt,
                    t_transition,
                    t_simulation,
                    gi,
                    gf,
                    dg,
                    fraction,
                    num_threads,
                    label,
                    RANDOMNESS,
                    print_snapshot,
                    ])

    Parallel(n_jobs=n_jobs)(
        map(delayed(run_command), arg))


network_labels = ["A"]
data_path = "../data/"
gi, gf, dg = 0, 0.5, 0.1

if __name__ == "__main__":

    N = 100
    graph_p0 = [0.12]
    graph_p1 = [0.12]

    t_transition = 100.
    t_simulation = 100.
    dt = 0.02
    fraction = 1.
    print_snapshot = 1
    RANDOMNESS = 0
    num_threads = 4
    n_jobs = 1
    seed = 1

    for i in range(len(network_labels)):
        generate_random_graph(N, graph_p0[i],
                              seed=seed,
                              verbocity=False,
                              file_name=join(data_path, "text",
                                             network_labels[i]+"0"))
        generate_random_graph(N, graph_p1[i],
                              seed=seed+1,
                              verbocity=False,
                              file_name=join(data_path, "text",
                                             network_labels[i]+"1"))

    start = time()
    batch_run()
    display_time(time()-start)
