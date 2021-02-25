import os
import igraph
import numpy as np
import networkx as nx
from os import system
from time import time
from os.path import join
from threading import Thread
from joblib import Parallel, delayed
from lib import generate_random_graph, display_time
# ---------------------------------------------------------#


# preparing the directories------------------------------------------
directories = ["../data", "../data/fig", "../data/text"]
for d in directories:
    if not os.path.exists(d):
        os.makedirs(d)
# -------------------------------------------------------------------


def run_command(arg):
    command = "{0} {1} {2} {3} {4} {5} {6} \
    {7} {8} {9} {10}".format(*arg)
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
                    ])

    Parallel(n_jobs=n_jobs)(
        map(delayed(run_command), arg))


if __name__ == "__main__":

    data_path = "../data/text"
    N = 50
    graph_p = 0.2
    network_labels = ["A"]
    t_transition = 50.
    t_simulation = 100.
    dt = 0.05
    gi, gf, dg = 0, 0.5, 0.05
    fraction = 1.
    RANDOMNESS = 0
    num_threads = 1
    n_jobs = 1
    seed = 1

    for i in network_labels:
        generate_random_graph(N, graph_p,
                              seed=seed,
                              verbocity=False,
                              file_name=join(data_path, i))

    start = time()
    batch_run()
    # display_time(time()-start)
