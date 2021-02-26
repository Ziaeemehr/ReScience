import igraph
import numpy as np
import pylab as plt
import networkx as nx
from os.path import join
from scipy.optimize import curve_fit
from numpy.linalg import matrix_power
from mpl_toolkits.axes_grid1 import make_axes_locatable


def display_time(time):

    hour = int(time/3600)
    minute = (int(time % 3600)) // 60
    second = time - (3600. * hour + 60. * minute)
    print("Done in %d hours %d minutes %09.6f seconds"
          % (hour, minute, second))
# ---------------------------------------------------------#


def generate_random_graph(N,
                          p,
                          seed=None,
                          verbocity=True,
                          max_num_trial=100,
                          file_name="adj.txt",
                          ):
    # Generate Random graph and save it to file
    n = 0
    for n in range(max_num_trial):
        G = nx.gnp_random_graph(N, p, seed=seed)

        adj = nx.to_numpy_array(G, dtype=int)
        g = nx.from_numpy_array(adj)

        if nx.is_connected(g):
            np.savetxt(file_name+".txt",
                       adj,
                       fmt="%d")
            if verbocity:
                print(nx.info(G))
            return 0

    print("could not make a connected random graph")
    exit(0)
# ------------------------------------------------------------------#
