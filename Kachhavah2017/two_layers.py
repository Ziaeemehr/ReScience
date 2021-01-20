"""
single layer Kuramoto model second order.
"""

import os
import numpy as np
import pylab as plt
import networkx as nx
from copy import copy
from symengine import sin, Symbol
from jitcode import jitcode, y
from numpy.random import choice
from numpy.random import uniform
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(2)
os.environ["CC"] = "clang"

if not os.path.exists("data"):
    os.makedirs("data")
# -------------------------------------------------------------------

# q for number of layers, i for index of oscillators in each layer


def X(i, q): return y(q * 2 * n + i)


def Y(i, q): return y(n + q * 2 * n + i)


def kuramotos_f_bilayer():

    qs = [0, 1]
    for q in [0, 1]:
        for i in range(n):
            yield Y(i, q)

        for i in range(n):
            coupling_sum = sum(sin(X(j, q) - X(i, q))
                               for j in range(n) if A[q][i, j])
            yield (-Y(i, q) + omega[i, q] + g_intra * coupling_sum +
                   g_inter * sin(X(i, q) - X(i, int(not q)))) * inv_m
# -------------------------------------------------------------------


# def kuramotos_f():

#     for i in range(n):
#         yield y(i+n)

#     for i in range(n):
#         coupling_sum = sum(sin(y(j)-y(i))
#                            for j in range(n)
#                            if A[i, j])
#         yield (-y(i+n) + omega[i] + g * coupling_sum) * inv_m
# # -------------------------------------------------------------------


def make_compiled_file():

    I = jitcode(kuramotos_f_bilayer, n=n4, control_pars=[g_inter, g_intra])
    I.generate_f_C(chunk_size=100)
    I.compile_C(omp=OMP)
    I.save_compiled(overwrite=True, destination="data/jitced.so")
# -------------------------------------------------------------------


def simulate(simulation_time,
             transition_time,
             coupling_inter,
             coupling_intra,
             initial_state=None,
             step=1, ):

    I = jitcode(n=n4, module_location="data/jitced.so")
    I.set_parameters(coupling_inter, coupling_intra)
    I.set_integrator("dopri5")
    if initial_state is None:
        initial_state = uniform(-np.pi, np.pi, 2*n)
    I.set_initial_value(initial_state, 0.0)

    times = np.arange(0, int(simulation_time), step)
    trans_index = int(transition_time/step)
    n_steps = len(times) - trans_index

    phases = np.empty((n_steps, n))
    order = np.empty(n_steps)

    for i in range(len(times)):
        phases_i = (I.integrate(times[i]) % (2*np.pi))
        if i >= trans_index:
            phases[i - trans_index, :] = phases_i[:n]
            order[i-trans_index] = order_parameter(phases_i[:n])

    return times[trans_index:], phases, order, phases_i
# -------------------------------------------------------------------


def order_parameter(phases):
    # calculate the order parameter

    n = phases.shape
    r = abs(sum(np.exp(1j * phases))) / n
    return r
# -------------------------------------------------------------------


def plot_order(t, r, ax=None, **kwargs):

    savefig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        savefig = True

    ax.plot(t, r, **kwargs)
    # ax.set_xlabel("times")
    ax.set_ylabel("r(t)")
    ax.set_ylim(0, 1.1)

    if savefig:
        plt.savefig("data/r.png", dpi=150)
        plt.close()
# -------------------------------------------------------------------


def plot_H_loop(filename, **kwargs):
    fig, ax = plt.subplots(1)

    data = np.load(filename)

    ax.plot(data['g'], data['Rf'], lw=1, label="F", color='r', **kwargs)
    ax.plot(data['g'][::-1], data['Rb'], lw=1, label="B", color='b', **kwargs)
    ax.set_xlabel("coupling")
    ax.set_ylabel("R")
    ax.legend()
    plt.savefig("data/H_loop.png", dpi=150)


if __name__ == "__main__":

    # ---------------------------------------------------------------
    def H_loop():

 
        if not os.path.exists("data/jitced.so"):
            make_compiled_file()

        initial_state = uniform(-np.pi, np.pi, n4)

        directionList = ["forward", "backward"]
        R = {}

        for dir in directionList:
            if dir == "forward":
                coupls = copy(couplings)
            else:
                coupls = copy(couplings[::-1])
            R0 = np.empty(len(couplings))
            for i in range(len(couplings)):
                print("{:s}, {:10.6f}".format(dir, coupls[i]*k_ave))
                _, _, order, phases_last = simulate(simulation_time,
                                                    transition_time,
                                                    coupls[i],
                                                    initial_state)
                initial_state = copy(phases_last)

                R0[i] = np.average(order)
            R[dir] = R0

        np.savez("data/data",
                 Rf=R["forward"],
                 Rb=R['backward'],
                 g=couplings*k_ave)
        plot_H_loop("data/data.npz")



    n = 10
    n4 = 4 * n
    m = 1.0
    OMP = False
    inv_m = 1.0 / m
    g_inter = Symbol("g_inter")
    g_intra = Symbol("g_intra")
    A = []
    for q in [0, 1]:
        Graph = nx.gnp_random_graph(n, p=1, seed=1)
        _A = nx.to_numpy_array(Graph, dtype=int)
        A.append(_A)

    omega = uniform(-1, 1, size=(n, 2))
    omega.sort(axis=0)

    Dx = 1.0
    coupling = 1.0
    ave_degree0 = 9
    ave_degree1 = 9
    simulation_time = 500
    transition_time = 100


    g_intra_ = [coupling/(ave_degree0 + Dx), coupling/(ave_degree1 + Dx)]
    g_inter_ = [Dx/(ave_degree0 + Dx), Dx/(ave_degree1 + Dx)]

    H_loop()
    # plot_H_loop("data/data.npz", marker="o")
