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


def kuramotos_f():

    for i in range(n):
        yield y(i+n)

    for i in range(n):
        coupling_sum = sum(sin(y(j)-y(i))
                           for j in range(n)
                           if A[i, j])
        yield (-y(i+n) + omega[i] + g * coupling_sum) * inv_m
# -------------------------------------------------------------------


def make_compiled_file():

    I = jitcode(kuramotos_f, n=2*n, control_pars=[g])
    I.generate_f_C(chunk_size=100)
    I.compile_C(omp=OMP)
    I.save_compiled(overwrite=True, destination="data/jitced.so")
# -------------------------------------------------------------------


def simulate(simulation_time,
             transition_time,
             coupling,
             initial_state=None,
             step=1, ):

    I = jitcode(n=2*n, module_location="data/jitced.so")
    I.set_parameters(coupling)
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

        couplings = np.linspace(0.5, 2, 26) / k_ave
        simulation_time = 500
        transition_time = 100

        if not os.path.exists("data/jitced.so"):
            make_compiled_file()

        initial_state = uniform(-np.pi, np.pi, 2*n)

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

    n = 100
    k_ave = 12
    m = 1.0
    OMP = True
    inv_m = 1.0 / m
    g = Symbol("g")
    Graph = nx.gnp_random_graph(n, p=0.12, seed=1)
    A = nx.to_numpy_array(Graph, dtype=int)
    omega = uniform(-1, 1, n)
    omega.sort()

    # H_loop()
    # plot_H_loop("data/data.npz", marker="o")
