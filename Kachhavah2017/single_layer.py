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

    # dxdt = list([0.]*2*n)
    # for i in range(n):
    #     dxdt[i] = y(i+n)

    # for i in range(n):
    #     coupling_sum = sum(sin(y(j)-y(i))
    #                        for j in range(n)
    #                        if A[i, j])
    #     dxdt[i+n] = (-y(i+n) + omega[i] + g * coupling_sum) * inv_m

    # I = jitcode(dxdt, n=2*n, control_pars=[g])
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
        phases_i = I.integrate(times[i])
        if i >= trans_index:
            phases[i - trans_index, :] = phases_i[:n] % (2*np.pi)
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


def plot_phases(phases, extent, ax=None):

    savefig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        savefig = True
    im = ax.imshow(phases.T,
                   origin="lower",
                   extent=extent,
                   aspect="auto",
                   cmap="afmhot")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    ax.set_xlabel("times")
    ax.set_ylabel("node index")

    if savefig:
        plt.savefig("data/kuramoto.png", dpi=150)
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

    def figure_r():

        global n, k_ave, inv_m, Graph, A, omega, OMP
        n = 10
        k_ave = 9
        inv_m = 1.0
        OMP = False

        c = 0.8
        coupling = c / k_ave
        simulation_time = 1000
        transition_time = 0

        Graph = nx.gnp_random_graph(n, p=1, seed=1)
        A = nx.to_numpy_array(Graph, dtype=int)
        omega = uniform(-1, 1, n)
        omega.sort()

        if not os.path.exists("data/jitced.so"):
            make_compiled_file()

        times, phases, order, _ = simulate(simulation_time,
                                           transition_time,
                                           coupling)

        fig, ax = plt.subplots(2, sharex=True)
        plot_order(times, order, ax=ax[0])
        plot_phases(phases, [0, times[-1], 0, n], ax[1])
        ax[0].set_xlim(0, times[-1])
        plt.savefig("data/fig1.png", dpi=150)
        
        print(np.average(order))

    def figure_R():

        global n, k_ave, inv_m, Graph, A, omega, OMP

        n = 10
        k_ave = 9
        inv_m = 1.0
        OMP = False

        c = 0.8
        
        Graph = nx.gnp_random_graph(n, p=1, seed=1)
        A = nx.to_numpy_array(Graph, dtype=int)
        omega = uniform(-1, 1, n)
        omega.sort()

        c = 2
        couplings = np.linspace(0, c, 20) / k_ave
        simulation_time = 100
        transition_time = 20

        if not os.path.exists("data/jitced.so"):
            make_compiled_file()

        ـ, ax = plt.subplots(1)

        R = np.empty(len(couplings))
        for i in range(len(couplings)):
            _, _, order, _ = simulate(simulation_time,
                                      transition_time,
                                      couplings[i])
            R[i] = np.average(order)

        ax.plot(couplings, R, lw=1, label="R")
        ax.set_xlabel("coupling")
        ax.set_ylabel("R")
        plt.savefig("data/R.png", dpi=150)

    # ---------------------------------------------------------------
    def H_loop():

        global n, k_ave, inv_m, Graph, A, omega, OMP

        OMP = True
        n = 100
        m = 1.0
        k_ave = 12
        inv_m = 1.0 / m

        Graph = nx.gnp_random_graph(n, p=0.12, seed=1)
        A = nx.to_numpy_array(Graph, dtype=int)
        omega = uniform(-1, 1, n)
        omega.sort()

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
        plot_H_loop("data/data.npz", marker="o")

    global n, k_ave, inv_m, Graph, A, omega, OMP

    g = Symbol("g")

    # figure_r()
    # figure_R()
    H_loop()

    # plot_H_loop("data/data.npz", marker="o")
