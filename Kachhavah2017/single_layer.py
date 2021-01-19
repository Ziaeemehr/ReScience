"""
single layer Kuramoto model second order.
"""

import os
import numpy as np
import pylab as plt
import networkx as nx
from symengine import sin, Symbol
from jitcode import jitcode, y
from numpy.random import choice
from numpy.random import uniform
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(2)

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
    I.generate_f_C()
    I.compile_C()
    I.save_compiled(overwrite=True, destination="data/jitced.so")
# -------------------------------------------------------------------


def simulate(simulation_time, transition_time, coupling, step=1):

    I = jitcode(n=2*n, module_location="data/jitced.so")
    I.set_parameters(coupling)
    I.set_integrator("dopri5")
    initial_state = uniform(-np.pi, np.pi, 2*n)
    I.set_initial_value(initial_state, 0.0)

    times = np.arange(0, int(simulation_time), step)
    trans_index = int(transition_time/step)
    n_steps = len(times) - trans_index

    phases = np.empty((n_steps, n))
    order = np.empty(n_steps)

    for i in range(len(times)):
        phases_i = (I.integrate(times[i]) % (2*np.pi))[:n]
        if i >= trans_index:
            phases[i - trans_index, :] = phases_i
            order[i-trans_index] = order_parameter(phases_i)

    return times[trans_index:], phases, order
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


if __name__ == "__main__":

    n = 10
    m = 1.0
    k_ave = 9
    inv_m = 1.0 / m
    g = Symbol("g")
    Graph = nx.complete_graph(n)
    A = nx.to_numpy_array(Graph, dtype=int)
    omega = uniform(-1, 1, n)
    omega.sort()
    
    def figure_r():
        c = 2
        coupling = c / k_ave
        simulation_time = 100
        transition_time = 0

        if not os.path.exists("data/jitced.so"):
            make_compiled_file()

        times, phases, order = simulate(simulation_time,
                                        transition_time,
                                        coupling)

        fig, ax = plt.subplots(2, sharex=True)
        plot_order(times, order, ax=ax[0])
        plot_phases(phases, [0, times[-1], 0, n], ax[1])
        ax[0].set_xlim(0, times[-1])
        plt.savefig("data/fig1.png", dpi=150)


    def figure_R():
        c = 2
        couplings = np.linspace(0, c, 20) / k_ave
        simulation_time = 100
        transition_time = 20

        if not os.path.exists("data/jitced.so"):
            make_compiled_file()

        fig, ax = plt.subplots(1)

        R = np.empty(len(couplings))
        for i in range(len(couplings)):
            _, _, order = simulate(simulation_time,
                                            transition_time,
                                            couplings[i])
            R[i] = np.average(order)

        ax.plot(couplings, R, lw=1, label="R")
        ax.set_xlabel("coupling")
        ax.set_ylabel("R")
        plt.savefig("data/R.png", dpi=150)

    

    figure_r()
    figure_R()