"""
single layer Kuramoto model second order.
"""

import os
import numpy as np
import pylab as plt
import networkx as nx
from symengine import sin
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


def simulate(simulation_time, coupling, step=1):

    I = jitcode(n=2*n, module_location="data/jitced.so")
    I.set_parameters(coupling)
    I.set_integrator("dopri5")
    initial_state = uniform(-np.pi, np.pi, 2*n)
    I.set_initial_value(initial_state, 0.0)
    times = np.arange(0, int(simulation_time), step)
    n_steps = len(times)

    phases = np.empty((len(times), n))
    order = np.zeros(len(times))

    for i in range(n_steps):
        phases_i = (I.integrate(times[i]) % (2*np.pi))[:n]
        phases[i, :] = phases_i
        order[i] = order_parameter(phases_i)

    return times, phases
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
    c = 0.8
    k_ave = 9
    g = c/k_ave
    m = 1.0
    inv_m = 1.0 / m

    Graph = nx.complete_graph(n)
    A = nx.to_numpy_array(Graph, dtype=int)
    initial_state = uniform(0, 2*np.pi, 2 * n)
    omega = uniform(-1, 1, n)
    omega.sort()

    times = np.arange(0, 1001, 0.5)

    I = jitcode(kuramotos_f, n=2*n)
    I.set_integrator("dopri5", atol=1e-6, rtol=1e-5)
    I.set_initial_value(initial_state, time=0.0)

    phases = np.empty((len(times), n))
    order = np.zeros(len(times))

    print("running simulation ...")
    for i in range(len(times)):
        phases_i = (I.integrate(times[i]) % (2*np.pi))[:n]
        phases[i, :] = phases_i
        order[i] = order_parameter(phases_i)

    fig, ax = plt.subplots(2, sharex=True)
    plot_order(times, order, ax=ax[0])
    plot_phases(phases, [0, times[-1], 0, n], ax[1])

    ax[0].set_xlim(0, times[-1])

    plt.savefig("data/fig.png", dpi=150)
