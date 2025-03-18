#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt

import clawpack.pyclaw.solution as sol

from setplot import exact_solution

def load_transect(path, frame=10, y0=0.0):

def load_transect():
    # Assumes single grid
    return x, q[:, :, index]


def plot_convergence(base_path, ax, ord=1, verbose=True):

    field = 0
    frame = 10
    title = r"Convergence"
    N = np.array([2**n for n in range(6, 12)])

    colors = ['red', 'blue']
    markers = ['-.', '-x', '-+', '-o']
    for (i, rp_type) in enumerate(['geoclaw', 'simple']):
        if verbose:
            print(f"Loading {rp_type} runs...")
        h_error = []
        hu_error = []
        hv_error = []
        vorticity_error = []
        for num_cells in N:
            if verbose:
                print(f"  N = {num_cells}")

            # Load computed solution
            out_path = os.path.join(base_path, f"{rp_type}",
                             f"n{str(num_cells).zfill(4)}_RP{rp_type}_output")
            solution = sol.Solution(path=path, frame=frame)
            q = solution.states[0].q
            x = solution.states[0].grid.centers[0]
            y = solution.states[0].grid.centers[1]
            t = solution.states[0].t

            # Compute exact solution
            h, u, v = exact_solution(x, y, t)

            # Compute vorticity

            # Compute errors
            h_error.append(np.linalg.norm(h - q[0, :, :]), ord=ord)
            hu_error.append(np.linalg.norm(h * u - q[1, :, :]), ord=ord)
            hv_error.append(np.linalg.norm(h * v - q[2, :, :]), ord=ord)
            # vorticity_error.append(np.linalg.norm())

        # Plot errors
        ax.loglog(N, h_error, color=colors[i], marker=markers[0], markersize=5, 
                            label=f"h - {rp_type}")
        ax.loglog(N, hu_error, color=colors[i], marker=markers[1], markersize=5, 
                            label=f"hu - {rp_type}")
        ax.loglog(N, hu_error, color=colors[i], marker=markers[2], markersize=5, 
                            label=f"hv - {rp_type}")
        # ax.plot(N, vorticity_error, color=colors[i], marker=markers[3], 
        #                             markersize=5, label=f"hv - {rp_type}")
        if verbose:
            print(f"done with {rp_type}.")
    
    ax.set_title(f"Convergence")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$q[{}]$".format(field))
    ax.legend()



def plot_comparison(base_path, ax, verbose=True):

    field = 0
    frame = 10
    title = r"$\eta$ Transects"
    y0 = 0.0

    colors = ['red', 'blue']
    markers = ['x', '+', 'o', 's', 'd', '.']
    for (i, rp_type) in enumerate(['geoclaw', 'simple']):
        if verbose:
            print(f"Loading {rp_type} runs...")
        for (j, num_cells) in enumerate([2**n for n in range(6, 12)]):
            if verbose:
                print(f"  N = {num_cells}")
            out_path = os.path.join(base_path, f"{rp_type}",
                             f"n{str(num_cells).zfill(4)}_RP{rp_type}_output")
            solution = sol.Solution(path=out_path, frame=frame)
            x = solution.states[0].grid.centers[0]
            y = solution.states[0].grid.centers[1]
            dy = solution.states[0].grid.delta[1]
            index = np.where(abs(y - y0) <= dy / 2.0)[0][0]
            ax.plot(x, solution.states[0].q[field, :, index], 
                            color=colors[i], marker=markers[j], markersize=5, 
                            label=f"N={num_cells}")
        if verbose:
            print(f"done with {rp_type}.")
    
    ax.set_title(f"{title} comparison")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q[{}]$".format(field))
    ax.legend()


if __name__ == '__main__':
    base_path = os.path.expandvars(os.path.join("${DATA_PATH}", 
                                                "vortex"))
    # Convergence
    fig, ax = plt.subplots()
    plot_convergence(os.path.join(base_path, "geoclaw"), ax)
    plot_convergence(os.path.join(base_path, "simple"), ax)
    # fig.savefig("convergence.pdf")

    # Plot transect comparison
    fig, ax = plt.subplots()
    plot_comparison(base_path, ax)
    # fig.savefig("comparison.pdf")

    plt.show()
