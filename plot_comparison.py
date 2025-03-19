#!/usr/bin/env python
"""Plot convergence and transect comparisons of test runs

Note:  This assumes that AMR is not being used, i.e. there is a single grid.  To
get that to work we would have to do more to access more than one grid.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import clawpack.pyclaw.solution as sol
import clawpack.geoclaw.surge.plot as surgeplot

from setplot import exact_solution

def exact_vorticity(x, y, t):
    M = .5
    g = 1.0
    c1 = 0.04
    c2 = 0.02
    alpha = np.pi / 6.
    x0 = -20.
    y0 = -10.

    f = lambda x,y,t: -c2*((x-x0-M*t*np.cos(alpha))**2+(y-y0-M*t*np.sin(alpha))**2)
    f_x = lambda x, y, t: -2 * c2 * (x - x0 - M * t * np.cos(alpha))
    f_y = lambda x, y, t: -2 * c2 * (y - y0 - M * t * np.sin(alpha))

    u_y = lambda x, y, t:  c1 * np.exp(f(x, y, t)) * (1 + (y - y0 - M * t * np.sin(alpha)) * f_y(x, y, t))
    v_x = lambda x, y, t: -c1 * np.exp(f(x, y, t)) * (1 + (x - x0 - M * t * np.cos(alpha)) * f_x(x, y, t))

    return v_x(x, y, t) - u_y(x, y, t)

def plot_convergence(base_path, ax, ord=1, color='red', verbose=True):

    field = 0
    frame = 10
    title = r"Convergence"
    # N = np.array([2**n for n in range(6, 12)])
    N = np.array([50, 100, 200, 400, 800, 1600])
    markers = ['.', 'x', '+', 'o']
    
    h_error = []
    hu_error = []
    hv_error = []
    vorticity_error = []
    for num_cells in N:
        # Load computed solution
        path = os.path.join(base_path, f"n{str(num_cells).zfill(4)}_output")
        if verbose:
            print(f" -> Loading from {path}")
        solution = sol.Solution(path=path, frame=frame, file_format='binary')
        # solution = sol.Solution(path=path, frame=frame)
        q = solution.states[0].q
        X, Y = solution.states[0].grid.p_centers
        delta = np.array(solution.states[0].delta)
        t = solution.states[0].t
        u = surgeplot.extract_velocity(q[0, :, :], q[1, :, :])
        v = surgeplot.extract_velocity(q[0, :, :], q[2, :, :])

        # Compute exact solution
        h_exact, u_exact, v_exact = exact_solution(X, Y, t)

        # Compute vorticity

        omega = surgeplot.extract_vorticity(delta, u, v)
        # omega_exact = surgeplot.extract_vorticity(delta, u_exact, v_exact)
        omega_exact = exact_vorticity(X, Y, t)

        # Compute errors
        h_error.append(np.linalg.norm(h_exact - q[0, :, :], ord=ord) * delta.prod())
        hu_error.append(np.linalg.norm(h_exact * u_exact - q[1, :, :], ord=ord) * delta.prod())
        hv_error.append(np.linalg.norm(h_exact * v_exact - q[2, :, :], ord=ord) * delta.prod())
        vorticity_error.append(np.linalg.norm(omega_exact - omega) * delta.prod())

    # Compute convergence rate
    rates = [-np.polyfit(np.log(N), np.log(h_error), 1)[0].round(4),
             -np.polyfit(np.log(N), np.log(hu_error), 1)[0].round(4),
             -np.polyfit(np.log(N), np.log(hv_error), 1)[0].round(4),
             -np.polyfit(np.log(N), np.log(vorticity_error), 1)[0].round(4)
            ]
    if verbose:
        print(f"N     h          hu         hv         vorticity ")
        print(f"----- ---------- ---------- ---------- ----------")
        for (i, num_cells) in enumerate(N):
            print("{0: >5} {1: >10.8f} {2: >10.8f} {3: >10.8f} {4: >10.8f}".format(num_cells, 
                                                  h_error[i].round(8), 
                                                  hu_error[i].round(8), 
                                                  hv_error[i].round(8),
                                                  vorticity_error[i].round(8)))
        print(f"----- ---------- ---------- ---------- ----------")
        print("Rates {0: >10.8f} {1: >10.8f} {2: >10.8f} {3: >10.8f}".format(*rates))
        print()
    
    # Plot errors
    ax.loglog(N, h_error, color=color, marker=markers[0], markersize=5, 
                          label=f"h: {rates[0]}")
    ax.loglog(N, hu_error, color=color, marker=markers[1], markersize=5, 
                           label=f"hu: {rates[1]}")
    ax.loglog(N, hv_error, color=color, marker=markers[2], markersize=5, 
                           label=f"hv: {rates[2]}")
    ax.plot(N, vorticity_error, color=color, marker=markers[3], 
                                markersize=5, 
                                label=r"$\omega$: {}".format(rates[3]))
    
    ax.set_title(f"Convergence")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$q[{}]$".format(field))
    ax.legend()


if __name__ == '__main__':
    base_path = os.path.expandvars(os.path.join("${DATA_PATH}", 
                                                "vortex_example"))
    # Convergence
    fig, ax = plt.subplots()
    print(f"Plotting GeoClaw RS")
    plot_convergence(os.path.join(base_path, "geoclaw"), ax, color='red')
    print(f"Plotting Simple RS")
    plot_convergence(os.path.join(base_path, "simple"), ax, color='blue')

    fig.savefig("convergence.pdf")
