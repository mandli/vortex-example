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

from setplot import exact_solution, exact_vorticity

def plot_convergence(base_path, ax, ord=1, frame=10, verbose=False):

    # N = np.array([2**n for n in range(6, 12)])
    N = np.array([50, 100, 200, 400, 800, 1600])
    colors = {"geoclaw": "red", "simple": "blue"}
    markers = ['.', 'x', '+', 'o']
    out_str = ""

    for rp_type in colors.keys():    
        h_error = []
        hu_error = []
        hv_error = []
        vorticity_error = []
        if verbose:
            print(f"{rp_type}")
        for num_cells in N:
            # Load computed solution
            path = os.path.join(base_path, rp_type, 
                                f"n{str(num_cells).zfill(4)}_output")
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
        out_str += f"{rp_type}\n"
        out_str += "N     h          hu         hv         vorticity \n"
        out_str += "----- ---------- ---------- ---------- ----------\n"
        for (i, num_cells) in enumerate(N):
            out_str += "{0: >5} ".format(num_cells)
            out_str += "{0: >10.8f} ".format(h_error[i].round(8))
            out_str += "{0: >10.8f} ".format(hu_error[i].round(8))
            out_str += "{0: >10.8f} ".format(hv_error[i].round(8))
            out_str += "{0: >10.8f}\n".format(vorticity_error[i].round(8))
        out_str += f"----- ---------- ---------- ---------- ----------\n"
        out_str += "Rates {0: >10.8f} {1: >10.8f} {2: >10.8f} {3: >10.8f}\n\n".format(*rates)
        if verbose:
            print(out_str)
        
        # Plot errors
        ax.loglog(N, h_error, color=colors[rp_type], marker=markers[0], markersize=5, 
                              label=f"h: {rates[0]}")
        ax.loglog(N, hu_error, color=colors[rp_type], marker=markers[1], markersize=5, 
                               label=f"hu: {rates[1]}")
        ax.loglog(N, hv_error, color=colors[rp_type], marker=markers[2], markersize=5, 
                               label=f"hv: {rates[2]}")
        ax.plot(N, vorticity_error, color=colors[rp_type], marker=markers[3], 
                                    markersize=5, 
                                    label=r"$\omega$: {}".format(rates[3]))
    

    with open('convergence.txt', 'w') as convergence_file:
        convergence_file.write(out_str)

    ax.set_title(f"Convergence")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$q[{}]$".format(field))
    ax.legend()


if __name__ == '__main__':
    base_path = os.path.expandvars(os.path.join("${DATA_PATH}", 
                                                "vortex_example"))
    # Convergence
    fig, ax = plt.subplots()
    plot_convergence(base_path, ax)
    fig.savefig("convergence.pdf")
