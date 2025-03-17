#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt

# import clawpack.visclaw.colormaps as colormap
# import clawpack.visclaw.gaugetools as gaugetools
import clawpack.clawutil.data as clawutil
import clawpack.amrclaw.data as amrclaw
import clawpack.geoclaw.data as geodata
import clawpack.visclaw.geoplot as geoplot
import clawpack.geoclaw.surge.plot as surgeplot

try:
    from setplotfg import setplotfg
except:
    setplotfg = None

def exact_solution(x, y, t):
    M = .5
    g = 1.0
    c1 = 0.04
    c2 = 0.02
    alpha = np.pi / 6.
    x0 = -20.
    y0 = -10.

    f = lambda x,y,t: -c2*((x-x0-M*t*np.cos(alpha))**2+(y-y0-M*t*np.sin(alpha))**2)
    h = lambda x,y,t: 1.-c1**2/(4.*c2*g)*np.exp(2.*f(x,y,t))
    u = lambda x,y,t: M*np.cos(alpha)+c1*(y-y0-M*t*np.sin(alpha))*np.exp(f(x,y,t))
    v = lambda x,y,t: M*np.sin(alpha)-c1*(x-x0-M*t*np.cos(alpha))*np.exp(f(x,y,t))

    return h(x, y, t), u(x, y, t), v(x, y, t)

def extract_vorticity(delta, u, v):
    omega = np.zeros(u.shape)
    omega[1:-1, 1:-1] += (v[2:,1:-1] - v[:-2,1:-1]) / (2.0 * delta[0])
    omega[1:-1, 1:-1] -= (u[1:-1,2:] - u[1:-1,:-2]) / (2.0 * delta[1])
    return omega

def water_vorticity(cd):
    delta = [cd.x[1, 0] - cd.x[0, 0], cd.y[0, 1] - cd.y[0, 0]]
    return extract_vorticity(delta, surgeplot.water_u(cd), surgeplot.water_v(cd))

def extract_eta_error(cd):
    eta = surgeplot.extract_eta(cd.q[0, :, :], cd.q[3, :, :])
    h, u, v = exact_solution(cd.x, cd.y, cd.t)
    return eta - (h - 1.0)

def eta_error(cd, order=1):
    delta = (cd.x[1, 0] - cd.x[0, 0]) * (cd.y[0, 1] - cd.y[0, 0])
    return np.linalg.norm(extract_eta_error(cd), ord=order) * delta

def extract_speed_error(cd):
    speed = surgeplot.water_speed(cd)
    h, u, v = exact_solution(cd.x, cd.y, cd.t)
    return speed - np.sqrt(u**2 + v**2)

def speed_error(cd, order=1):
    delta = (cd.x[1, 0] - cd.x[0, 0]) * (cd.y[0, 1] - cd.y[0, 0])
    return np.linalg.norm(extract_speed_error(cd), ord=order) * delta

# def error_vorticity(cd, order=1):
#     omega = water_vorticity(cd)
#     h, u, v = exact_solution(cd.x, cd.y, cd.t)
#     delta = [cd.x[1, 0] - cd.x[0, 0], cd.y[0, 1] - cd.y[0, 0]]
#     exact_omega = extract_vorticity(delta, u, v)
#     return np.linalg.norm(omega - exact_omega, ord=order) * delta[0] * delta[1]

def add_vorticity(plotaxes, plot_type="pcolor", bounds=None, contours=None, shrink=1.0):
    """Add vorticity plot to plotaxes"""

    vorticity_cmap = plt.get_cmap('PRGn')

    if plot_type == 'pcolor' or plot_type == 'imshow':
        plotitem = plotaxes.new_plotitem(name='surface', plot_type='2d_pcolor')
        plotitem.plot_var = water_vorticity

        if bounds is not None:
            if bounds[0] == 0.0:
                plotitem.pcolor_cmap = plt.get_cmap('OrRd')
            else:
                plotitem.pcolor_cmap = vorticity_cmap
            plotitem.pcolor_cmin = bounds[0]
            plotitem.pcolor_cmax = bounds[1]
        plotitem.add_colorbar = True
        plotitem.colorbar_shrink = shrink
        plotitem.colorbar_label = "Vorticity (1/s)"
        plotitem.amr_celledges_show = [0] * 10
        plotitem.amr_patchedges_show = [1, 1, 1, 0, 0, 0, 0]

def setplot(plotdata=None):
    """"""

    if plotdata is None:
        from clawpack.visclaw.data import ClawPlotData
        plotdata = ClawPlotData()

    # clear any old figures,axes,items data
    plotdata.clearfigures()
    plotdata.format = 'ascii'

    # Load data from output
    clawdata = clawutil.ClawInputData(2)
    clawdata.read(os.path.join(plotdata.outdir, 'claw.data'))
    physics = geodata.GeoClawData()
    physics.read(os.path.join(plotdata.outdir, 'geoclaw.data'))
    surge_data = geodata.SurgeData()
    surge_data.read(os.path.join(plotdata.outdir, 'surge.data'))
    friction_data = geodata.FrictionData()
    friction_data.read(os.path.join(plotdata.outdir, 'friction.data'))

    # Color limits
    surface_limits = [-0.02, 0.02]
    speed_limits = [0.0, 0.17]
    vorticity_limits = [-0.08, 0.08]
    eta_error_limits = [-0.02, 0.02]
    speed_error_limits = [-0.07, 0.07]

    # ==========================================================================
    #   Plot specifications
    # ==========================================================================
    # Surface Figure
    plotfigure = plotdata.new_plotfigure(name="Surface")
    plotfigure.kwargs = {"figsize": (6.4, 4.8)}
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Surface"
    plotaxes.xlimits = (clawdata.lower[0], clawdata.upper[0])
    plotaxes.ylimits = (clawdata.lower[1], clawdata.upper[1])

    surgeplot.add_surface_elevation(plotaxes, bounds=surface_limits)
    surgeplot.add_land(plotaxes, bounds=[0.0, 20.0])
    plotaxes.plotitem_dict['surface'].amr_patchedges_show = [0] * 10

    # Speed Figure
    plotfigure = plotdata.new_plotfigure(name="Currents")
    plotfigure.kwargs = {"figsize": (6.4, 4.8)}
    plotfigure.show = False
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Currents"
    plotaxes.xlimits = (clawdata.lower[0], clawdata.upper[0])
    plotaxes.ylimits = (clawdata.lower[1], clawdata.upper[1])

    # Vorticity
    plotfigure = plotdata.new_plotfigure(name="Vorticity")
    plotfigure.kwargs = {}
    plotfigure.show = True
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Vorticity"
    plotaxes.xlimits = (clawdata.lower[0], clawdata.upper[0])
    plotaxes.ylimits = (clawdata.lower[1], clawdata.upper[1])

    add_vorticity(plotaxes, bounds=vorticity_limits)
    plotaxes.plotitem_dict['surface'].amr_patchedges_show = [0] * 10

    # ========================================================================
    # Error Plots
    def add_eta_error_title(cd):
        plt.gca().set_title(r"$\eta$ Error (t = {}), ".format(cd.t) + 
                            r"$|| E ||_{\ell_1} = $" +
                            f"{eta_error(cd).round(6)}")

    def add_speed_error_title(cd):
        plt.gca().set_title(r"Speed Error (t = {}), ".format(cd.t) + 
                            r"$|| E ||_{\ell_1} = $" +
                            f"{speed_error(cd).round(6)}")

    # Surface
    plotfigure = plotdata.new_plotfigure(name="Surface Error")
    plotfigure.kwargs = {"figsize": (6.4, 4.8)}
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = r"$\eta$ Error"
    plotaxes.xlimits = (clawdata.lower[0], clawdata.upper[0])
    plotaxes.ylimits = (clawdata.lower[1], clawdata.upper[1])
    plotaxes.afteraxes = add_eta_error_title
    
    plotitem = plotaxes.new_plotitem(name='surface error', plot_type='2d_pcolor')
    plotitem.plot_var = extract_eta_error
    plotitem.pcolor_cmax = eta_error_limits[1]
    plotitem.pcolor_cmin = eta_error_limits[0]
    plotitem.pcolor_cmap = plt.get_cmap("RdBu")
    plotitem.add_colorbar = True
    plotitem.colorbar_label = r"$\eta$ Error"

    # Speed
    plotfigure = plotdata.new_plotfigure(name="Speed Error")
    plotfigure.kwargs = {"figsize": (6.4, 4.8)}
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = r"Speed Error"
    plotaxes.xlimits = (clawdata.lower[0], clawdata.upper[0])
    plotaxes.ylimits = (clawdata.lower[1], clawdata.upper[1])
    plotaxes.afteraxes = add_speed_error_title
    
    plotitem = plotaxes.new_plotitem(name='speed error', plot_type='2d_pcolor')
    plotitem.plot_var = extract_speed_error
    plotitem.pcolor_cmax = speed_error_limits[1]
    plotitem.pcolor_cmin = speed_error_limits[0]
    plotitem.pcolor_cmap = plt.get_cmap("RdBu")
    plotitem.add_colorbar = True
    plotitem.colorbar_label = r"Speed Error"


    # ========================================================================
    # Transects
    def compute_max(current_data, field=3, title=r"Field {} - $\max = {}$"):
        ax = plt.gca()
        max_value = np.max(np.abs(current_data.q[field, :, :]))
        ax.set_title(title.format(field, max_value))

    def transect(current_data, field=3, y0=0.0):
        y = current_data.y
        dy = current_data.dy
        index = np.where(abs(y - y0) <= dy / 2.0)[1][0]
        if field < 0:
            # Extract velocity
            h = current_data.q[0, :, index]
            hu = current_data.q[abs(field), :, index]
            u = np.where(h > 1e-3, hu / h, np.zeros(h.shape))
            return current_data.x[:, index], u
        elif field == 4:
            # Plot topography
            h = current_data.q[0, :, index]
            eta = current_data.q[3, :, index]
            return current_data.x[:, index], eta - h
        else:
            return current_data.x[:, index], current_data.q[field, :, index]

    # === Surface/Topography ===
    # plotfigure = plotdata.new_plotfigure(name="Surface Transect")
    # plotfigure.show = True
    # plotaxes = plotfigure.new_plotaxes()
    # plotaxes.title = "Surface Transect"
    # plotaxes.xlabel = "x (m)"
    # plotaxes.ylabel = r"$\eta$"
    # plotaxes.xlimits = [clawdata.lower[0], clawdata.upper[0]]
    # # plotaxes.ylimits = [-1.1, 0.1]
    # plotaxes.grid = True
    # plotaxes.afteraxes = lambda cd: compute_max(cd)

    # plotitem = plotaxes.new_plotitem(plot_type="1d_from_2d_data")
    # plotitem.map_2d_to_1d = transect
    # plotitem.plotstyle = 'ko-'
    # plotitem.kwargs = {"markersize": 3}

    # plotitem = plotaxes.new_plotitem(plot_type="1d_from_2d_data")
    # plotitem.map_2d_to_1d = lambda cd:transect(cd, field=4)
    # plotitem.plotstyle = 'g'
    # plotitem.kwargs = {"markersize": 3}

    # === Depth ===
    plotfigure = plotdata.new_plotfigure(name="Depth Transect")
    plotfigure.show = True
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Depth Transect"
    plotaxes.xlabel = "x (m)"
    plotaxes.ylabel = r"$h$"
    plotaxes.xlimits = [clawdata.lower[0], clawdata.upper[0]]
    plotaxes.ylimits = surface_limits
    plotaxes.grid = True
    plotaxes.afteraxes = lambda cd: compute_max(cd, field=0)

    plotitem = plotaxes.new_plotitem(plot_type="1d_from_2d_data")
    plotitem.map_2d_to_1d = lambda cd: transect(cd, field=0)
    plotitem.plotstyle = 'ko-'
    plotitem.kwargs = {"markersize": 3}

    # === Momentum/Velocity ===
    plotfigure = plotdata.new_plotfigure(name="Momentum Transect")
    plotfigure.show = True
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = r"Momentum Transect - $\max |hu|$"
    plotaxes.xlabel = "x (m)"
    plotaxes.ylabel = r"$hu$"
    plotaxes.xlimits = [clawdata.lower[0], clawdata.upper[0]]
    # plotaxes.ylimits = [0.0, 0.35]
    plotaxes.grid = True
    plotaxes.afteraxes = lambda cd: compute_max(cd, field=1)

    plotitem = plotaxes.new_plotitem(plot_type="1d_from_2d_data")
    plotitem.map_2d_to_1d = lambda cd: transect(cd, field=1)
    plotitem.plotstyle = 'ko-'
    plotitem.kwargs = {"markersize": 3}

    plotitem = plotaxes.new_plotitem(plot_type="1d_from_2d_data")
    plotitem.map_2d_to_1d = lambda cd: transect(cd, field=-1)
    plotitem.plotstyle = 'bx-'
    plotitem.kwargs = {"markersize": 3}

    # # ========================================================================
    # #  Figures for gauges
    # # ==========================================================================
    # plotfigure = plotdata.new_plotfigure(name='Gauge Surfaces', figno=300,
    #                                      type='each_gauge')
    # plotfigure.show = True
    # plotfigure.clf_each_gauge = True

    # plotaxes = plotfigure.new_plotaxes()
    # # plotaxes.time_scale = 1 / (24 * 60**2)
    # plotaxes.grid = True
    # plotaxes.xlimits = 'auto'
    # plotaxes.ylimits = 'auto'
    # plotaxes.title = "Surface"
    # plotaxes.ylabel = "Surface (m)"
    # plotaxes.time_label = "t (s)"

    # plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    # plotitem.plot_var = surgeplot.gauge_surface

    # #  Gauge Location Plot
    # def gauge_location_afteraxes(cd):
    #     plt.subplots_adjust(left=0.12, bottom=0.06, right=0.97, top=0.97)
    #     surge_afteraxes(cd)
    #     gaugetools.plot_gauge_locations(cd.plotdata, gaugenos='all',
    #                                     format_string='ko', add_labels=True)

    # #  Gauge Location Plot
    # # def gauge_location_afteraxes(cd):
    # #     plt.subplots_adjust(left=0.12, bottom=0.06, right=0.97, top=0.97)
    # #     surge_afteraxes(cd)
    # #     gaugetools.plot_gauge_locations(cd.plotdata, gaugenos='all',
    # #                                     format_string='bx', add_labels=False)
    # #     gaugetools.plot_gauge_locations(cd.plotdata, gaugenos=[0, 11, 22, 10, 21, 32],
    # #                                     format_string='ko', add_labels=True)

    # plotfigure = plotdata.new_plotfigure(name="Gauge Locations")
    # plotfigure.show = False

    # # Set up for axes in this figure:
    # plotaxes = plotfigure.new_plotaxes()
    # plotaxes.title = 'Gauge Locations'
    # plotaxes.scaled = True
    # plotaxes.xlimits = (clawdata.lower[0], clawdata.upper[0])
    # plotaxes.ylimits = (clawdata.lower[1], clawdata.upper[1])
    # plotaxes.afteraxes = gauge_location_afteraxes
    # surgeplot.add_surface_elevation(plotaxes, bounds=surface_limits)
    # surgeplot.add_land(plotaxes, bounds=[0.0, 20.0])
    # plotaxes.plotitem_dict['surface'].amr_patchedges_show = [0] * 10
    # plotaxes.plotitem_dict['land'].amr_patchedges_show = [0] * 10

    # -----------------------------------------
    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_gaugenos = 'all'        # list of gauges to print
    # plotdata.print_gaugenos = [10, 21, 32]   # list of gauges to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?
    plotdata.parallel = True                 # parallel plotting

    return plotdata
