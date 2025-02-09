# Earth-Mars transfer window design using Porkchop Plots
"""
Copyright (c) 2010-2023, Delft University of Technology
All rigths reserved
This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

"""

## Summary
"""
This example demonstrates the usage of the tudatpy `porkchop` module to determine an optimal launch window (departure and arrival date) for an Earth-Mars transfer mission.

By default, the porkchop module uses a Lambert arc to compute the $\Delta V$ required to depart from the departure body (Earth in this case) and be captured by the target body (in this case Mars).

Users can provide a custom function to calculate the $\Delta V$ required for any given transfer. This can be done by supplying a `callable` (a function) to the `porkchop` function via the argument

    function_to_calculate_delta_v

This opens the possibility to calculate the $\Delta V$ required for any transfer; potential applications include: low-thrust transfers, perturbed transfers with course correction manoeuvres, transfers making use of Gravity Assists, and more.
"""

## Import statements
"""

The required import statements are made here, starting with standard imports (`os`, `pickle` from the Python Standard Library), followed by tudatpy imports.
"""

# General imports
import os
import pickle

# Tudat imports
from tudatpy import constants
from tudatpy.interface import spice_interface
from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import environment_setup
from tudatpy.trajectory_design.porkchop import porkchop

''' 
Copyright (c) 2010-2023, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

# General imports
import numpy as np
from scipy import ndimage

# Plotting imports
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import matplotlib.dates as mdates

# Tudat imports
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion


plt.rcParams.update({'font.size': 14})

def plot_porkchop_of_single_field(
    # Data arguments
    departure_body: str,
    target_body: str,
    departure_epochs: np.ndarray, 
    arrival_epochs: np.ndarray, 
    delta_v: np.ndarray,
    # Plot arguments
    C3: bool = False,
    threshold: float = 10,
    upscale: bool = False,
    filled_contours: bool = True,
    plot_out_of_range_field: bool = True,
    plot_isochrones: bool = False,
    plot_global_optimum: bool = True,
    plot_minor_ticks: bool = True,
    number_of_levels: int = 10,
    line_width: float = 0.5,
    line_color: str = 'black',
    font_size: float = 12,
    label: str = False,
    # Figure arguments
    colorbar: bool = False,
    percent_margin: float = 5,
    fig: matplotlib.figure.Figure = None,
    ax:  matplotlib.axes._axes.Axes = None,
    figsize: tuple[int, int] = (8, 8),
    show: bool = True,
    save: bool = False,
    filename: str = 'porkchop.png',
    ) -> matplotlib.contour.QuadContourSet:
    """
    Create a ΔV/C3 porkchop mission design plot of single time window-ΔV field.

    Parameters
    ----------
    departure_body: str
        The name of the body from which the transfer is to be computed
    target_body: str
        The name of the body to which the transfer is to be computed
    departure_epochs: np.ndarray
        Discretized departure time window
    arrival_epochs: np.ndarray
        Discretized arrival time window
    delta_v: np.ndarray
        Array containing the ΔV of all coordinates of the grid of departure/arrival epochs
    C3: bool = False
        Whether to plot C3 (specific energy) instead of ΔV
    threshold: float = 10
        Upper threshold beyond which ΔV/C3 is not plotted. This is useful to mask regions of the plot where the ΔV/C3 is too high to be of interest.
    upscale: bool = False
        Whether to use interpolation to increase the resolution of the plot. This is not always reliable, and the detail generated cannot be relied upon for analysis. Its only purpose is aesthetic improvement.
    filled_contours: bool = True
        Whether to plot filled contours or else just the contour lines
    plot_out_of_range_field: bool = Tru
        Whether to plot the out-of-range field (ΔV/C3 above the threshold) in a different color
    plot_isochrones: bool = True
        Whether to plot the isochrone lines (constant time of flight) on the plot
    plot_global_optimum: bool = True
        Whether to mark the global optimum with a cross
    plot_minor_ticks: bool = True
        Whether to show minor ticks on the axes
    number_of_levels: int = 10
        The number of levels in the ΔV/C3 contour plot
    line_width: float = 0.5
        Width of the contour plot lines
    line_color: str = 'black'
        Color of the contour plot lines
    font_size: float = 7
        Font size of the contour plot labels
    label: str = False
        Label used to identify the contour plot in legends
    colorbar: bool = False
        Whether to plot a colorbar
    percent_margin: float = 5
        Empty margin between the axes of the plot and the plotted data
    fig: matplotlib.figure.Figure = None
        Figure on which to plot
    ax:  matplotlib.axes._axes.Axes = None
        Axis on which to plot
    figsize: tuple[int, int] = (8, 8)
        Size of the figure
    show: bool bool = True
        Whether to show the plot        
    save: bool = False
        Whether to save the plot
    filename: str = 'porkchop.png'
        The filename used for the saved plot

    Output
    ------
    contour_lines: matplotlib.contour.QuadContourSet
        The contour plot Matplotlib object
    """

    #--------------------------------------------------------------------
    #%% DATA PREPARATION
    #--------------------------------------------------------------------

    # Transpose ΔV array such as to show departure on the horizontal axis and arrival on the vertical axis
    delta_v = delta_v.T
    
    # Divide ΔV array to display results in km/s
    delta_v = delta_v / 1000

    # Interpolate data to improve plot resolution
    if upscale:
        delta_v = ndimage.zoom(delta_v, 10)
        departure_epochs = np.linspace(departure_epochs.min(), departure_epochs.max(), delta_v.shape[1])
        arrival_epochs   = np.linspace(arrival_epochs.min(), arrival_epochs.max(), delta_v.shape[0])

    # Transform ΔV to characteristic energy (C3) if needed
    field = delta_v**2 if C3 else delta_v

    # Mask field array to discard excessive values
    field_within_range = np.ma.array(field, mask=field >= threshold)
    if colorbar == True:
        field_within_range = np.ma.array(field_within_range, mask=field_within_range <= 6)
    
    # Calculate departure and arrival time spans
    departure_epoch_span = (departure_epochs.max() - departure_epochs.min())
    arrival_epoch_span   = (arrival_epochs.max() - arrival_epochs.min())

    #--------------------------------------------------------------------
    #%% PLOT
    #--------------------------------------------------------------------

    # Use monospaced font for all text
    plt.rcParams["font.family"] = plt.rcParams["font.monospace"][0]
    plt.rcParams["mathtext.default"] = 'tt'

    # Create axis and figure
    if fig is None and ax is None:

        fig, ax = plt.subplots(figsize=figsize)

        # Title
        fig.suptitle(
            f'{departure_body}-{target_body} Transfer {"C3-Launch" if C3 else "ΔV-Launch"}',
            y=0.97
        )

    # Layout
    plt.subplots_adjust(
        top=0.92,
        bottom=0.155,
        left=0.145,
        right=0.885,
        hspace=0.2,
        wspace=0.2
    )
    
    # Determine contour levels
    levels = np.logspace(np.log10(field_within_range.min()), np.log10(field_within_range.max()), number_of_levels)

    # Filled contour
    if filled_contours:
        # Define custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'TudatMap',
            ['#d1edff', '#0076C2'],
            N=256)
        contour = plt.contourf(
            departure_epochs,
            arrival_epochs,
            field_within_range,
            cmap=cmap,
            levels=levels,
            zorder=1.5)
    # Levels
    contour_lines = plt.contour(
        departure_epochs,
        arrival_epochs,
        field,
        colors=line_color,
        linewidths=line_width,
        levels=levels,
        zorder=1.5)
    plt.clabel(
        contour_lines,
        fontsize=font_size,
    )

    # Colorbar
    if colorbar:
        cax = fig.add_axes([
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height]
        )
        cbar = fig.colorbar(
            contour,
            cax=cax)
        cbar.ax.set_title(
            '$km^2/s^2$' if C3 else 'km/s',
             pad=12.5,
             x=2.5
        )
        cbar.ax.yaxis.set_ticks(levels)
        plt.sca(ax)

    # Global optimum
    if plot_global_optimum:
        coordinates = np.unravel_index(np.argmin(field_within_range), field_within_range.shape)
        contour = plt.scatter(
            departure_epochs[coordinates[1]],
            arrival_epochs[coordinates[0]],
            marker='+', s=100, color=line_color,
            zorder=1.5,
            label=f'{label+" " if label else ""}{"C3" if C3 else "ΔV"}$_{{\infty,min}}$')
    
    # Out-of-range field plot
    if plot_out_of_range_field:
        if filled_contours:
            plt.contourf(
                departure_epochs,
                arrival_epochs,
                np.log(field),
                cmap='Reds',
                zorder=1.25)
        else:
            plt.contour(
                departure_epochs,
                arrival_epochs,
                np.log(field),
                cmap='Reds',
                levels=np.logspace(np.log10(field_within_range.min()), np.log10(field_within_range.max()), number_of_levels),
                linewidths=line_width,
                zorder=1.25)

    # Plot isochrones
    if plot_isochrones:
        x, y = np.meshgrid(departure_epochs, arrival_epochs)
        isoc = (y - x)
        isochrones = plt.contour(
            x, y, (isoc - isoc.min()) / constants.JULIAN_DAY,
            colors='black', alpha=0.5,
            levels=10,
            linewidths=2,
            linestyles='--',
            zorder=3)
        plt.clabel(
            isochrones,
            colors='black',
            fontsize=font_size + 3,
        )

    # Axis labels
    ax.set_xlabel(
        'Departure date',
        fontsize=font_size+3,
        weight='bold'
    )
    ax.set_ylabel(
        'Arrival date',
        fontsize=font_size+3,
        weight='bold'
    )

    # Axis limits
    ax.set_xlim([
        departure_epochs.min() - departure_epoch_span * percent_margin / 2 / 100,
        departure_epochs.max() + departure_epoch_span * percent_margin / 2 / 100,
    ])
    ax.set_ylim([
        arrival_epochs.min() - arrival_epoch_span * percent_margin / 2 / 100,
        arrival_epochs.max() + arrival_epoch_span * percent_margin / 2 / 100,
    ])

    # Tick labels
    dt = 1000
    tick_formatter = lambda epoch: time_conversion.date_time_from_epoch(epoch).iso_string()[:10]
    # X axis
    nx = int(np.floor(
        departure_epoch_span / (dt * constants.JULIAN_DAY))
    )

    x_ticks = np.linspace(departure_epochs.min(), departure_epochs.max(), nx)
    ax.xaxis.set_ticks(
        x_ticks,
        [f'{tick_formatter(t)}' for t in x_ticks],
        rotation=45
    )
    # Y axis
    ny = int(np.floor(
        arrival_epoch_span / (dt * constants.JULIAN_DAY))
    )
    y_ticks = np.linspace(arrival_epochs.min(), arrival_epochs.max(), ny)
    ax.yaxis.set_ticks(
        y_ticks,
        [f'{tick_formatter(t)}' for t in y_ticks]
    )

    # Grid
    plt.grid(True, linewidth=0.5, color='black')

    # Minor ticks
    if plot_minor_ticks:
        # X axis
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, index: tick_formatter(value)))
        plt.setp(ax.xaxis.get_minorticklabels(), fontsize=font_size-1, rotation=45)
        # Y axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda value, index: tick_formatter(value)))
        plt.setp(ax.yaxis.get_minorticklabels(), fontsize=font_size-1)
        # Minor tick grid
        plt.grid(True, which='minor', linewidth=0.4, color='black', alpha=0.6, linestyle='--')

    # Save
    if save:
        plt.savefig(filename, dpi=300)

    # Show
    if show:
        plt.show()

    return contour_lines


def plot_porkchop(
    # Data arguments
    departure_body: str,
    target_body: str,
    departure_epochs: np.ndarray, 
    arrival_epochs: np.ndarray, 
    delta_v: np.ndarray,
    # Plot arguments
    C3: bool = False,
    total: bool = False,
    threshold: float = 10,
    upscale: bool = False,
    number_of_levels: int = 10,
    # Figure arguments
    percent_margin: float = 5,
    figsize: tuple[int, int] = (8, 8),
    show: bool = True,
    save: bool = False,
    filename: str = 'porkchop.png',
    ) -> None:
    """
    ΔV/C3 porkchop mission design plot.
    
    Parameters
    ----------
    departure_body: str
        The name of the body from which the transfer is to be computed
    target_body: str
        The name of the body to which the transfer is to be computed
    departure_epochs: np.ndarray
        Discretized departure time window
    arrival_epochs: np.ndarray
        Discretized arrival time window
    delta_v: np.ndarray
        Array containing the ΔV of all coordinates of the grid of departure/arrival epochs
    C3: bool = False
        Whether to plot C3 (specific energy) instead of ΔV
    total: bool = False
        Whether to plot departure and arrival ΔV/C3, or only the total ΔV/C3. This option is only respected if the ΔV map obtained from
    threshold: float = 10
        Upper threshold beyond which ΔV/C3 is not plotted. This is useful to mask regions of the plot where the ΔV/C3 is too high to be of interest.
    upscale: bool = False
        Whether to use interpolation to increase the resolution of the plot. This is not always reliable, and the detail generated cannot be relied upon for analysis. Its only purpose is aesthetic improvement.
    number_of_levels: int = 10
        The number of levels in the ΔV/C3 contour plot
    percent_margin: float = 5
        Empty margin between the axes of the plot and the plotted data
    figsize: tuple[int, int] = (8, 8)
        Size of the figure
    show: bool bool = True
        Whether to show the plot        
    save: bool = False
        Whether to save the plot
    filename: str = 'porkchop.png'
        The filename used for the saved plot
    """

    if (
        # Plotting total ΔV or C3
        total 
        or 
        # The ΔV array contains TOTAL ΔV for each departure
        # and arrival date combination, instead of a tuple
        # containing (departure ΔV, arrival ΔV)
        delta_v.shape[2] == 1
    ):

        # Plot departure ΔV or C3
        plot_porkchop_of_single_field(
            departure_body          = departure_body,
            target_body             = target_body,
            departure_epochs        = departure_epochs,
            arrival_epochs          = arrival_epochs,
            delta_v                 = delta_v.sum(axis=2),
            C3                      = C3,
            threshold               = threshold,
            upscale                 = upscale,
            # Plot arguments
            line_width              = 0.5,
            line_color              = 'black',
            filled_contours         = True,
            plot_out_of_range_field = True,
            number_of_levels        = number_of_levels,
            # Figure arguments
            colorbar                = True,
            percent_margin          = percent_margin,
            figsize                 = figsize,
            show                    = False,
        )

    else:

        # Plot departure ΔV or C3
        plot_porkchop_of_single_field(
            departure_body          = departure_body,
            target_body             = target_body,
            departure_epochs        = departure_epochs,
            arrival_epochs          = arrival_epochs,
            delta_v                 = delta_v[:, :, 0],
            C3                      = C3,
            threshold               = threshold,
            upscale                 = upscale,
            # Plot arguments
            plot_out_of_range_field = False,
            number_of_levels        = number_of_levels,
            label                   = 'Dep.',
            # Figure arguments
            colorbar                = True,
            percent_margin          = percent_margin,
            figsize                 = figsize,
            show                    = False,
        )

        # Plot departure ΔV or C3
        plot_porkchop_of_single_field(
            departure_body          = departure_body,
            target_body             = target_body,
            departure_epochs        = departure_epochs,
            arrival_epochs          = arrival_epochs,
            delta_v                 = delta_v[:, :, 1],
            C3                      = C3,
            threshold               = threshold,
            upscale                 = upscale,
            # Plot arguments
            line_width              = 0.5,
            line_color              = '#d90028',
            filled_contours         = False,
            plot_out_of_range_field = False,
            plot_isochrones         = False,
            number_of_levels        = number_of_levels,
            label                   = 'Arr.',
            # Figure arguments
            percent_margin          = percent_margin,
            fig                     = plt.gcf(),
            ax                      = plt.gca(),
            show                    = False,
        )

    # Legend
    plt.gcf().legend(
        loc='center', ncol=1, 
        bbox_to_anchor=(0, 0, plt.gca().get_position().x0, plt.gca().get_position().y0),
        prop={'size': 8}
    )

    # Save
    if save:
        plt.savefig(filename, dpi=300)

    # Show
    if show:
        plt.show()
        

## Environment setup
"""

We proceed to set up the simulation environment, by loading the standard Spice kernels, defining the origin of the global frame and creating all necessary bodies. 
"""

# Load spice kernels
#spice_interface.load_standard_kernels( )
#spice_interface.load_kernel("/Users/luigiserra/.tudat/resource/spice_kernels/de430.bsp")

bodies = environment_setup.create_simplified_system_of_bodies()

# Define global frame orientation
global_frame_orientation = 'ECLIPJ2000'

# Create bodies
#bodies_to_create = ['Sun', 'Venus', 'Earth', 'Moon', 'Mars', 'Jupiter', 'Saturn','Neptune']
global_frame_origin = 'Sun'
#body_settings = environment_setup.get_default_body_settings(
#    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create environment model
#bodies = environment_setup.create_system_of_bodies(body_settings)

## Porkchop Plots
"""
The departure and target bodies and the time window for the transfer are then defined using tudatpy `astro.time_conversion.DateTime` objects.
"""

departure_body = 'Jupiter'
target_body = 'Neptune'

earliest_departure_time = DateTime(2050,  3,  11)
latest_departure_time   = DateTime(2060 , 6,   7)

earliest_arrival_time   = DateTime(2063, 11,  16)
latest_arrival_time     = DateTime(2080, 12,  21)

# To ensure the porkchop plot is rendered with good resolution, the time resolution of the plot is defined as 0.5% of the smallest time window (either the arrival or the departure window):

# Set time resolution IN DAYS as 0.5% of the smallest window (be it departure, or arrival)
# This will result in fairly good time resolution, at a runtime of approximately 10 seconds
# Tune the time resolution to obtain results to your liking!
time_window_percentage = 0.1
time_resolution = time_resolution = min(
        latest_departure_time.epoch() - earliest_departure_time.epoch(),
        latest_arrival_time.epoch()   - earliest_arrival_time.epoch()
) / constants.JULIAN_DAY * time_window_percentage / 100

# Generating a high-resolution plot may be time-consuming: reusing saved data might be desirable; we proceed to ask the user whether to reuse saved data or generate the plot from scratch.

# File
data_file = 'porkchop.pkl'

# Whether to recalculate the porkchop plot or use saved data
RECALCULATE_delta_v = input(
    '\n    Recalculate ΔV for porkchop plot? [y/N] '
).strip().lower() == 'y'
print()

# Lastly, we call the `porkchop` function, which will calculate the $\Delta V$ required at each departure-arrival coordinate and display the plot, giving us
# 
# - The optimal departure-arrival date combination
# - The constant time-of-flight isochrones
# - And more

if not os.path.isfile(data_file) or RECALCULATE_delta_v:
    # Regenerate plot
    [departure_epochs, arrival_epochs, ΔV] = porkchop(
        bodies,
        departure_body,
        target_body,
        earliest_departure_time,
        latest_departure_time,
        earliest_arrival_time,
        latest_arrival_time,
        time_resolution,
        total = False
    )
    # Save data
    pickle.dump(
        [departure_epochs, arrival_epochs, ΔV],
        open(data_file, 'wb')
    )
else:
    # Read saved data
    [departure_epochs, arrival_epochs, ΔV] = pickle.load(
        open(data_file, 'rb')
    )

    # Plot saved data
    plot_porkchop(
        departure_body   = departure_body,
        target_body      = target_body,
        departure_epochs = departure_epochs, 
        arrival_epochs   = arrival_epochs, 
        delta_v          = ΔV,
        threshold        = 15,
        total=False
    )
