"""Wrappers to plot movement data."""

import xarray as xr
from matplotlib import pyplot as plt
import movement.kinematics as kin
import numpy as np


DEFAULT_PLOTTING_ARGS = {
    "s": 15,
    "marker": "o",
    "alpha": 1.0,
}


def plot_centroid_trajectory(
        da: xr.DataArray,
        individual: str | None = None,
        keypoints: str | list[str] | None = None,
        ax: plt.Axes | None = None,
        manual_color_var=False,
        suppress_colorbar=False,
        **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot centroid trajectory.

    This function plots the trajectory of the centroid
    of multiple keypoints for a given individual. By default, the trajectory
    is colored by time (using the default colormap). Pass a different colormap
    through ``cmap`` if desired. If a single keypoint is passed, the trajectory
    will be the same as the trajectory of the keypoint.

    Parameters
    ----------
    da : xr.DataArray
        A data array containing position information, with `time` and `space`
        as required dimensions. Optionally, it may have `individuals` and/or
        `keypoints` dimensions.
    individual : str, optional
        The name of the individual to be plotted. By default, the first
        individual is plotted.
    keypoints : str, list[str], optional
        The name of the keypoint to be plotted, or a list of keypoint names
        (their centroid will be plotted). By default, the centroid of all
        keypoints is plotted.
    ax : matplotlib.axes.Axes or None, optional
        Axes object on which to draw the trajectory. If None, a new
        figure and axes are created.
    **kwargs : dict
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.scatter()``.

    Returns
    -------
    (figure, axes) : tuple of (matplotlib.pyplot.Figure, matplotlib.axes.Axes)
        The figure and axes containing the trajectory plot.

    """
    if isinstance(individual, list):
        raise ValueError("Only one individual can be selected.")

    selection = {}

    if "individuals" in da.dims:
        if individual is None:
            selection["individuals"] = da.individuals.values[0]
        else:
            selection["individuals"] = individual

    if "keypoints" in da.dims:
        if keypoints is None:
            selection["keypoints"] = da.keypoints.values
        else:
            selection["keypoints"] = keypoints

    plot_point = da.sel(**selection)

    # If there are multiple selected keypoints, calculate the centroid
    plot_point = (
        plot_point.mean(dim="keypoints", skipna=True)
        if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1
        else plot_point
    )

    plot_point = plot_point.squeeze()  # Only space and time should remain

    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)

    # Merge default plotting args with user-provided kwargs
    for key, value in DEFAULT_PLOTTING_ARGS.items():
        kwargs.setdefault(key, value)

    colorbar = True if not suppress_colorbar else False

    if "c" not in kwargs:
        if manual_color_var and (manual_color_var in plot_point.coords):
            kwargs["c"] = plot_point[manual_color_var].values
            time_label = manual_color_var
        else:
            kwargs["c"] = plot_point.time
            time_label = "Time"

    sc = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        **kwargs,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory")

    time_label = manual_color_var if (manual_color_var and (manual_color_var in plot_point.attrs)) else "Time"
    fig.colorbar(sc, ax=ax, label=time_label).solids.set(
        alpha=1.0
    ) if colorbar else None

    return fig, ax


def selection(
        da: xr.DataArray,
        individual: str | None = None,
        keypoints: str | list[str] | None = None,
        **kwargs,

):
    if isinstance(individual, list):
        raise ValueError("Only one individual can be selected.")

    selection = {}

    if "individuals" in da.dims:
        if individual is None:
            selection["individuals"] = da.individuals.values[0]
        else:
            selection["individuals"] = individual

    if "keypoints" in da.dims:
        if keypoints is None:
            selection["keypoints"] = da.keypoints.values
        else:
            selection["keypoints"] = keypoints

    plot_point = da.sel(**selection)

    # If there are multiple selected keypoints, calculate the centroid
    plot_point = (
        plot_point.mean(dim="keypoints", skipna=True)
        if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1
        else plot_point
    )

    plot_point = plot_point.squeeze()  # Only space and time should remain
    return plot_point


def plot_centroid_trajectory_quiver(
        da: xr.DataArray,
        individual: str | None = None,
        keypoints: str | list[str] | None = None,
        ax: plt.Axes | None = None,
        manual_color_var=False,
        suppress_colorbar=False,
        **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot centroid trajectory.

    This function plots the trajectory of the centroid
    of multiple keypoints for a given individual. By default, the trajectory
    is colored by time (using the default colormap). Pass a different colormap
    through ``cmap`` if desired. If a single keypoint is passed, the trajectory
    will be the same as the trajectory of the keypoint.

    Parameters
    ----------
    da : xr.DataArray
        A data array containing position information, with `time` and `space`
        as required dimensions. Optionally, it may have `individuals` and/or
        `keypoints` dimensions.
    individual : str, optional
        The name of the individual to be plotted. By default, the first
        individual is plotted.
    keypoints : str, list[str], optional
        The name of the keypoint to be plotted, or a list of keypoint names
        (their centroid will be plotted). By default, the centroid of all
        keypoints is plotted.
    ax : matplotlib.axes.Axes or None, optional
        Axes object on which to draw the trajectory. If None, a new
        figure and axes are created.
    **kwargs : dict
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.scatter()``.

    Returns
    -------
    (figure, axes) : tuple of (matplotlib.pyplot.Figure, matplotlib.axes.Axes)
        The figure and axes containing the trajectory plot.

    """
    if isinstance(individual, list):
        raise ValueError("Only one individual can be selected.")

    selection = {}

    if "individuals" in da.dims:
        if individual is None:
            selection["individuals"] = da.individuals.values[0]
        else:
            selection["individuals"] = individual

    if "keypoints" in da.dims:
        if keypoints is None:
            selection["keypoints"] = da.keypoints.values
        else:
            selection["keypoints"] = keypoints

    plot_point = da.sel(**selection)

    plot_point = (
        plot_point.mean(dim="keypoints", skipna=True)
        if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1
        else plot_point
    )

    plot_point = plot_point.squeeze()  # Only space and time should remain

    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)

    # Merge default plotting args with user-provided kwargs
    for key, value in DEFAULT_PLOTTING_ARGS.items():
        kwargs.setdefault(key, value)

    colorbar = True if not suppress_colorbar else False

    if "c" not in kwargs:
        if manual_color_var and (manual_color_var in plot_point.attrs):
            kwargs["c"] = plot_point.attrs[manual_color_var]
            time_label = manual_color_var
        else:
            kwargs["c"] = plot_point.time
            time_label = "Time"

    sc = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        **kwargs,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory")

    time_label = manual_color_var if (manual_color_var and (manual_color_var in plot_point.attrs)) else "Time"
    fig.colorbar(sc, ax=ax, label=time_label).solids.set(
        alpha=1.0
    ) if colorbar else None

    return fig, ax

def plot_centroid_trajectory_by_states(
    da: xr.DataArray,
    c: np.ndarray | list,  # State values for masking
    speed: np.ndarray | list,  # Speed values for coloring
    individual: str | None = None,
    keypoints: str | list[str] | None = None,
    figsize: tuple = None,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot centroid trajectory separated by state values.

    This function plots the trajectory of the centroid of multiple keypoints 
    for a given individual, with separate subplots for each unique state value in c.
    Each subplot shows only the trajectory points corresponding to that state,
    colored by speed values. All subplots share the same 1024x1024 scale.

    Parameters
    ----------
    da : xr.DataArray
        A data array containing position information, with `time` and `space`
        as required dimensions. Optionally, it may have `individuals` and/or
        `keypoints` dimensions.
    c : array-like
        State values used to create masks for separate plots. Must have the
        same length as the time dimension of the selected data.
    speed : array-like
        Speed values used for coloring points. Must have the same length as
        the time dimension of the selected data.
    individual : str, optional
        The name of the individual to be plotted. By default, the first
        individual is plotted.
    keypoints : str, list[str], optional
        The name of the keypoint to be plotted, or a list of keypoint names
        (their centroid will be plotted). By default, the centroid of all
        keypoints is plotted.
    figsize : tuple, optional
        Figure size. If None, automatically determined based on number of states.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib.axes.Axes.scatter().

    Returns
    -------
    (figure, axes_list) : tuple of (matplotlib.pyplot.Figure, list[matplotlib.axes.Axes])
        The figure and list of axes containing the trajectory plots.
    """
    if isinstance(individual, list):
        raise ValueError("Only one individual can be selected.")

    selection = {}

    if "individuals" in da.dims:
        if individual is None:
            selection["individuals"] = da.individuals.values[0]
        else:
            selection["individuals"] = individual

    if "keypoints" in da.dims:
        if keypoints is None:
            selection["keypoints"] = da.keypoints.values
        else:
            selection["keypoints"] = keypoints

    plot_point = da.sel(**selection)

    # If there are multiple selected keypoints, calculate the centroid
    plot_point = (
        plot_point.mean(dim="keypoints", skipna=True)
        if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1
        else plot_point
    )

    plot_point = plot_point.squeeze()  # Only space and time should remain
    
    # Convert c and speed to numpy arrays for easier handling
    c = np.array(c)
    speed = np.array(speed)
    
    # Check that c and speed have the same length as time dimension
    if len(c) != len(plot_point.time):
        raise ValueError(f"Length of c ({len(c)}) must match time dimension ({len(plot_point.time)})")
    
    if len(speed) != len(plot_point.time):
        raise ValueError(f"Length of speed ({len(speed)}) must match time dimension ({len(plot_point.time)})")
    
    # Get unique state values
    unique_states = np.unique(c)
    n_states = len(unique_states)
    
    # Determine subplot layout
    n_cols = min(3, n_states)  # Max 3 columns
    n_rows = (n_states + n_cols - 1) // n_cols  # Ceiling division
    
    # Set figure size if not provided
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 4)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    # Merge default plotting args with user-provided kwargs
    for key, value in DEFAULT_PLOTTING_ARGS.items():
        kwargs.setdefault(key, value)
    
    # Remove 'c' from kwargs if present (we'll handle coloring differently)
    kwargs.pop('c', None)
    
    # Get global speed range for consistent coloring across subplots
    speed_min, speed_max = np.nanmin(speed), np.nanpercentile(speed, 95)  # ignores top 1% of values
    print('max speed:', np.nanmax(speed))

    axes_list = []
    
    for i, state in enumerate(unique_states):
        ax = axes_flat[i]
        axes_list.append(ax)
        
        # Create mask for current state
        mask = c == state
        
        # Apply mask to get data for this state
        masked_data = plot_point.isel(time=mask)
        masked_speed = speed[mask]
        
        # Skip if no data for this state
        if len(masked_data.time) == 0:
            ax.set_title(f'State {state} (No data)')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, 1024)
            ax.set_ylim(0, 1024)
            continue
        
        # Plot the scatter for this state
        sc = ax.scatter(
            masked_data.sel(space="x"),
            masked_data.sel(space="y"),
            c=masked_speed,  # Color by speed for this state
            vmin=speed_min,  # Use global speed range
            vmax=speed_max,
            **kwargs,
        )
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f'State {state}')
        ax.set_xlim(0, 1024)  # Set consistent scale
        ax.set_ylim(0, 1024)  # Set consistent scale
        
        # Add colorbar for speed
        fig.colorbar(sc, ax=ax, label="Speed").solids.set(alpha=1.0)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    return fig, axes_list