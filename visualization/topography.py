#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Topography Visualization functions.

This module provides functions for visualizing EEG data as
scalp topographies.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def default_electrode_positions(n_electrodes=64):
    """
    Generate default electrode positions in a circular layout.
    
    Args:
        n_electrodes: int - number of electrodes
    
    Returns:
        np.ndarray of shape (n_electrodes, 2) - (x, y) positions
    """
    angles = np.linspace(0, 2 * np.pi, n_electrodes, endpoint=False)
    
    # Create concentric circles
    positions = np.zeros((n_electrodes, 2))
    
    n_rings = max(1, n_electrodes // 8)
    electrodes_per_ring = n_electrodes // n_rings
    
    idx = 0
    for ring in range(n_rings):
        r = 0.3 + 0.6 * (ring + 1) / n_rings
        n_in_ring = min(electrodes_per_ring, n_electrodes - idx)
        
        for i in range(n_in_ring):
            angle = 2 * np.pi * i / n_in_ring + ring * 0.1
            positions[idx, 0] = r * np.cos(angle)
            positions[idx, 1] = r * np.sin(angle)
            idx += 1
    
    return positions


def plot_topography_simple(values, out_path=None, title="Topography",
                           positions=None, vmin=None, vmax=None,
                           cmap='RdBu_r', dpi=300, figsize=(6, 6)):
    """
    Plot a simple topographic map of electrode values.
    
    Args:
        values: np.ndarray of shape (n_electrodes,) - values per electrode
        out_path: str, optional - path to save figure
        title: str - plot title
        positions: np.ndarray, optional - electrode positions (n, 2)
        vmin, vmax: float, optional - color scale limits
        cmap: str - colormap
        dpi: int - figure resolution
        figsize: tuple - figure size
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    n_electrodes = len(values)
    
    if positions is None:
        positions = default_electrode_positions(n_electrodes)
    
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    
    # Make symmetric if using diverging colormap
    if 'RdBu' in cmap or 'coolwarm' in cmap:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw head outline
    circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=2, color='black')
    ax.add_patch(circle)
    
    # Draw nose
    ax.plot([0, 0], [1.0, 1.15], 'k-', linewidth=2)
    ax.plot([-0.1, 0, 0.1], [1.1, 1.15, 1.1], 'k-', linewidth=2)
    
    # Draw ears
    ax.plot([-1.0, -1.1, -1.0], [-0.1, 0, 0.1], 'k-', linewidth=2)
    ax.plot([1.0, 1.1, 1.0], [-0.1, 0, 0.1], 'k-', linewidth=2)
    
    # Plot electrode values
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=values,
                         cmap=cmap, vmin=vmin, vmax=vmax, s=150,
                         edgecolors='black', linewidths=0.5, zorder=5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Value', fontsize=10)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.2, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_topography_contour(values, out_path=None, title="Topography",
                            positions=None, vmin=None, vmax=None,
                            cmap='RdBu_r', dpi=300, figsize=(6, 6),
                            n_contours=20, resolution=100):
    """
    Plot a contour-interpolated topographic map.
    
    Args:
        values: np.ndarray of shape (n_electrodes,) - values per electrode
        out_path: str, optional - path to save figure
        title: str - plot title
        positions: np.ndarray, optional - electrode positions
        vmin, vmax: float, optional - color scale limits
        cmap: str - colormap
        dpi: int - figure resolution
        figsize: tuple - figure size
        n_contours: int - number of contour levels
        resolution: int - interpolation grid resolution
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    from scipy.interpolate import griddata
    
    n_electrodes = len(values)
    
    if positions is None:
        positions = default_electrode_positions(n_electrodes)
    
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    
    # Create interpolation grid
    xi = np.linspace(-1.1, 1.1, resolution)
    yi = np.linspace(-1.1, 1.1, resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate
    zi = griddata(positions, values, (xi, yi), method='cubic')
    
    # Mask outside head
    mask = np.sqrt(xi**2 + yi**2) > 1.0
    zi[mask] = np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Contour plot
    levels = np.linspace(vmin, vmax, n_contours)
    contour = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, extend='both')
    ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.3, alpha=0.5)
    
    # Draw head outline
    circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=2, color='black')
    ax.add_patch(circle)
    
    # Draw nose and ears
    ax.plot([0, 0], [1.0, 1.15], 'k-', linewidth=2)
    ax.plot([-0.1, 0, 0.1], [1.1, 1.15, 1.1], 'k-', linewidth=2)
    ax.plot([-1.0, -1.1, -1.0], [-0.1, 0, 0.1], 'k-', linewidth=2)
    ax.plot([1.0, 1.1, 1.0], [-0.1, 0, 0.1], 'k-', linewidth=2)
    
    # Mark electrode positions
    ax.scatter(positions[:, 0], positions[:, 1], c='black', s=10, zorder=5)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.6)
    cbar.set_label('Value', fontsize=10)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.2, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_topography_grid(values_list, titles_list, out_path=None,
                         positions=None, vmin=None, vmax=None,
                         cmap='RdBu_r', dpi=300, ncols=4):
    """
    Plot multiple topographies in a grid.
    
    Args:
        values_list: list of np.ndarray - values for each topography
        titles_list: list of str - titles for each topography
        out_path: str, optional - path to save figure
        positions: np.ndarray, optional - electrode positions
        vmin, vmax: float, optional - color scale limits (shared)
        cmap: str - colormap
        dpi: int - figure resolution
        ncols: int - number of columns in grid
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    n_plots = len(values_list)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    
    if vmin is None:
        vmin = min(np.nanmin(v) for v in values_list)
    if vmax is None:
        vmax = max(np.nanmax(v) for v in values_list)
    
    n_electrodes = len(values_list[0])
    if positions is None:
        positions = default_electrode_positions(n_electrodes)
    
    for idx, (values, title) in enumerate(zip(values_list, titles_list)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Draw head outline
        circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=1, color='black')
        ax.add_patch(circle)
        
        # Plot values
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=values,
                             cmap=cmap, vmin=vmin, vmax=vmax, s=50,
                             edgecolors='black', linewidths=0.3)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cb = fig.colorbar(scatter, cax=cax)
    cb.set_label('Value', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


if __name__ == "__main__":
    # Test the functions
    import tempfile
    import os
    
    # Create test data
    n_electrodes = 64
    values = np.random.randn(n_electrodes)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_topography_simple(values, os.path.join(tmpdir, "topo_simple.png"))
        plot_topography_contour(values, os.path.join(tmpdir, "topo_contour.png"))
        
        values_list = [np.random.randn(n_electrodes) for _ in range(8)]
        titles_list = [f"Time {i*100}ms" for i in range(8)]
        plot_topography_grid(values_list, titles_list,
                             os.path.join(tmpdir, "topo_grid.png"))
        
        print("Topography tests passed!")

