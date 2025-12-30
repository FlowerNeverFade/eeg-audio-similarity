#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Histogram and Distribution Visualization.

This module provides functions for plotting histograms and distributions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_histogram(values, title="Distribution", out_path=None, 
                   bins=30, color='#1f77b4', dpi=300, figsize=(8, 4),
                   xlabel='Value', ylabel='Count'):
    """
    Plot a histogram of values.
    
    Args:
        values: array-like - values to plot
        title: str - plot title
        out_path: str, optional - path to save figure
        bins: int - number of bins
        color: str - bar color
        dpi: int - figure resolution
        figsize: tuple - figure size
        xlabel: str - x-axis label
        ylabel: str - y-axis label
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(values, bins=bins, alpha=0.85, color=color, edgecolor='white')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_rsa_distribution(rsa_values, title="RSA Distribution", out_path=None,
                          dpi=300, figsize=(8, 4)):
    """
    Plot distribution of RSA values with statistics.
    
    Args:
        rsa_values: array-like - RSA correlation values
        title: str - plot title
        out_path: str, optional - path to save figure
        dpi: int - figure resolution
        figsize: tuple - figure size
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    values = np.asarray(rsa_values).flatten()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(values, bins=30, alpha=0.7, color='#1f77b4', edgecolor='white')
    
    # Add statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.3f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5,
               label=f'±1 SD: {std_val:.3f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('RSA (Spearman)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_lag_histogram(lags, title="Best Lag Distribution", out_path=None,
                       dpi=300, figsize=(8, 4)):
    """
    Plot histogram of best lag values.
    
    Args:
        lags: array-like - lag values in ms
        title: str - plot title
        out_path: str, optional - path to save figure
        dpi: int - figure resolution
        figsize: tuple - figure size
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    lags = np.asarray(lags).flatten()
    unique_lags = sorted(np.unique(lags))
    counts = [np.sum(lags == u) for u in unique_lags]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar([str(int(u)) for u in unique_lags], counts, color='#9467bd', 
           edgecolor='white')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Lag (ms)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_scatter_with_colorbar(x, y, c, title="Scatter Plot", out_path=None,
                                xlabel='X', ylabel='Y', clabel='Color',
                                cmap='plasma', dpi=300, figsize=(6, 6)):
    """
    Plot scatter with colorbar.
    
    Args:
        x: array-like - x values
        y: array-like - y values
        c: array-like - color values
        title: str - plot title
        out_path: str, optional - path to save figure
        xlabel, ylabel, clabel: str - axis labels
        cmap: str - colormap
        dpi: int - figure resolution
        figsize: tuple - figure size
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(x, y, c=c, cmap=cmap, s=20, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(clabel, fontsize=10)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


if __name__ == "__main__":
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test histogram
        values = np.random.randn(1000)
        plot_histogram(values, out_path=os.path.join(tmpdir, "hist.png"))
        
        # Test RSA distribution
        rsa = np.random.randn(500) * 0.2 + 0.1
        plot_rsa_distribution(rsa, out_path=os.path.join(tmpdir, "rsa_dist.png"))
        
        # Test lag histogram
        lags = np.random.choice([0, 50, 100, 150], size=200)
        plot_lag_histogram(lags, out_path=os.path.join(tmpdir, "lag_hist.png"))
        
        print("Histogram tests passed!")

