#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDM Visualization functions.

This module provides functions for visualizing Representational
Dissimilarity Matrices (RDMs) and their comparisons.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform


def rdm_vec_to_matrix(rdm_vec):
    """
    Convert an upper triangular RDM vector to a full symmetric matrix.
    
    Args:
        rdm_vec: np.ndarray of shape (n_pairs,) - upper triangular values
    
    Returns:
        np.ndarray of shape (T, T) - symmetric RDM matrix
    """
    if rdm_vec is None:
        return None
    
    if rdm_vec.ndim == 1:
        try:
            return squareform(rdm_vec).astype(np.float32)
        except Exception:
            return None
    
    return rdm_vec.astype(np.float32)


def plot_rdm(rdm_matrix, title="RDM", out_path=None, cmap='viridis_r',
             vmin=0.0, vmax=1.0, dpi=300, figsize=(6, 5)):
    """
    Plot a single RDM matrix.
    
    Args:
        rdm_matrix: np.ndarray of shape (T, T) - RDM matrix
        title: str - plot title
        out_path: str, optional - path to save figure
        cmap: str - colormap (default: 'viridis_r')
        vmin, vmax: float - color scale limits
        dpi: int - figure resolution
        figsize: tuple - figure size
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(rdm_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Frame', fontsize=10)
    ax.set_ylabel('Time Frame', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Dissimilarity', fontsize=10)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_rdm_pair(rdm1, rdm2, title1="Audio RDM", title2="EEG RDM",
                  out_path=None, cmap='viridis_r', vmin=0.0, vmax=1.0,
                  dpi=300, show_correlation=True):
    """
    Plot a pair of RDMs side by side with optional correlation.
    
    Args:
        rdm1: np.ndarray - first RDM matrix (or vector)
        rdm2: np.ndarray - second RDM matrix (or vector)
        title1: str - title for first RDM
        title2: str - title for second RDM
        out_path: str, optional - path to save figure
        cmap: str - colormap
        vmin, vmax: float - color scale limits
        dpi: int - figure resolution
        show_correlation: bool - whether to show Spearman correlation
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    
    Example:
        >>> rdm_audio = np.random.rand(100, 100)
        >>> rdm_eeg = np.random.rand(100, 100)
        >>> plot_rdm_pair(rdm_audio, rdm_eeg, out_path="rdm_comparison.png")
    """
    # Convert vectors to matrices if needed
    if rdm1.ndim == 1:
        rdm1 = rdm_vec_to_matrix(rdm1)
    if rdm2.ndim == 1:
        rdm2 = rdm_vec_to_matrix(rdm2)
    
    if rdm1 is None or rdm2 is None:
        return None
    
    # Ensure same size
    T = min(rdm1.shape[0], rdm2.shape[0])
    rdm1 = rdm1[:T, :T]
    rdm2 = rdm2[:T, :T]
    
    # Clip values
    rdm1 = np.clip(rdm1, vmin, vmax)
    rdm2 = np.clip(rdm2, vmin, vmax)
    
    # Calculate Spearman correlation
    rho, pval = None, None
    if show_correlation:
        triu_idx = np.triu_indices(T, k=1)
        rdm1_triu = rdm1[triu_idx]
        rdm2_triu = rdm2[triu_idx]
        rho, pval = spearmanr(rdm1_triu, rdm2_triu)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Left RDM
    im0 = axes[0].imshow(rdm1, cmap=cmap, vmin=vmin, vmax=vmax,
                         interpolation='nearest', aspect='equal')
    axes[0].set_title(title1, fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Frame', fontsize=10)
    axes[0].set_ylabel('Time Frame', fontsize=10)
    
    # Right RDM
    im1 = axes[1].imshow(rdm2, cmap=cmap, vmin=vmin, vmax=vmax,
                         interpolation='nearest', aspect='equal')
    axes[1].set_title(title2, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frame', fontsize=10)
    axes[1].set_ylabel('Time Frame', fontsize=10)
    
    # Add correlation info
    if show_correlation and rho is not None:
        pval_str = f'{pval:.2e}' if pval < 0.001 else f'{pval:.4f}'
        fig.suptitle(f'RDM Similarity: Spearman ρ = {rho:.3f} (p = {pval_str})',
                     fontsize=13, fontweight='bold', y=0.98)
    
    # Colorbar
    fig.subplots_adjust(bottom=0.18, wspace=0.35, top=0.88)
    cax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    cb = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cb.set_label('Dissimilarity', fontsize=10)
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_layer_profile(layer_values, title="Layer Profile", xlabel="Layer Index",
                       ylabel="RSA (Spearman)", out_path=None, color='#1f77b4',
                       dpi=300, figsize=(10, 4)):
    """
    Plot RSA values across model layers.
    
    Args:
        layer_values: array-like - RSA values per layer
        title: str - plot title
        xlabel: str - x-axis label
        ylabel: str - y-axis label
        out_path: str, optional - path to save figure
        color: str - line color
        dpi: int - figure resolution
        figsize: tuple - figure size
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(layer_values))
    ax.plot(x, layer_values, color=color, linewidth=2, marker='o', markersize=4)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_heatmap_1d(values, title="", out_path=None, cmap='viridis',
                    cbar_label=None, dpi=300, xticks=None, figsize=(12, 1.8)):
    """
    Plot a 1D heatmap (single row).
    
    Args:
        values: array-like - values to plot
        title: str - plot title
        out_path: str, optional - path to save figure
        cmap: str - colormap
        cbar_label: str, optional - colorbar label
        dpi: int - figure resolution
        xticks: list, optional - x-axis tick labels
        figsize: tuple - figure size
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    
    ax.set_yticks([0])
    ax.set_yticklabels([''])
    
    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks)
    
    ax.set_title(title)
    plt.tight_layout()
    
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
    T = 50
    rdm1 = np.random.rand(T, T)
    rdm1 = (rdm1 + rdm1.T) / 2  # Make symmetric
    np.fill_diagonal(rdm1, 0)
    
    rdm2 = rdm1 + 0.1 * np.random.rand(T, T)
    rdm2 = (rdm2 + rdm2.T) / 2
    np.fill_diagonal(rdm2, 0)
    
    # Test plots
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_rdm(rdm1, "Test RDM", os.path.join(tmpdir, "rdm.png"))
        plot_rdm_pair(rdm1, rdm2, out_path=os.path.join(tmpdir, "rdm_pair.png"))
        
        layer_vals = np.random.rand(32)
        plot_layer_profile(layer_vals, out_path=os.path.join(tmpdir, "layers.png"))
        
        print("Visualization tests passed!")

