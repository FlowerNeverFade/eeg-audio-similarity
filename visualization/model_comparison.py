#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model comparison visualization.

Functions for comparing multiple audio models across layers and metrics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_model_results(csv_path: str) -> pd.DataFrame:
    """
    Load model results from CSV file.
    
    Args:
        csv_path: str - path to CSV file
    
    Returns:
        pd.DataFrame with results
    """
    return pd.read_csv(csv_path)


def compute_layer_averages(df: pd.DataFrame, metric_col: str = 'rsa_spearman',
                           layer_col: str = 'layer_idx') -> pd.DataFrame:
    """
    Compute average metric values per layer.
    
    Args:
        df: pd.DataFrame - results dataframe
        metric_col: str - column name for metric
        layer_col: str - column name for layer index
    
    Returns:
        pd.DataFrame with layer averages
    """
    grouped = df.groupby(layer_col).agg({
        metric_col: ['mean', 'std', 'count']
    }).reset_index()
    grouped.columns = [layer_col, 'mean', 'std', 'count']
    return grouped


def plot_layerwise_comparison(models_data: Dict[str, pd.DataFrame],
                               metric: str = 'rsa_spearman',
                               title: str = "Layer-wise Comparison",
                               out_path: str = None,
                               figsize: Tuple[int, int] = (12, 6),
                               dpi: int = 300) -> Optional[plt.Figure]:
    """
    Plot layer-wise comparison across multiple models.
    
    Args:
        models_data: dict - {model_name: DataFrame} with layer averages
        metric: str - metric name for y-axis label
        title: str - plot title
        out_path: str, optional - path to save figure
        figsize: tuple - figure size
        dpi: int - figure resolution
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    for (model_name, df), color in zip(models_data.items(), colors):
        layers = df['layer_idx'].values
        means = df['mean'].values
        stds = df['std'].values if 'std' in df.columns else np.zeros_like(means)
        
        ax.plot(layers, means, 'o-', label=model_name, color=color, linewidth=2, markersize=6)
        ax.fill_between(layers, means - stds, means + stds, alpha=0.2, color=color)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_model_grid(models_data: Dict[str, pd.DataFrame],
                    metrics: List[str] = None,
                    out_path: str = None,
                    figsize: Tuple[int, int] = None,
                    dpi: int = 300) -> Optional[plt.Figure]:
    """
    Plot grid of models vs metrics.
    
    Args:
        models_data: dict - {model_name: DataFrame}
        metrics: list of str - metrics to plot
        out_path: str, optional - path to save figure
        figsize: tuple - figure size
        dpi: int - figure resolution
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    if metrics is None:
        metrics = ['rsa_spearman', 'rsa_pearson', 'cka_linear', 'cka_rbf']
    
    n_models = len(models_data)
    n_metrics = len(metrics)
    
    if figsize is None:
        figsize = (4 * n_metrics, 3 * n_models)
    
    fig, axes = plt.subplots(n_models, n_metrics, figsize=figsize, squeeze=False)
    
    for i, (model_name, df) in enumerate(models_data.items()):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            if 'layer_idx' in df.columns and metric in df.columns:
                layers = df['layer_idx'].values
                values = df[metric].values
                ax.plot(layers, values, 'o-', color='#1f77b4', linewidth=1.5, markersize=4)
            
            if i == 0:
                ax.set_title(metric, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(model_name, fontsize=10)
            if i == n_models - 1:
                ax.set_xlabel('Layer', fontsize=9)
            
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def create_summary_table(models_data: Dict[str, pd.DataFrame],
                         metrics: List[str] = None,
                         out_path: str = None) -> pd.DataFrame:
    """
    Create summary table of best layer performance for each model.
    
    Args:
        models_data: dict - {model_name: DataFrame}
        metrics: list of str - metrics to summarize
        out_path: str, optional - path to save CSV
    
    Returns:
        pd.DataFrame with summary
    """
    if metrics is None:
        metrics = ['rsa_spearman', 'rsa_pearson', 'cka_linear']
    
    rows = []
    for model_name, df in models_data.items():
        row = {'model': model_name}
        
        for metric in metrics:
            if metric in df.columns:
                row[f'{metric}_max'] = df[metric].max()
                row[f'{metric}_best_layer'] = df.loc[df[metric].idxmax(), 'layer_idx'] if 'layer_idx' in df.columns else -1
                row[f'{metric}_mean'] = df[metric].mean()
        
        rows.append(row)
    
    summary = pd.DataFrame(rows)
    
    if out_path:
        summary.to_csv(out_path, index=False)
    
    return summary


def plot_best_layer_bar(summary_df: pd.DataFrame,
                        metric: str = 'rsa_spearman_max',
                        title: str = "Best Layer Performance",
                        out_path: str = None,
                        figsize: Tuple[int, int] = (10, 6),
                        dpi: int = 300) -> Optional[plt.Figure]:
    """
    Plot bar chart of best layer performance across models.
    
    Args:
        summary_df: pd.DataFrame - summary table from create_summary_table
        metric: str - metric column to plot
        title: str - plot title
        out_path: str, optional - path to save figure
        figsize: tuple - figure size
        dpi: int - figure resolution
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    if metric not in summary_df.columns:
        raise ValueError(f"Metric {metric} not found in summary")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = summary_df['model'].values
    values = summary_df[metric].values
    
    # Sort by value
    sorted_idx = np.argsort(values)[::-1]
    models = models[sorted_idx]
    values = values[sorted_idx]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    bars = ax.barh(range(len(models)), values, color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    
    models = {
        'Model_A': pd.DataFrame({
            'layer_idx': range(12),
            'rsa_spearman': np.random.randn(12) * 0.1 + 0.3,
            'cka_linear': np.random.randn(12) * 0.1 + 0.5,
        }),
        'Model_B': pd.DataFrame({
            'layer_idx': range(24),
            'rsa_spearman': np.random.randn(24) * 0.1 + 0.25,
            'cka_linear': np.random.randn(24) * 0.1 + 0.45,
        }),
    }
    
    summary = create_summary_table(models)
    print(summary)

