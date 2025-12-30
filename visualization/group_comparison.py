#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group comparison visualization.

Functions for comparing RSA/CKA results across different groups
(e.g., high vs low arousal, different prosody clusters).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from pathlib import Path


def load_group_data(csv_path: str, group_col: str = 'group') -> pd.DataFrame:
    """
    Load data with group labels.
    
    Args:
        csv_path: str - path to CSV file
        group_col: str - column name for group labels
    
    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found")
    return df


def compute_group_statistics(df: pd.DataFrame, 
                              group_col: str = 'group',
                              metric_cols: List[str] = None) -> pd.DataFrame:
    """
    Compute statistics for each group.
    
    Args:
        df: pd.DataFrame - data with group labels
        group_col: str - column name for group
        metric_cols: list of str - metric columns to analyze
    
    Returns:
        pd.DataFrame with group statistics
    """
    if metric_cols is None:
        metric_cols = ['rsa_spearman', 'cka_linear']
    
    # Filter to existing columns
    metric_cols = [c for c in metric_cols if c in df.columns]
    
    if not metric_cols:
        raise ValueError("No valid metric columns found")
    
    stats = df.groupby(group_col)[metric_cols].agg(['mean', 'std', 'count'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    
    return stats.reset_index()


def plot_group_boxplot(df: pd.DataFrame,
                       group_col: str = 'group',
                       metric_col: str = 'rsa_spearman',
                       title: str = None,
                       out_path: str = None,
                       figsize: Tuple[int, int] = (8, 6),
                       dpi: int = 300,
                       colors: List[str] = None) -> Optional[plt.Figure]:
    """
    Plot boxplot comparing groups.
    
    Args:
        df: pd.DataFrame - data
        group_col: str - group column
        metric_col: str - metric column
        title: str - plot title
        out_path: str, optional - save path
        figsize: tuple - figure size
        dpi: int - resolution
        colors: list of str - colors for each group
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    groups = df[group_col].unique()
    n_groups = len(groups)
    
    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data = [df[df[group_col] == g][metric_col].dropna().values for g in groups]
    
    bp = ax.boxplot(data, patch_artist=True, labels=groups)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_xlabel(group_col, fontsize=12)
    
    if title is None:
        title = f'{metric_col} by {group_col}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_group_violin(df: pd.DataFrame,
                      group_col: str = 'group',
                      metric_col: str = 'rsa_spearman',
                      title: str = None,
                      out_path: str = None,
                      figsize: Tuple[int, int] = (8, 6),
                      dpi: int = 300) -> Optional[plt.Figure]:
    """
    Plot violin plot comparing groups.
    
    Args:
        df: pd.DataFrame - data
        group_col: str - group column
        metric_col: str - metric column
        title: str - plot title
        out_path: str, optional - save path
        figsize: tuple - figure size
        dpi: int - resolution
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    groups = sorted(df[group_col].unique())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data = [df[df[group_col] == g][metric_col].dropna().values for g in groups]
    
    parts = ax.violinplot(data, positions=range(len(groups)), showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_xlabel(group_col, fontsize=12)
    
    if title is None:
        title = f'{metric_col} Distribution by {group_col}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def plot_group_comparison_grid(df: pd.DataFrame,
                                group_col: str = 'group',
                                metrics: List[str] = None,
                                out_path: str = None,
                                figsize: Tuple[int, int] = None,
                                dpi: int = 300) -> Optional[plt.Figure]:
    """
    Plot grid of group comparisons for multiple metrics.
    
    Args:
        df: pd.DataFrame - data
        group_col: str - group column
        metrics: list of str - metrics to compare
        out_path: str, optional - save path
        figsize: tuple - figure size
        dpi: int - resolution
    
    Returns:
        matplotlib.figure.Figure if out_path is None
    """
    if metrics is None:
        metrics = ['rsa_spearman', 'rsa_pearson', 'cka_linear', 'cka_rbf']
    
    metrics = [m for m in metrics if m in df.columns]
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        raise ValueError("No valid metrics found")
    
    if figsize is None:
        figsize = (4 * min(n_metrics, 3), 4 * ((n_metrics - 1) // 3 + 1))
    
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    groups = sorted(df[group_col].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    
    for idx, metric in enumerate(metrics):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        data = [df[df[group_col] == g][metric].dropna().values for g in groups]
        
        bp = ax.boxplot(data, patch_artist=True, labels=groups)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return None
    
    return fig


def perform_group_ttest(df: pd.DataFrame,
                        group_col: str = 'group',
                        metric_col: str = 'rsa_spearman') -> Dict:
    """
    Perform t-test between groups.
    
    Args:
        df: pd.DataFrame - data
        group_col: str - group column
        metric_col: str - metric column
    
    Returns:
        dict with t-statistic and p-value for each pair
    """
    from scipy import stats
    
    groups = sorted(df[group_col].unique())
    results = {}
    
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            d1 = df[df[group_col] == g1][metric_col].dropna().values
            d2 = df[df[group_col] == g2][metric_col].dropna().values
            
            if len(d1) > 1 and len(d2) > 1:
                t_stat, p_val = stats.ttest_ind(d1, d2)
                results[f'{g1}_vs_{g2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'n1': len(d1),
                    'n2': len(d2),
                    'mean1': np.mean(d1),
                    'mean2': np.mean(d2),
                }
    
    return results


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    
    df = pd.DataFrame({
        'group': np.random.choice(['High', 'Low'], 100),
        'rsa_spearman': np.random.randn(100) * 0.1 + 0.3,
        'cka_linear': np.random.randn(100) * 0.1 + 0.5,
    })
    
    # Adjust high group to be higher
    df.loc[df['group'] == 'High', 'rsa_spearman'] += 0.1
    
    stats = compute_group_statistics(df)
    print(stats)
    
    ttest = perform_group_ttest(df)
    print(ttest)

