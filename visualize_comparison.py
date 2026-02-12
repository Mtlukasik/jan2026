"""
Visualization and Comparison Script for BNN Experiment Results.

This script:
1. Loads all completed runs from the results directory
2. Compares SVGD vs MFVI across temperatures
3. Compares Laplace vs Gaussian priors (for SVGD)
4. Analyzes entropy decomposition (aleatoric vs epistemic)
5. Generates publication-quality plots and tables

Usage:
    python visualize_comparison.py --save-dir ./results
    python visualize_comparison.py --save-dir ./results --output-dir ./figures
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


# =============================================================================
# Data Loading
# =============================================================================

def load_all_results(save_dir: str) -> pd.DataFrame:
    """
    Load all results into a single DataFrame.
    
    Columns:
        - step: 'deterministic', 'step2', 'step3'
        - method: 'deterministic', 'svgd', 'mfvi'
        - prior_type: 'laplace', 'gaussian', None (for MFVI/deterministic)
        - temperature: float or None
        - replicate: int
        - error, nll, ece, ood_auroc, ...
        - All entropy metrics
    """
    rows = []
    
    if not os.path.exists(save_dir):
        print(f"ERROR: Directory not found: {save_dir}")
        return pd.DataFrame()
    
    # Load deterministic model
    det_path = os.path.join(save_dir, "deterministic_model", "metrics.json")
    if os.path.exists(det_path):
        with open(det_path, 'r') as f:
            metrics = json.load(f)
        rows.append({
            'step': 'deterministic',
            'method': 'deterministic',
            'prior_type': None,
            'temperature': None,
            'replicate': 1,
            'run_name': 'deterministic_model',
            **metrics
        })
        print(f"Loaded: deterministic_model")
    
    # Scan for Step 2 and Step 3 runs
    for item in sorted(os.listdir(save_dir)):
        item_path = os.path.join(save_dir, item)
        results_path = os.path.join(item_path, "results.json")
        
        if not os.path.isdir(item_path):
            continue
        if not os.path.exists(results_path):
            continue
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Determine step
        if item.startswith("last_layer_"):
            step = "step2"
        elif item.startswith("joint_"):
            step = "step3"
        else:
            continue
        
        # Extract metrics
        metrics = data.get('metrics', {})
        
        row = {
            'step': step,
            'method': data.get('method'),
            'prior_type': data.get('prior_type'),  # May be None for old runs or MFVI
            'temperature': data.get('temperature'),
            'replicate': data.get('replicate'),
            'run_name': item,
            'training_time': data.get('training_time'),
            **metrics
        }
        rows.append(row)
        print(f"Loaded: {item}")
    
    df = pd.DataFrame(rows)
    
    # Summary
    print(f"\nLoaded {len(df)} total runs:")
    print(f"  Deterministic: {len(df[df['step'] == 'deterministic'])}")
    print(f"  Step 2: {len(df[df['step'] == 'step2'])}")
    print(f"  Step 3: {len(df[df['step'] == 'step3'])}")
    
    return df


def aggregate_results(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Aggregate results by specified columns, computing mean ± std.
    """
    if df.empty:
        return df
    
    # Numeric columns to aggregate
    numeric_cols = [
        'error', 'nll', 'ece', 'accuracy',
        'ood_auroc', 'ood_auroc_total', 'ood_auroc_aleatoric', 'ood_auroc_epistemic',
        'mean_in_total_entropy', 'mean_in_aleatoric_entropy', 'mean_in_epistemic_entropy',
        'mean_ood_total_entropy', 'mean_ood_aleatoric_entropy', 'mean_ood_epistemic_entropy',
        'training_time'
    ]
    
    # Filter to existing columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    agg_dict = {col: ['mean', 'std', 'count'] for col in numeric_cols}
    
    grouped = df.groupby(group_cols).agg(agg_dict)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    return grouped.reset_index()


# =============================================================================
# Comparison Tables
# =============================================================================

def create_main_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create main comparison table: Method × Temperature with key metrics.
    """
    # Filter to step2 for main comparison
    step2 = df[df['step'] == 'step2'].copy()
    
    if step2.empty:
        return pd.DataFrame()
    
    # Create method_prior column for SVGD
    step2['method_full'] = step2.apply(
        lambda r: f"{r['method']}_{r['prior_type']}" if r['prior_type'] else r['method'],
        axis=1
    )
    
    # Aggregate
    agg = aggregate_results(step2, ['method_full', 'temperature'])
    
    # Format table
    rows = []
    for _, row in agg.iterrows():
        n = int(row.get('error_count', row.get('nll_count', 1)))
        rows.append({
            'Method': row['method_full'],
            'Temperature': row['temperature'],
            'Error (%)': f"{row['error_mean']*100:.2f}±{row['error_std']*100:.2f}",
            'NLL': f"{row['nll_mean']:.4f}±{row['nll_std']:.4f}",
            'ECE': f"{row['ece_mean']:.4f}±{row['ece_std']:.4f}",
            'OOD AUROC': f"{row.get('ood_auroc_mean', 0):.4f}±{row.get('ood_auroc_std', 0):.4f}",
            'n': n
        })
    
    return pd.DataFrame(rows)


def create_entropy_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create entropy decomposition comparison table.
    """
    step2 = df[df['step'] == 'step2'].copy()
    
    if step2.empty:
        return pd.DataFrame()
    
    step2['method_full'] = step2.apply(
        lambda r: f"{r['method']}_{r['prior_type']}" if r['prior_type'] else r['method'],
        axis=1
    )
    
    agg = aggregate_results(step2, ['method_full', 'temperature'])
    
    rows = []
    for _, row in agg.iterrows():
        # Check if entropy metrics exist
        if 'mean_in_aleatoric_entropy_mean' not in row:
            continue
            
        rows.append({
            'Method': row['method_full'],
            'T': row['temperature'],
            'In-Dist Aleatoric': f"{row['mean_in_aleatoric_entropy_mean']:.4f}",
            'In-Dist Epistemic': f"{row['mean_in_epistemic_entropy_mean']:.4f}",
            'OOD Aleatoric': f"{row['mean_ood_aleatoric_entropy_mean']:.4f}",
            'OOD Epistemic': f"{row['mean_ood_epistemic_entropy_mean']:.4f}",
            'Epistemic Ratio': f"{row['mean_ood_epistemic_entropy_mean'] / (row['mean_in_epistemic_entropy_mean'] + 1e-10):.1f}x",
            'AUROC (Aleat)': f"{row.get('ood_auroc_aleatoric_mean', 0):.4f}",
            'AUROC (Epist)': f"{row.get('ood_auroc_epistemic_mean', 0):.4f}",
        })
    
    return pd.DataFrame(rows)


def create_prior_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Laplace vs Gaussian priors for SVGD.
    """
    svgd = df[(df['method'] == 'svgd') & (df['step'] == 'step2')].copy()
    
    if svgd.empty or 'prior_type' not in svgd.columns:
        return pd.DataFrame()
    
    agg = aggregate_results(svgd, ['prior_type', 'temperature'])
    
    # Pivot to compare side by side
    rows = []
    temps = sorted(agg['temperature'].unique())
    
    for temp in temps:
        row = {'Temperature': temp}
        for prior in ['laplace', 'gaussian']:
            prior_data = agg[(agg['prior_type'] == prior) & (agg['temperature'] == temp)]
            if not prior_data.empty:
                p = prior_data.iloc[0]
                row[f'{prior.title()} Error'] = f"{p['error_mean']*100:.2f}%"
                row[f'{prior.title()} NLL'] = f"{p['nll_mean']:.4f}"
                row[f'{prior.title()} OOD'] = f"{p.get('ood_auroc_mean', 0):.4f}"
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_method_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot SVGD vs MFVI comparison across temperatures.
    """
    if not HAS_MATPLOTLIB:
        return
    
    step2 = df[df['step'] == 'step2'].copy()
    if step2.empty:
        return
    
    # Get deterministic baseline
    det = df[df['step'] == 'deterministic']
    det_metrics = {}
    if not det.empty:
        det_metrics = det.iloc[0].to_dict()
    
    # Create method_full column
    step2['method_full'] = step2.apply(
        lambda r: f"SVGD ({r['prior_type']})" if r['method'] == 'svgd' and r['prior_type'] else 
                  ('MFVI' if r['method'] == 'mfvi' else r['method']),
        axis=1
    )
    
    metrics = [
        ('error', 'Error Rate', False),
        ('nll', 'NLL', False),
        ('ece', 'ECE', False),
        ('ood_auroc', 'OOD AUROC', True)  # True = higher is better
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {
        'SVGD (laplace)': 'blue',
        'SVGD (gaussian)': 'cyan',
        'MFVI': 'red'
    }
    markers = {
        'SVGD (laplace)': 'o',
        'SVGD (gaussian)': 's',
        'MFVI': '^'
    }
    
    for ax, (metric, label, higher_better) in zip(axes.flat, metrics):
        if metric not in step2.columns:
            continue
            
        for method_full in step2['method_full'].unique():
            method_data = step2[step2['method_full'] == method_full]
            
            agg = method_data.groupby('temperature')[metric].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(
                agg['temperature'],
                agg['mean'],
                yerr=agg['std'],
                label=method_full,
                color=colors.get(method_full, 'gray'),
                marker=markers.get(method_full, 'x'),
                capsize=3,
                linewidth=2,
                markersize=8
            )
        
        # Add deterministic baseline
        if metric in det_metrics and det_metrics[metric] is not None:
            ax.axhline(y=det_metrics[metric], color='black', linestyle='--', 
                      linewidth=2, label='Deterministic')
        
        ax.set_xscale('log')
        ax.set_xlabel('Temperature')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Method Comparison: SVGD vs MFVI', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: method_comparison.png")


def plot_entropy_decomposition(df: pd.DataFrame, output_dir: str):
    """
    Plot entropy decomposition: aleatoric vs epistemic for in-dist and OOD.
    """
    if not HAS_MATPLOTLIB:
        return
    
    step2 = df[df['step'] == 'step2'].copy()
    
    # Check if entropy columns exist
    if 'mean_in_aleatoric_entropy' not in step2.columns:
        print("No entropy decomposition data found, skipping plot")
        return
    
    step2['method_full'] = step2.apply(
        lambda r: f"SVGD ({r['prior_type']})" if r['method'] == 'svgd' and r['prior_type'] else 
                  ('MFVI' if r['method'] == 'mfvi' else r['method']),
        axis=1
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {'SVGD (laplace)': 'blue', 'SVGD (gaussian)': 'cyan', 'MFVI': 'red'}
    
    # Plot 1: In-dist entropy components
    ax = axes[0, 0]
    for method in step2['method_full'].unique():
        data = step2[step2['method_full'] == method]
        agg = data.groupby('temperature').agg({
            'mean_in_aleatoric_entropy': 'mean',
            'mean_in_epistemic_entropy': 'mean'
        }).reset_index()
        
        ax.plot(agg['temperature'], agg['mean_in_aleatoric_entropy'], 
                linestyle='-', label=f'{method} (Aleatoric)',
                color=colors.get(method, 'gray'), marker='o')
        ax.plot(agg['temperature'], agg['mean_in_epistemic_entropy'],
                linestyle='--', label=f'{method} (Epistemic)',
                color=colors.get(method, 'gray'), marker='s')
    
    ax.set_xscale('log')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Entropy')
    ax.set_title('In-Distribution Entropy Decomposition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: OOD entropy components
    ax = axes[0, 1]
    for method in step2['method_full'].unique():
        data = step2[step2['method_full'] == method]
        agg = data.groupby('temperature').agg({
            'mean_ood_aleatoric_entropy': 'mean',
            'mean_ood_epistemic_entropy': 'mean'
        }).reset_index()
        
        ax.plot(agg['temperature'], agg['mean_ood_aleatoric_entropy'],
                linestyle='-', label=f'{method} (Aleatoric)',
                color=colors.get(method, 'gray'), marker='o')
        ax.plot(agg['temperature'], agg['mean_ood_epistemic_entropy'],
                linestyle='--', label=f'{method} (Epistemic)',
                color=colors.get(method, 'gray'), marker='s')
    
    ax.set_xscale('log')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Entropy')
    ax.set_title('OOD Entropy Decomposition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Epistemic ratio (OOD / In-dist)
    ax = axes[1, 0]
    for method in step2['method_full'].unique():
        data = step2[step2['method_full'] == method]
        agg = data.groupby('temperature').agg({
            'mean_in_epistemic_entropy': 'mean',
            'mean_ood_epistemic_entropy': 'mean'
        }).reset_index()
        
        ratio = agg['mean_ood_epistemic_entropy'] / (agg['mean_in_epistemic_entropy'] + 1e-10)
        ax.plot(agg['temperature'], ratio, label=method,
                color=colors.get(method, 'gray'), marker='o', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Epistemic Ratio (OOD / In-Dist)')
    ax.set_title('Epistemic Uncertainty Amplification on OOD')
    ax.axhline(y=1, color='black', linestyle=':', label='No difference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: AUROC comparison (aleatoric vs epistemic)
    ax = axes[1, 1]
    if 'ood_auroc_aleatoric' in step2.columns and 'ood_auroc_epistemic' in step2.columns:
        for method in step2['method_full'].unique():
            data = step2[step2['method_full'] == method]
            agg = data.groupby('temperature').agg({
                'ood_auroc_aleatoric': 'mean',
                'ood_auroc_epistemic': 'mean'
            }).reset_index()
            
            ax.plot(agg['temperature'], agg['ood_auroc_aleatoric'],
                    linestyle='-', label=f'{method} (Aleatoric)',
                    color=colors.get(method, 'gray'), marker='o')
            ax.plot(agg['temperature'], agg['ood_auroc_epistemic'],
                    linestyle='--', label=f'{method} (Epistemic)',
                    color=colors.get(method, 'gray'), marker='s')
        
        ax.set_xscale('log')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('AUROC')
        ax.set_title('OOD Detection: Aleatoric vs Epistemic')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Entropy Decomposition Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: entropy_decomposition.png")


def plot_prior_comparison(df: pd.DataFrame, output_dir: str):
    """
    Compare Laplace vs Gaussian priors for SVGD.
    """
    if not HAS_MATPLOTLIB:
        return
    
    svgd = df[(df['method'] == 'svgd') & (df['step'] == 'step2')].copy()
    
    if svgd.empty or 'prior_type' not in svgd.columns:
        print("No prior comparison data found")
        return
    
    if svgd['prior_type'].nunique() < 2:
        print("Need both Laplace and Gaussian results for comparison")
        return
    
    metrics = [
        ('error', 'Error Rate'),
        ('nll', 'NLL'),
        ('ece', 'ECE'),
        ('ood_auroc', 'OOD AUROC')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for ax, (metric, label) in zip(axes.flat, metrics):
        if metric not in svgd.columns:
            continue
        
        for prior, color, marker in [('laplace', 'blue', 'o'), ('gaussian', 'orange', 's')]:
            data = svgd[svgd['prior_type'] == prior]
            if data.empty:
                continue
            
            agg = data.groupby('temperature')[metric].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(
                agg['temperature'],
                agg['mean'],
                yerr=agg['std'],
                label=prior.title(),
                color=color,
                marker=marker,
                capsize=3,
                linewidth=2,
                markersize=8
            )
        
        ax.set_xscale('log')
        ax.set_xlabel('Temperature')
        ax.set_ylabel(label)
        ax.set_title(f'{label}: Laplace vs Gaussian Prior')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('SVGD Prior Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prior_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: prior_comparison.png")


def plot_calibration_diagram(df: pd.DataFrame, output_dir: str):
    """
    Plot reliability/calibration comparison across methods.
    """
    if not HAS_MATPLOTLIB:
        return
    
    step2 = df[df['step'] == 'step2'].copy()
    if 'ece' not in step2.columns:
        return
    
    step2['method_full'] = step2.apply(
        lambda r: f"SVGD ({r['prior_type']})" if r['method'] == 'svgd' and r['prior_type'] else 
                  ('MFVI' if r['method'] == 'mfvi' else r['method']),
        axis=1
    )
    
    # Get deterministic baseline
    det = df[df['step'] == 'deterministic']
    det_ece = det.iloc[0]['ece'] if not det.empty and 'ece' in det.columns else None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'SVGD (laplace)': 'blue', 'SVGD (gaussian)': 'cyan', 'MFVI': 'red'}
    
    for method in step2['method_full'].unique():
        data = step2[step2['method_full'] == method]
        agg = data.groupby('temperature')['ece'].agg(['mean', 'std']).reset_index()
        
        ax.errorbar(
            agg['temperature'],
            agg['mean'],
            yerr=agg['std'],
            label=method,
            color=colors.get(method, 'gray'),
            marker='o',
            capsize=3,
            linewidth=2
        )
    
    if det_ece is not None:
        ax.axhline(y=det_ece, color='black', linestyle='--', linewidth=2, 
                  label=f'Deterministic ({det_ece:.4f})')
    
    ax.set_xscale('log')
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    ax.set_title('Calibration Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: calibration_comparison.png")


# =============================================================================
# Summary Printing
# =============================================================================

def print_summary(df: pd.DataFrame):
    """Print comprehensive text summary."""
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Deterministic baseline
    det = df[df['step'] == 'deterministic']
    if not det.empty:
        d = det.iloc[0]
        print("\n📊 BASELINE (Deterministic ResNet-18):")
        print(f"   Error:      {d.get('error', 0)*100:.2f}%")
        print(f"   NLL:        {d.get('nll', 0):.4f}")
        print(f"   ECE:        {d.get('ece', 0):.4f}")
        if d.get('ood_auroc') is not None:
            print(f"   OOD AUROC:  {d.get('ood_auroc'):.4f}")
    
    # Step 2 results
    step2 = df[df['step'] == 'step2']
    if not step2.empty:
        print("\n📊 STEP 2 (Bayesian Last Layer - Frozen Features):")
        
        # Group by method and prior
        step2['method_full'] = step2.apply(
            lambda r: f"{r['method']}_{r['prior_type']}" if r['prior_type'] else r['method'],
            axis=1
        )
        
        for method_full in sorted(step2['method_full'].unique()):
            method_data = step2[step2['method_full'] == method_full]
            
            print(f"\n   {method_full.upper()}:")
            
            # Best results
            if 'error' in method_data.columns:
                best_error_idx = method_data['error'].idxmin()
                best = method_data.loc[best_error_idx]
                print(f"     Best Error: {best['error']*100:.2f}% (T={best['temperature']})")
            
            if 'nll' in method_data.columns:
                best_nll_idx = method_data['nll'].idxmin()
                best = method_data.loc[best_nll_idx]
                print(f"     Best NLL:   {best['nll']:.4f} (T={best['temperature']})")
            
            if 'ece' in method_data.columns:
                best_ece_idx = method_data['ece'].idxmin()
                best = method_data.loc[best_ece_idx]
                print(f"     Best ECE:   {best['ece']:.4f} (T={best['temperature']})")
            
            if 'ood_auroc' in method_data.columns:
                best_ood_idx = method_data['ood_auroc'].idxmax()
                best = method_data.loc[best_ood_idx]
                print(f"     Best OOD:   {best['ood_auroc']:.4f} (T={best['temperature']})")
    
    # Entropy analysis
    if 'mean_in_epistemic_entropy' in df.columns and 'mean_ood_epistemic_entropy' in df.columns:
        print("\n📊 ENTROPY ANALYSIS:")
        
        step2 = df[df['step'] == 'step2']
        if not step2.empty:
            # Average across all runs
            mean_in_aleat = step2['mean_in_aleatoric_entropy'].mean()
            mean_in_epist = step2['mean_in_epistemic_entropy'].mean()
            mean_ood_aleat = step2['mean_ood_aleatoric_entropy'].mean()
            mean_ood_epist = step2['mean_ood_epistemic_entropy'].mean()
            
            print(f"   Average across all Step 2 runs:")
            print(f"     In-Dist Aleatoric:  {mean_in_aleat:.4f}")
            print(f"     In-Dist Epistemic:  {mean_in_epist:.4f}")
            print(f"     OOD Aleatoric:      {mean_ood_aleat:.4f}")
            print(f"     OOD Epistemic:      {mean_ood_epist:.4f}")
            print(f"     Epistemic Ratio:    {mean_ood_epist / (mean_in_epist + 1e-10):.1f}x")
            
            if 'ood_auroc_aleatoric' in step2.columns and 'ood_auroc_epistemic' in step2.columns:
                mean_auroc_aleat = step2['ood_auroc_aleatoric'].mean()
                mean_auroc_epist = step2['ood_auroc_epistemic'].mean()
                print(f"     AUROC (Aleatoric):  {mean_auroc_aleat:.4f}")
                print(f"     AUROC (Epistemic):  {mean_auroc_epist:.4f}")
    
    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize and compare BNN experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_comparison.py --save-dir ./results
    python visualize_comparison.py --save-dir ./results --output-dir ./figures
        """
    )
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files (default: {save-dir}/figures)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.save_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from: {args.save_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all results
    df = load_all_results(args.save_dir)
    
    if df.empty:
        print("\nERROR: No results found!")
        return
    
    # Print summary
    print_summary(df)
    
    # Save detailed CSV
    csv_path = os.path.join(output_dir, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results: {csv_path}")
    
    # Create comparison tables
    main_table = create_main_comparison_table(df)
    if not main_table.empty:
        table_path = os.path.join(output_dir, "main_comparison.csv")
        main_table.to_csv(table_path, index=False)
        print(f"Saved: {table_path}")
        print("\nMain Comparison Table:")
        print(main_table.to_string(index=False))
    
    entropy_table = create_entropy_comparison_table(df)
    if not entropy_table.empty:
        table_path = os.path.join(output_dir, "entropy_comparison.csv")
        entropy_table.to_csv(table_path, index=False)
        print(f"\nSaved: {table_path}")
    
    prior_table = create_prior_comparison_table(df)
    if not prior_table.empty:
        table_path = os.path.join(output_dir, "prior_comparison.csv")
        prior_table.to_csv(table_path, index=False)
        print(f"Saved: {table_path}")
        print("\nPrior Comparison Table:")
        print(prior_table.to_string(index=False))
    
    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        plot_method_comparison(df, output_dir)
        plot_entropy_decomposition(df, output_dir)
        plot_prior_comparison(df, output_dir)
        plot_calibration_diagram(df, output_dir)
    
    print(f"\n✓ Analysis complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
