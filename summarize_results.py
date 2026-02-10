"""
Summary and Analysis Script for BNN Experiment Results.

This script:
1. Scans the results directory for completed runs
2. Loads all results.json files
3. Aggregates metrics across replicates (mean ± std)
4. Generates comparison tables and plots
5. Exports summary to CSV and JSON

Usage:
    python summarize_results.py --save-dir ./results
    python summarize_results.py --save-dir ./results --output-dir ./analysis
    python summarize_results.py --save-dir ./results --format csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


def load_all_results(save_dir: str) -> Dict[str, List[Dict]]:
    """
    Load all results from completed runs.
    
    Returns:
        Dictionary with keys: 'deterministic', 'step2_svgd', 'step2_mfvi', 
                              'step3_svgd', 'step3_mfvi'
        Each value is a list of result dictionaries.
    """
    results = {
        'deterministic': [],
        'step2_svgd': [],
        'step2_mfvi': [],
        'step3_svgd': [],
        'step3_mfvi': []
    }
    
    if not os.path.exists(save_dir):
        print(f"ERROR: Directory not found: {save_dir}")
        return results
    
    # Load deterministic model
    det_dir = os.path.join(save_dir, "deterministic_model")
    if os.path.exists(os.path.join(det_dir, "metrics.json")):
        with open(os.path.join(det_dir, "metrics.json"), 'r') as f:
            det_metrics = json.load(f)
        det_metrics['method'] = 'deterministic'
        det_metrics['temperature'] = None
        det_metrics['replicate'] = 1
        results['deterministic'].append(det_metrics)
        print(f"Loaded: deterministic_model")
    
    # Scan for Step 2 and Step 3 runs
    for item in sorted(os.listdir(save_dir)):
        item_path = os.path.join(save_dir, item)
        results_path = os.path.join(item_path, "results.json")
        
        if not os.path.isdir(item_path):
            continue
        if not os.path.exists(results_path):
            continue
        
        # Load results
        with open(results_path, 'r') as f:
            run_results = json.load(f)
        
        # Categorize
        if item.startswith("last_layer_svgd_"):
            results['step2_svgd'].append(run_results)
            print(f"Loaded: {item}")
        elif item.startswith("last_layer_mfvi_"):
            results['step2_mfvi'].append(run_results)
            print(f"Loaded: {item}")
        elif item.startswith("joint_svgd_"):
            results['step3_svgd'].append(run_results)
            print(f"Loaded: {item}")
        elif item.startswith("joint_mfvi_"):
            results['step3_mfvi'].append(run_results)
            print(f"Loaded: {item}")
    
    # Print summary
    print(f"\nLoaded results:")
    print(f"  Deterministic: {len(results['deterministic'])}")
    print(f"  Step 2 SVGD: {len(results['step2_svgd'])}")
    print(f"  Step 2 MFVI: {len(results['step2_mfvi'])}")
    print(f"  Step 3 SVGD: {len(results['step3_svgd'])}")
    print(f"  Step 3 MFVI: {len(results['step3_mfvi'])}")
    
    return results


def aggregate_by_temperature(runs: List[Dict]) -> pd.DataFrame:
    """
    Aggregate runs by temperature, computing mean ± std across replicates.
    
    Returns:
        DataFrame with columns: temperature, error_mean, error_std, nll_mean, nll_std, etc.
    """
    if not runs:
        return pd.DataFrame()
    
    # Group by temperature
    by_temp = defaultdict(list)
    for run in runs:
        temp = run.get('temperature', 0)
        metrics = run.get('metrics', run)  # Handle both formats
        by_temp[temp].append(metrics)
    
    # Aggregate
    rows = []
    for temp in sorted(by_temp.keys()):
        metrics_list = by_temp[temp]
        
        row = {'temperature': temp, 'n_replicates': len(metrics_list)}
        
        for metric in ['error', 'nll', 'ece', 'ood_auroc', 'accuracy']:
            values = [m.get(metric) for m in metrics_list if m.get(metric) is not None]
            if values:
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_comparison_table(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Create a comparison table of all methods and temperatures.
    """
    all_rows = []
    
    # Deterministic baseline
    if results['deterministic']:
        det = results['deterministic'][0]
        metrics = det.get('metrics', det)
        ood_auroc = metrics.get('ood_auroc')
        all_rows.append({
            'step': 'Baseline',
            'method': 'Deterministic',
            'temperature': '-',
            'error': f"{metrics.get('error', 0)*100:.2f}%",
            'nll': f"{metrics.get('nll', 0):.4f}",
            'ece': f"{metrics.get('ece', 0):.4f}",
            'ood_auroc': f"{ood_auroc:.4f}" if ood_auroc is not None else "-",
            'n': 1
        })
    
    # Step 2
    for method, key in [('SVGD', 'step2_svgd'), ('MFVI', 'step2_mfvi')]:
        df = aggregate_by_temperature(results[key])
        for _, row in df.iterrows():
            all_rows.append({
                'step': 'Step 2',
                'method': method,
                'temperature': row['temperature'],
                'error': f"{row.get('error_mean', 0)*100:.2f}±{row.get('error_std', 0)*100:.2f}%",
                'nll': f"{row.get('nll_mean', 0):.4f}±{row.get('nll_std', 0):.4f}",
                'ece': f"{row.get('ece_mean', 0):.4f}±{row.get('ece_std', 0):.4f}",
                'ood_auroc': f"{row.get('ood_auroc_mean', 0):.4f}±{row.get('ood_auroc_std', 0):.4f}",
                'n': row['n_replicates']
            })
    
    # Step 3
    for method, key in [('SVGD', 'step3_svgd'), ('MFVI', 'step3_mfvi')]:
        df = aggregate_by_temperature(results[key])
        for _, row in df.iterrows():
            all_rows.append({
                'step': 'Step 3',
                'method': method,
                'temperature': row['temperature'],
                'error': f"{row.get('error_mean', 0)*100:.2f}±{row.get('error_std', 0)*100:.2f}%",
                'nll': f"{row.get('nll_mean', 0):.4f}±{row.get('nll_std', 0):.4f}",
                'ece': f"{row.get('ece_mean', 0):.4f}±{row.get('ece_std', 0):.4f}",
                'ood_auroc': f"{row.get('ood_auroc_mean', 0):.4f}±{row.get('ood_auroc_std', 0):.4f}",
                'n': row['n_replicates']
            })
    
    return pd.DataFrame(all_rows)


def create_detailed_csv(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Create a detailed CSV with all individual runs.
    """
    rows = []
    
    # Deterministic
    if results['deterministic']:
        det = results['deterministic'][0]
        metrics = det.get('metrics', det)
        rows.append({
            'step': 'baseline',
            'method': 'deterministic',
            'temperature': None,
            'replicate': 1,
            'error': metrics.get('error'),
            'accuracy': metrics.get('accuracy', 1 - metrics.get('error', 0)),
            'nll': metrics.get('nll'),
            'ece': metrics.get('ece'),
            'ood_auroc': metrics.get('ood_auroc'),
            'training_time': det.get('training_time')
        })
    
    # Step 2 & 3
    for step, methods in [('step2', ['svgd', 'mfvi']), ('step3', ['svgd', 'mfvi'])]:
        for method in methods:
            key = f'{step}_{method}'
            for run in results[key]:
                metrics = run.get('metrics', {})
                rows.append({
                    'step': step,
                    'method': method,
                    'temperature': run.get('temperature'),
                    'replicate': run.get('replicate'),
                    'error': metrics.get('error'),
                    'accuracy': metrics.get('accuracy', 1 - metrics.get('error', 0) if metrics.get('error') else None),
                    'nll': metrics.get('nll'),
                    'ece': metrics.get('ece'),
                    'ood_auroc': metrics.get('ood_auroc'),
                    'training_time': run.get('training_time')
                })
    
    return pd.DataFrame(rows)


def plot_metric_vs_temperature(
    results: Dict[str, List[Dict]], 
    metric: str,
    output_path: str,
    title: str = None
):
    """
    Plot a metric vs temperature for SVGD and MFVI (Step 2 and Step 3).
    Includes deterministic baseline as horizontal dashed line.
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metric_labels = {
        'error': 'Error Rate',
        'nll': 'Negative Log-Likelihood',
        'ece': 'Expected Calibration Error',
        'ood_auroc': 'OOD Detection AUROC'
    }
    
    # Get deterministic baseline value
    det_value = None
    if results['deterministic']:
        det = results['deterministic'][0]
        det_metrics = det.get('metrics', det)
        det_value = det_metrics.get(metric)
    
    # Step 2
    ax = axes[0]
    for method, key, color, marker in [('SVGD', 'step2_svgd', 'blue', 'o'), 
                                        ('MFVI', 'step2_mfvi', 'red', 's')]:
        df = aggregate_by_temperature(results[key])
        if not df.empty and f'{metric}_mean' in df.columns:
            ax.errorbar(
                df['temperature'], 
                df[f'{metric}_mean'],
                yerr=df[f'{metric}_std'],
                label=method, 
                color=color, 
                marker=marker,
                capsize=3,
                linewidth=2,
                markersize=8
            )
    
    # Add deterministic baseline
    if det_value is not None:
        ax.axhline(y=det_value, color='black', linestyle='--', linewidth=2, 
                   label=f'Deterministic ({det_value:.4f})')
    
    ax.set_xscale('log')
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title('Step 2: Frozen Features', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Step 3
    ax = axes[1]
    for method, key, color, marker in [('SVGD', 'step3_svgd', 'blue', 'o'), 
                                        ('MFVI', 'step3_mfvi', 'red', 's')]:
        df = aggregate_by_temperature(results[key])
        if not df.empty and f'{metric}_mean' in df.columns:
            ax.errorbar(
                df['temperature'], 
                df[f'{metric}_mean'],
                yerr=df[f'{metric}_std'],
                label=method, 
                color=color, 
                marker=marker,
                capsize=3,
                linewidth=2,
                markersize=8
            )
    
    # Add deterministic baseline
    if det_value is not None:
        ax.axhline(y=det_value, color='black', linestyle='--', linewidth=2,
                   label=f'Deterministic ({det_value:.4f})')
    
    ax.set_xscale('log')
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title('Step 3: Joint Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title or f'{metric_labels.get(metric, metric)} vs Temperature', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_comparison_grid(results: Dict[str, List[Dict]], output_path: str):
    """
    Create a 2x2 grid comparing all metrics with deterministic baseline.
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics = [('error', 'Error Rate'), ('nll', 'NLL'), 
               ('ece', 'ECE'), ('ood_auroc', 'OOD AUROC')]
    
    # Get deterministic baseline values
    det_values = {}
    if results['deterministic']:
        det = results['deterministic'][0]
        det_metrics = det.get('metrics', det)
        for metric, _ in metrics:
            det_values[metric] = det_metrics.get(metric)
    
    for ax, (metric, label) in zip(axes.flat, metrics):
        # Plot Step 2 and Step 3 results
        for method, key, color, marker, ls in [
            ('SVGD (Step2)', 'step2_svgd', 'blue', 'o', '-'),
            ('MFVI (Step2)', 'step2_mfvi', 'red', 's', '-'),
            ('SVGD (Step3)', 'step3_svgd', 'blue', '^', '--'),
            ('MFVI (Step3)', 'step3_mfvi', 'red', 'v', '--')
        ]:
            df = aggregate_by_temperature(results[key])
            if not df.empty and f'{metric}_mean' in df.columns:
                ax.errorbar(
                    df['temperature'], 
                    df[f'{metric}_mean'],
                    yerr=df[f'{metric}_std'],
                    label=method, 
                    color=color, 
                    marker=marker,
                    linestyle=ls,
                    capsize=3,
                    linewidth=2,
                    markersize=6
                )
        
        # Add deterministic baseline
        if metric in det_values and det_values[metric] is not None:
            ax.axhline(y=det_values[metric], color='black', linestyle=':', linewidth=2,
                       label=f'Deterministic')
        
        ax.set_xscale('log')
        ax.set_xlabel('Temperature')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('SVGD vs MFVI: All Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def print_summary(results: Dict[str, List[Dict]]):
    """Print a text summary of results."""
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Deterministic baseline
    if results['deterministic']:
        det = results['deterministic'][0]
        metrics = det.get('metrics', det)
        print("\n📊 BASELINE (Deterministic ResNet-18):")
        print(f"   Error:      {metrics.get('error', 0)*100:.2f}%")
        print(f"   Accuracy:   {(1-metrics.get('error', 0))*100:.2f}%")
        print(f"   NLL:        {metrics.get('nll', 0):.4f}")
        print(f"   ECE:        {metrics.get('ece', 0):.4f}")
        if metrics.get('ood_auroc') is not None:
            print(f"   OOD AUROC:  {metrics.get('ood_auroc'):.4f} (using Shannon entropy)")
        if metrics.get('mean_in_entropy') is not None:
            print(f"   Mean In-Dist Entropy:  {metrics.get('mean_in_entropy'):.4f}")
            print(f"   Mean OOD Entropy:      {metrics.get('mean_ood_entropy'):.4f}")
    
    # Best results for each method
    for step_name, step_keys in [("Step 2 (Frozen Features)", ['step2_svgd', 'step2_mfvi']),
                                  ("Step 3 (Joint Training)", ['step3_svgd', 'step3_mfvi'])]:
        print(f"\n📊 {step_name}:")
        
        for key in step_keys:
            runs = results[key]
            if not runs:
                continue
            
            method = 'SVGD' if 'svgd' in key else 'MFVI'
            df = aggregate_by_temperature(runs)
            
            if df.empty:
                continue
            
            # Find best temperature for each metric
            best_error_idx = df['error_mean'].idxmin() if 'error_mean' in df.columns else None
            best_nll_idx = df['nll_mean'].idxmin() if 'nll_mean' in df.columns else None
            best_ece_idx = df['ece_mean'].idxmin() if 'ece_mean' in df.columns else None
            best_ood_idx = df['ood_auroc_mean'].idxmax() if 'ood_auroc_mean' in df.columns else None
            
            print(f"\n   {method}:")
            if best_error_idx is not None:
                row = df.loc[best_error_idx]
                print(f"     Best Error:    {row['error_mean']*100:.2f}±{row['error_std']*100:.2f}% (T={row['temperature']})")
            if best_nll_idx is not None:
                row = df.loc[best_nll_idx]
                print(f"     Best NLL:      {row['nll_mean']:.4f}±{row['nll_std']:.4f} (T={row['temperature']})")
            if best_ece_idx is not None:
                row = df.loc[best_ece_idx]
                print(f"     Best ECE:      {row['ece_mean']:.4f}±{row['ece_std']:.4f} (T={row['temperature']})")
            if best_ood_idx is not None:
                row = df.loc[best_ood_idx]
                print(f"     Best OOD AUROC: {row['ood_auroc_mean']:.4f}±{row['ood_auroc_std']:.4f} (T={row['temperature']})")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize BNN experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Print summary to console
    python summarize_results.py --save-dir ./results
    
    # Save to specific output directory
    python summarize_results.py --save-dir ./results --output-dir ./analysis
    
    # Export only CSV (no plots)
    python summarize_results.py --save-dir ./results --no-plots
        """
    )
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save summary files (default: {save-dir}/summary)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.save_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from: {args.save_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all results
    results = load_all_results(args.save_dir)
    
    # Check if we have any results
    total_runs = sum(len(v) for v in results.values())
    if total_runs == 0:
        print("\nERROR: No results found!")
        return
    
    # Print summary
    print_summary(results)
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    if not comparison_df.empty:
        table_path = os.path.join(output_dir, "comparison_table.csv")
        comparison_df.to_csv(table_path, index=False)
        print(f"\nSaved comparison table: {table_path}")
        print("\nComparison Table:")
        print(comparison_df.to_string(index=False))
    
    # Create detailed CSV
    detailed_df = create_detailed_csv(results)
    if not detailed_df.empty:
        detailed_path = os.path.join(output_dir, "detailed_results.csv")
        detailed_df.to_csv(detailed_path, index=False)
        print(f"\nSaved detailed results: {detailed_path}")
    
    # Save raw aggregated data as JSON
    summary_json = {
        'generated_at': datetime.now().isoformat(),
        'save_dir': args.save_dir,
        'n_runs': {k: len(v) for k, v in results.items()},
        'step2_svgd': aggregate_by_temperature(results['step2_svgd']).to_dict('records'),
        'step2_mfvi': aggregate_by_temperature(results['step2_mfvi']).to_dict('records'),
        'step3_svgd': aggregate_by_temperature(results['step3_svgd']).to_dict('records'),
        'step3_mfvi': aggregate_by_temperature(results['step3_mfvi']).to_dict('records'),
    }
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved summary JSON: {json_path}")
    
    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        
        for metric in ['error', 'nll', 'ece', 'ood_auroc']:
            plot_path = os.path.join(output_dir, f"{metric}_vs_temperature.png")
            plot_metric_vs_temperature(results, metric, plot_path)
        
        grid_path = os.path.join(output_dir, "comparison_grid.png")
        plot_comparison_grid(results, grid_path)
    
    print(f"\n✓ Summary complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
