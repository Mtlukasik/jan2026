"""
Visualization Module for BNN Comparison Results.

This module generates plots similar to those in the paper (Figures 4, 5, A.33):
- Line plots showing metrics vs temperature
- Error bars representing standard error across replicates
- Comparison of MCMC vs MFVI across all four metrics
- Reversed y-axis for OOD AUROC (so lower = better visually for all plots)

Plot Layout (following the paper):
- 4 columns: Error, Likelihood (NLL), Calibration (ECE), OOD Detection (AUROC)
- Different colored lines for different methods/priors
- Log scale x-axis for temperature
- Shaded regions for standard error
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

# Try to use seaborn style if available
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    figure_width: float = 16
    figure_height: float = 4
    dpi: int = 150
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha_fill: float = 0.2
    font_size_title: int = 12
    font_size_label: int = 10
    font_size_tick: int = 9
    font_size_legend: int = 9


# Color scheme for different methods
METHOD_COLORS = {
    "mcmc": "#1f77b4",       # Blue
    "mfvi": "#ff7f0e",       # Orange
    "sgd": "#2ca02c",        # Green (baseline)
    "gaussian": "#1f77b4",   # Blue
    "laplace": "#ff7f0e",    # Orange
    "student-t": "#2ca02c",  # Green
    "correlated": "#d62728", # Red
}

METHOD_MARKERS = {
    "mcmc": "o",
    "mfvi": "s",
    "sgd": "--",
    "gaussian": "o",
    "laplace": "s",
    "student-t": "^",
    "correlated": "d",
}

METRIC_LABELS = {
    "error": "Error",
    "nll": "NLL",
    "ece": "ECE",
    "ood_auroc": "OOD AUROC"
}


class ResultsPlotter:
    """Class for generating publication-quality plots of experiment results."""
    
    def __init__(self, config: PlotConfig = PlotConfig()):
        self.config = config
    
    def _setup_axes(self, ax: plt.Axes, metric: str, temperatures: List[float]):
        """Set up a single axes for a metric."""
        ax.set_xlabel("Temperature", fontsize=self.config.font_size_label)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=self.config.font_size_label)
        ax.set_xscale("log")
        
        # Set x-ticks to temperature values
        ax.set_xticks(temperatures)
        ax.set_xticklabels([f"{t:.0e}" if t < 0.01 else f"{t}" for t in temperatures])
        ax.tick_params(labelsize=self.config.font_size_tick)
        
        # Reverse y-axis for OOD AUROC (so lower appears better like other metrics)
        if metric == "ood_auroc":
            ax.invert_yaxis()
        
        ax.grid(True, alpha=0.3)
    
    def plot_metrics_vs_temperature(
        self,
        results: Dict[str, Dict[float, Tuple[float, float]]],
        metrics: List[str] = ["error", "nll", "ece", "ood_auroc"],
        title: str = "",
        save_path: Optional[str] = None,
        sgd_baseline: Optional[Dict[str, float]] = None
    ) -> plt.Figure:
        """Plot metrics vs temperature for multiple methods.
        
        Args:
            results: Nested dict: results[method][temperature] = (mean, std)
                    for each metric in metrics
            metrics: List of metric names to plot
            title: Overall figure title
            save_path: Path to save figure (optional)
            sgd_baseline: Optional dict of SGD baseline values per metric
            
        Returns:
            matplotlib Figure object
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, num_metrics,
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi
        )
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Get temperatures (assuming same for all methods)
            first_method = list(results.keys())[0]
            temperatures = sorted(results[first_method][metric].keys())
            
            for method in results.keys():
                if metric not in results[method]:
                    continue
                
                temps = sorted(results[method][metric].keys())
                means = [results[method][metric][t][0] for t in temps]
                stds = [results[method][metric][t][1] for t in temps]
                
                color = METHOD_COLORS.get(method, "gray")
                marker = METHOD_MARKERS.get(method, "o")
                
                # Plot line with markers
                ax.plot(
                    temps, means,
                    color=color,
                    marker=marker,
                    markersize=self.config.marker_size,
                    linewidth=self.config.line_width,
                    label=method.upper()
                )
                
                # Add shaded error region (standard error)
                means = np.array(means)
                stds = np.array(stds)
                ax.fill_between(
                    temps,
                    means - stds,
                    means + stds,
                    color=color,
                    alpha=self.config.alpha_fill
                )
            
            # Add SGD baseline if provided
            if sgd_baseline is not None and metric in sgd_baseline:
                ax.axhline(
                    y=sgd_baseline[metric],
                    color=METHOD_COLORS["sgd"],
                    linestyle="--",
                    linewidth=self.config.line_width,
                    label="SGD"
                )
            
            self._setup_axes(ax, metric, temperatures)
            ax.legend(fontsize=self.config.font_size_legend)
        
        if title:
            fig.suptitle(title, fontsize=self.config.font_size_title, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_comparison_grid(
        self,
        mcmc_results: Dict[str, Dict[float, Tuple[float, float]]],
        mfvi_results: Dict[str, Dict[float, Tuple[float, float]]],
        sgd_baseline: Optional[Dict[str, float]] = None,
        title: str = "MCMC vs MFVI Last Layer Inference on CIFAR-10",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of MCMC vs MFVI in paper style.
        
        This generates a figure similar to Figures 4, 5, and A.33 in the paper.
        
        Args:
            mcmc_results: MCMC results by metric and temperature
            mfvi_results: MFVI results by metric and temperature  
            sgd_baseline: Optional SGD baseline values
            title: Figure title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        metrics = ["error", "nll", "ece", "ood_auroc"]
        
        fig, axes = plt.subplots(
            1, 4,
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi
        )
        
        # Get temperatures
        first_metric = list(mcmc_results.keys())[0]
        temperatures = sorted(mcmc_results[first_metric].keys())
        
        for ax, metric in zip(axes, metrics):
            # Plot MCMC
            if metric in mcmc_results:
                temps = sorted(mcmc_results[metric].keys())
                mcmc_means = [mcmc_results[metric][t][0] for t in temps]
                mcmc_stds = [mcmc_results[metric][t][1] for t in temps]
                
                ax.plot(
                    temps, mcmc_means,
                    color=METHOD_COLORS["mcmc"],
                    marker=METHOD_MARKERS["mcmc"],
                    markersize=self.config.marker_size,
                    linewidth=self.config.line_width,
                    label="MCMC"
                )
                
                mcmc_means = np.array(mcmc_means)
                mcmc_stds = np.array(mcmc_stds)
                ax.fill_between(
                    temps,
                    mcmc_means - mcmc_stds,
                    mcmc_means + mcmc_stds,
                    color=METHOD_COLORS["mcmc"],
                    alpha=self.config.alpha_fill
                )
            
            # Plot MFVI
            if metric in mfvi_results:
                temps = sorted(mfvi_results[metric].keys())
                mfvi_means = [mfvi_results[metric][t][0] for t in temps]
                mfvi_stds = [mfvi_results[metric][t][1] for t in temps]
                
                ax.plot(
                    temps, mfvi_means,
                    color=METHOD_COLORS["mfvi"],
                    marker=METHOD_MARKERS["mfvi"],
                    markersize=self.config.marker_size,
                    linewidth=self.config.line_width,
                    label="MFVI"
                )
                
                mfvi_means = np.array(mfvi_means)
                mfvi_stds = np.array(mfvi_stds)
                ax.fill_between(
                    temps,
                    mfvi_means - mfvi_stds,
                    mfvi_means + mfvi_stds,
                    color=METHOD_COLORS["mfvi"],
                    alpha=self.config.alpha_fill
                )
            
            # Add SGD baseline
            if sgd_baseline is not None and metric in sgd_baseline:
                ax.axhline(
                    y=sgd_baseline[metric],
                    color=METHOD_COLORS["sgd"],
                    linestyle="--",
                    linewidth=self.config.line_width,
                    label="SGD"
                )
            
            self._setup_axes(ax, metric, temperatures)
            ax.legend(fontsize=self.config.font_size_legend, loc='best')
            ax.set_title(METRIC_LABELS[metric], fontsize=self.config.font_size_title)
        
        fig.suptitle(title, fontsize=self.config.font_size_title + 2, y=1.05)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_calibration_vs_rotation(
        self,
        results: Dict[str, Dict[float, Tuple[float, float]]],
        title: str = "Calibration under Distribution Shift",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot ECE vs rotation angle for calibration analysis.
        
        Args:
            results: results[method][rotation_angle] = (ece_mean, ece_std)
            title: Figure title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(
            figsize=(8, 5),
            dpi=self.config.dpi
        )
        
        for method in results.keys():
            angles = sorted(results[method].keys())
            means = [results[method][a][0] for a in angles]
            stds = [results[method][a][1] for a in angles]
            
            color = METHOD_COLORS.get(method, "gray")
            
            ax.plot(
                angles, means,
                color=color,
                marker="o",
                markersize=self.config.marker_size,
                linewidth=self.config.line_width,
                label=method.upper()
            )
            
            means = np.array(means)
            stds = np.array(stds)
            ax.fill_between(
                angles,
                means - stds,
                means + stds,
                color=color,
                alpha=self.config.alpha_fill
            )
        
        ax.set_xlabel("Rotation Angle (degrees)", fontsize=self.config.font_size_label)
        ax.set_ylabel("ECE", fontsize=self.config.font_size_label)
        ax.set_title(title, fontsize=self.config.font_size_title)
        ax.legend(fontsize=self.config.font_size_legend)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig


def convert_aggregated_results_to_plot_format(
    aggregated_results: List,  # List[AggregatedResults]
) -> Dict[str, Dict[str, Dict[float, Tuple[float, float]]]]:
    """Convert AggregatedResults to plotting format.
    
    Args:
        aggregated_results: List of AggregatedResults from experiment
        
    Returns:
        Nested dict: results[method][metric][temperature] = (mean, std)
    """
    plot_data = {}
    
    for agg in aggregated_results:
        method = agg.method
        temp = agg.temperature
        
        if method not in plot_data:
            plot_data[method] = {}
        
        for metric in agg.metrics_mean.keys():
            if metric not in plot_data[method]:
                plot_data[method][metric] = {}
            
            plot_data[method][metric][temp] = (
                agg.metrics_mean[metric],
                agg.metrics_std[metric]
            )
    
    return plot_data


def create_paper_style_figure(
    mcmc_data: Dict[str, Dict[float, Tuple[float, float]]],
    mfvi_data: Dict[str, Dict[float, Tuple[float, float]]],
    sgd_baseline: Optional[Dict[str, float]] = None,
    save_path: str = "./results/figures/comparison.png"
) -> plt.Figure:
    """Create a figure in the style of the paper's Figure 5 / A.33.
    
    Args:
        mcmc_data: MCMC results - mcmc_data[metric][temp] = (mean, std)
        mfvi_data: MFVI results - mfvi_data[metric][temp] = (mean, std)
        sgd_baseline: Optional baseline values
        save_path: Where to save the figure
        
    Returns:
        matplotlib Figure
    """
    plotter = ResultsPlotter()
    return plotter.plot_comparison_grid(
        mcmc_data, mfvi_data, sgd_baseline,
        title="MCMC vs MFVI Last Layer Inference on CIFAR-10",
        save_path=save_path
    )


# ============================================================================
# EXAMPLE USAGE AND TESTS
# ============================================================================

def create_example_data():
    """Create example data for testing visualization."""
    temperatures = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    
    # Simulated MCMC results (metrics improve with lower temperature, 
    # mimicking cold posterior effect)
    mcmc_data = {
        "error": {t: (0.15 - 0.03 * np.log10(1/t + 1) + np.random.normal(0, 0.005), 0.01) 
                  for t in temperatures},
        "nll": {t: (0.5 - 0.1 * np.log10(1/t + 1) + np.random.normal(0, 0.02), 0.03) 
                for t in temperatures},
        "ece": {t: (0.1 - 0.02 * np.log10(1/t + 1) + np.random.normal(0, 0.01), 0.01) 
                for t in temperatures},
        "ood_auroc": {t: (0.85 + 0.05 * np.log10(1/t + 1) + np.random.normal(0, 0.01), 0.02) 
                      for t in temperatures},
    }
    
    # Simulated MFVI results (generally worse than MCMC, as per paper)
    mfvi_data = {
        "error": {t: (0.18 - 0.02 * np.log10(1/t + 1) + np.random.normal(0, 0.005), 0.015) 
                  for t in temperatures},
        "nll": {t: (0.6 - 0.08 * np.log10(1/t + 1) + np.random.normal(0, 0.02), 0.04) 
                for t in temperatures},
        "ece": {t: (0.15 - 0.02 * np.log10(1/t + 1) + np.random.normal(0, 0.01), 0.015) 
                for t in temperatures},
        "ood_auroc": {t: (0.75 + 0.03 * np.log10(1/t + 1) + np.random.normal(0, 0.01), 0.025) 
                      for t in temperatures},
    }
    
    sgd_baseline = {
        "error": 0.12,
        "nll": 0.45,
        "ece": 0.08,
        "ood_auroc": 0.80
    }
    
    return mcmc_data, mfvi_data, sgd_baseline


def test_plotting():
    """Test the plotting functionality."""
    mcmc_data, mfvi_data, sgd_baseline = create_example_data()
    
    plotter = ResultsPlotter()
    
    # Test comparison grid
    fig = plotter.plot_comparison_grid(
        mcmc_data, mfvi_data, sgd_baseline,
        title="Test: MCMC vs MFVI",
        save_path="./results/test_comparison.png"
    )
    plt.close(fig)
    
    print("✓ Comparison grid plot test passed")
    
    # Test single method plot
    combined_results = {"mcmc": mcmc_data}
    fig = plotter.plot_metrics_vs_temperature(
        combined_results,
        title="Test: MCMC Results"
    )
    plt.close(fig)
    
    print("✓ Single method plot test passed")


def generate_demo_figure():
    """Generate a demo figure showing expected results."""
    mcmc_data, mfvi_data, sgd_baseline = create_example_data()
    
    fig = create_paper_style_figure(
        mcmc_data, mfvi_data, sgd_baseline,
        save_path="./results/figures/demo_comparison.png"
    )
    
    print("Demo figure generated!")
    return fig


if __name__ == "__main__":
    test_plotting()
    
    # Generate demo figure
    fig = generate_demo_figure()
    plt.show()
    
    print("\nAll visualization tests passed!")
