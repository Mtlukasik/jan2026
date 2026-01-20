"""
Main Script for SVGD vs MFVI Last Layer Bayesian Inference Comparison.

This script runs the complete experiment comparing SVGD and MFVI for
last-layer Bayesian inference on CIFAR-10, following the methodology
in "Bayesian Neural Network Priors Revisited" (Fortuin et al., ICLR 2022).

Usage:
    python main.py --mode full          # Run full experiment
    python main.py --mode quick         # Run quick test
    python main.py --mode demo          # Generate demo plots
    python main.py --mode test          # Run all unit tests

Experiment Summary:
------------------
1. Network: 2-layer FCNN with BatchNorm and Dropout
2. Dataset: CIFAR-10 (in-distribution), SVHN (OOD)
3. Methods: SVGD (Stein Variational Gradient Descent) vs MFVI (Mean-Field VI) for last layer
4. Temperatures: 0.001, 0.01, 0.03, 0.1, 0.3, 1.0
5. Metrics: Error, NLL, ECE, OOD AUROC
6. Replicates: 3 per configuration for statistics

Key Findings Expected (based on paper):
- MFVI generally performs worse than SVGD
- Cold posteriors (T < 1) often improve performance
- The choice of prior affects the cold posterior effect
"""

import argparse
import sys
import os
import torch
import numpy as np
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ExperimentConfig, DataConfig, NetworkConfig, SVGDConfig, 
    MFVIConfig, TemperatureConfig, OODConfig, CalibrationConfig,
    DEVICE, SEED
)
from experiment_runner import ExperimentRunner, run_quick_experiment
from visualization import (
    ResultsPlotter, create_paper_style_figure, 
    convert_aggregated_results_to_plot_format, create_example_data
)


def run_full_experiment(
    num_replicates: int = 3,
    pretrain_epochs: int = 50,
    mfvi_epochs: int = 500,
    svgd_epochs: int = 200,
    save_dir: str = "./results"
):
    """Run the complete SVGD vs MFVI comparison experiment.
    
    This follows the paper's methodology:
    1. For each temperature and method:
       a. Train feature extractor with SGD
       b. Apply Bayesian inference to last layer
       c. Evaluate on all four metrics
    2. Aggregate results across 3 replicates for mean ± std
    3. Generate comparison figures
    """
    print("="*70)
    print("SVGD vs MFVI Last Layer Comparison on CIFAR-10")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Replicates: {num_replicates}")
    print(f"Pretrain epochs: {pretrain_epochs}")
    print(f"MFVI epochs: {mfvi_epochs}")
    print(f"SVGD epochs: {svgd_epochs}")
    print("="*70)
    
    # Full configuration
    config = ExperimentConfig(
        data=DataConfig(batch_size=128),
        network=NetworkConfig(
            hidden_dim=512,
            dropout_rate=0.2,
            use_batch_norm=True
        ),
        svgd=SVGDConfig(
            n_particles=20,
            num_epochs=svgd_epochs,
            svgd_lr=1e-3,
            feature_lr=1e-3,
            use_laplace_prior=True
        ),
        mfvi=MFVIConfig(
            num_epochs=mfvi_epochs,
            num_test_samples=10
        ),
        temperature=TemperatureConfig(
            temperatures=[0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
        ),
        num_replicates=num_replicates,
        save_dir=save_dir
    )
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Run all experiments
    print("\nRunning experiments...")
    results = runner.run_all_experiments(
        methods=["svgd", "mfvi"],
        pretrain_epochs=pretrain_epochs,
        mfvi_epochs=mfvi_epochs,
        svgd_epochs=svgd_epochs
    )
    
    # Save raw results
    results_path = os.path.join(save_dir, "raw_results.json")
    runner.save_results(results_path)
    
    # Aggregate results
    aggregated = runner.aggregate_results()
    
    # Convert to plotting format
    plot_data = convert_aggregated_results_to_plot_format(aggregated)
    
    # Separate SVGD and MFVI data for plotting
    svgd_data = plot_data.get("svgd", {})
    mfvi_data = plot_data.get("mfvi", {})
    
    # Generate figures
    print("\nGenerating figures...")
    plotter = ResultsPlotter()
    
    fig = plotter.plot_comparison_grid(
        svgd_data, mfvi_data,
        title="SVGD vs MFVI Last Layer Inference on CIFAR-10",
        save_path=os.path.join(save_dir, "figures", "comparison.png")
    )
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY (mean ± std over 3 replicates)")
    print("="*70)
    
    for method in ["svgd", "mfvi"]:
        print(f"\n{method.upper()} Results:")
        print("-" * 40)
        method_results = [a for a in aggregated if a.method == method]
        for agg in sorted(method_results, key=lambda x: x.temperature):
            print(f"  T={agg.temperature}:")
            for metric in ["error", "nll", "ece", "ood_auroc"]:
                mean = agg.metrics_mean[metric]
                std = agg.metrics_std[metric]
                print(f"    {metric}: {mean:.4f} ± {std:.4f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {save_dir}")
    print("="*70)
    
    return runner, aggregated


def run_unit_tests():
    """Run all unit tests for the project."""
    print("="*70)
    print("Running Unit Tests")
    print("="*70)
    
    test_modules = [
        ("config", None),  # Config doesn't have explicit tests
        ("data_loading", "data_loading"),
        ("networks", "networks"),
        ("metrics", "metrics"),
        ("svgd_inference", "svgd_inference"),
        ("mfvi_inference", "mfvi_inference"),
        ("experiment_runner", "experiment_runner"),
        ("visualization", "visualization"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, test_module in test_modules:
        if test_module is None:
            continue
        
        print(f"\n--- Testing {module_name} ---")
        try:
            module = __import__(test_module)
            
            # Find and run test functions
            for name in dir(module):
                if name.startswith("test_"):
                    func = getattr(module, name)
                    if callable(func):
                        try:
                            func()
                            passed += 1
                        except Exception as e:
                            print(f"✗ {name}: {e}")
                            failed += 1
        except Exception as e:
            print(f"✗ Failed to import {module_name}: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print("="*70)
    
    return failed == 0


def run_demo():
    """Generate demo figures with example data."""
    print("="*70)
    print("Generating Demo Figures")
    print("="*70)
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    mcmc_data, mfvi_data, sgd_baseline = create_example_data()
    
    os.makedirs("./results/figures", exist_ok=True)
    
    fig = create_paper_style_figure(
        mcmc_data, mfvi_data, sgd_baseline,
        save_path="./results/figures/demo_comparison.png"
    )
    
    plt.close(fig)
    
    print("\nDemo figure saved to ./results/figures/demo_comparison.png")
    print("\nThis figure shows the expected format of results:")
    print("- 4 panels: Error, NLL, ECE, OOD AUROC")
    print("- X-axis: Temperature (log scale)")
    print("- Lines: MCMC (blue), MFVI (orange), SGD baseline (dashed green)")
    print("- Shaded regions: Standard error")
    print("- Note: OOD AUROC y-axis is reversed (lower = better visually)")


def print_experiment_explanation():
    """Print detailed explanation of the experiment design."""
    explanation = """
================================================================================
                    EXPERIMENT DESIGN EXPLANATION
================================================================================

1. NETWORK ARCHITECTURE
   --------------------
   - Input: Flattened CIFAR-10 images (32x32x3 = 3072 dimensions)
   - Layer 1: Linear(3072 → 512) + BatchNorm + ReLU + Dropout(0.2)
   - Layer 2 (Last Layer): Linear(512 → 10) - This is where we apply BNN
   
   Why no convolutions? The task specifies a simple FCNN to isolate the 
   effect of Bayesian inference methods without confounding with architecture.

2. BAYESIAN INFERENCE METHODS
   ---------------------------
   
   MCMC (Stochastic Gradient MCMC):
   - Uses Langevin dynamics to sample from the posterior
   - Cyclical learning rate schedule for better exploration
   - Collects ~300 samples, discards first 50 as burn-in
   - Temperature appears as: p(w|D)^(1/T)
   
   MFVI (Mean-Field Variational Inference):
   - Approximates posterior with factorized Gaussian
   - Optimizes ELBO: E_q[log p(y|x,w)] - λ * KL(q||p)
   - Uses reparameterization trick for gradient estimation
   - Temperature (λ) appears in KL term weight

3. TEMPERATURE / COLD POSTERIOR EFFECT
   ------------------------------------
   - T = 1: Standard Bayesian posterior
   - T < 1: "Cold" posterior - more concentrated, often better performance
   - T > 1: "Warm" posterior - more spread, usually worse
   
   The paper investigates WHY cold posteriors work better - is it due to:
   - Misspecified prior?
   - Misspecified likelihood?
   - Data augmentation effects?

4. METRICS
   --------
   
   ERROR (Classification Error Rate):
   - (# wrong predictions) / (# total)
   - Lower is better
   
   NLL (Negative Log-Likelihood):
   - -log p(y|x) averaged over test set
   - Accounts for predictive uncertainty
   - Lower is better
   
   ECE (Expected Calibration Error):
   - Measures if confidence matches accuracy
   - |accuracy(bin) - confidence(bin)| averaged across bins
   - Lower is better (0 = perfectly calibrated)
   
   OOD AUROC (Out-of-Distribution Detection):
   - Uses predictive entropy to distinguish in-distribution vs OOD
   - AUROC for binary classification (CIFAR-10 vs SVHN)
   - Higher is better (but plotted inverted for visual consistency)

5. DATA SELECTION
   ---------------
   
   In-Distribution: CIFAR-10
   - 50,000 training images, 10,000 test images
   - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
   - Standard benchmark for image classification
   
   Out-of-Distribution: SVHN (Street View House Numbers)
   - Same image dimensions (32x32x3)
   - Semantically different (digits vs objects)
   - Not trivially separable by simple statistics
   - Commonly used as OOD benchmark for CIFAR-10
   
   Why SVHN? It's:
   - Same dimensionality (no resizing needed)
   - Different enough to be truly OOD
   - Challenging enough to reveal uncertainty quality

6. EXPECTED RESULTS (Based on Paper)
   -----------------------------------
   
   - MFVI generally worse than MCMC (especially at T=1)
   - Cold posteriors (T < 1) often improve error and NLL
   - Calibration and OOD detection may not follow same pattern
   - Heavy-tailed priors can reduce cold posterior effect in FCNNs
   
================================================================================
"""
    print(explanation)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCMC vs MFVI Last Layer Comparison on CIFAR-10"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["full", "quick", "demo", "test", "explain"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of replicates for full experiment"
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=100,
        help="Epochs for feature extractor pretraining"
    )
    parser.add_argument(
        "--mfvi-epochs",
        type=int,
        default=500,
        help="Epochs for MFVI training"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    if args.mode == "full":
        run_full_experiment(
            num_replicates=args.replicates,
            pretrain_epochs=args.pretrain_epochs,
            mfvi_epochs=args.mfvi_epochs,
            save_dir=args.save_dir
        )
    
    elif args.mode == "quick":
        print("Running quick experiment for testing...")
        run_quick_experiment()
    
    elif args.mode == "demo":
        run_demo()
    
    elif args.mode == "test":
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    elif args.mode == "explain":
        print_experiment_explanation()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
