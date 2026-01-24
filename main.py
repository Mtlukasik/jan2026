"""
Main Script for SVGD vs MFVI Last Layer Bayesian Inference Comparison.

Enhanced version with:
- Training/validation loss curves plotted after each model
- Individual results saved to JSON with full hyperparameters
- Google Drive integration for automatic backup
- Comprehensive logging

Usage:
    python main_enhanced.py --mode full          # Run full experiment
    python main_enhanced.py --mode quick         # Run quick test
    python main_enhanced.py --mode test          # Run all unit tests
    
    # With Google Drive (in Colab):
    python main_enhanced.py --mode full --save-to-drive --drive-path "/content/drive/MyDrive/bnn_results"
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
from experiment_runner_enhanced import EnhancedExperimentRunner, run_quick_experiment_enhanced


def run_full_experiment(
    num_replicates: int = 3,
    pretrain_epochs: int = 50,
    mfvi_epochs: int = 500,
    svgd_epochs: int = 200,
    save_dir: str = "./results",
    save_to_drive: bool = False,
    drive_path: str = "/content/drive/MyDrive/bnn_experiments"
):
    """Run the complete SVGD vs MFVI comparison experiment with full tracking."""
    print("="*70)
    print("SVGD vs MFVI Last Layer Comparison on CIFAR-10")
    print("Enhanced Version with Training Curves & Individual Saving")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Replicates: {num_replicates}")
    print(f"Pretrain epochs: {pretrain_epochs}")
    print(f"MFVI epochs: {mfvi_epochs}")
    print(f"SVGD epochs: {svgd_epochs}")
    print(f"Save directory: {save_dir}")
    if save_to_drive:
        print(f"Google Drive path: {drive_path}")
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
    
    # Create enhanced experiment runner
    runner = EnhancedExperimentRunner(
        config,
        save_dir=save_dir,
        save_to_drive=save_to_drive,
        drive_path=drive_path
    )
    
    # Run all experiments
    print("\nRunning experiments...")
    results = runner.run_all_experiments(
        methods=["svgd", "mfvi"],
        pretrain_epochs=pretrain_epochs,
        mfvi_epochs=mfvi_epochs,
        svgd_epochs=svgd_epochs
    )
    
    # Generate final comparison plot
    print("\nGenerating final comparison figures...")
    runner.plot_final_comparison()
    
    # Print summary
    aggregated = runner.aggregate_results()
    
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
    if save_to_drive:
        print(f"Also saved to Google Drive: {drive_path}")
    print("="*70)
    
    return runner, aggregated


def run_unit_tests():
    """Run all unit tests for the project."""
    print("="*70)
    print("Running Unit Tests")
    print("="*70)
    
    test_modules = [
        ("config", None),
        ("data_loading", "data_loading"),
        ("networks", "networks"),
        ("metrics", "metrics"),
        ("svgd_inference", "svgd_inference"),
        ("mfvi_inference", "mfvi_inference"),
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


def main():
    parser = argparse.ArgumentParser(
        description="SVGD vs MFVI Last Layer Comparison (Enhanced)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "quick", "test"],
        default="quick",
        help="Experiment mode"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of replicates for statistics"
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Epochs for feature extractor pretraining"
    )
    parser.add_argument(
        "--mfvi-epochs",
        type=int,
        default=500,
        help="Epochs for MFVI training"
    )
    parser.add_argument(
        "--svgd-epochs",
        type=int,
        default=200,
        help="Epochs for SVGD training"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--save-to-drive",
        action="store_true",
        help="Save results to Google Drive (for Colab)"
    )
    parser.add_argument(
        "--drive-path",
        type=str,
        default="/content/drive/MyDrive/bnn_experiments",
        help="Google Drive path for saving results"
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
            svgd_epochs=args.svgd_epochs,
            save_dir=args.save_dir,
            save_to_drive=args.save_to_drive,
            drive_path=args.drive_path
        )
    
    elif args.mode == "quick":
        print("Running quick experiment for testing...")
        run_quick_experiment_enhanced()
    
    elif args.mode == "test":
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
