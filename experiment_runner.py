"""
Main Experiment Runner for SVGD vs MFVI Comparison.

This module orchestrates the full experiment:
1. Data loading and preparation
2. Model training (feature extractor pretraining + Bayesian last layer)
3. Evaluation across temperatures
4. Result collection and aggregation
5. Statistical analysis across multiple replicates

Experiment Design (following the paper):
---------------------------------------
1. Train feature extractor with standard SGD
2. Freeze feature extractor
3. Apply Bayesian inference (SVGD or MFVI) to last layer only
4. Evaluate across temperatures T ∈ {0.001, 0.01, 0.03, 0.1, 0.3, 1.0}
5. Compute four metrics: Error, NLL, ECE, OOD AUROC
6. Repeat with 3 random seeds for statistical significance (mean ± std)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import copy
import time

from config import (
    ExperimentConfig, DataConfig, NetworkConfig, MCMCConfig, 
    MFVIConfig, SVGDConfig, TemperatureConfig, OODConfig, CalibrationConfig,
    DEVICE, SEED
)
from data_loading import DataLoaderManager
from networks import FullNetwork
from metrics import MetricsComputer, MetricResults
from mfvi_inference import MFVITrainer
from svgd_inference import SVGDTrainer, SVGDEnsemble


@dataclass
class ExperimentResult:
    """Container for single experiment run results."""
    method: str  # "svgd" or "mfvi"
    temperature: float
    replicate: int
    metrics: Dict[str, float]
    training_time: float  # seconds


@dataclass
class AggregatedResults:
    """Container for aggregated results across replicates."""
    method: str
    temperature: float
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    num_replicates: int


class ExperimentRunner:
    """Main class for running SVGD vs MFVI comparison experiments.
    
    This class handles:
    - Data loading and management
    - Running experiments with different methods and temperatures
    - Computing metrics (Error, NLL, ECE, OOD AUROC)
    - Aggregating results across replicates for statistical significance
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = DEVICE
        
        # Initialize data loaders
        self.data_manager = DataLoaderManager(
            config.data,
            config.ood,
            config.calibration,
            flatten=True
        )
        
        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(
            num_bins=config.calibration.num_bins,
            device=self.device
        )
        
        # Results storage
        self.results: List[ExperimentResult] = []
    
    def _create_model(self, last_layer_type: str) -> FullNetwork:
        """Create a new model instance."""
        return FullNetwork(
            self.config.network,
            last_layer_type=last_layer_type,
            prior_log_var=self.config.mfvi.prior_log_var
        ).to(self.device)
    
    def run_svgd_experiment(
        self,
        temperature: float,
        replicate: int,
        pretrain_epochs: int = 50,
        svgd_epochs: int = 200
    ) -> ExperimentResult:
        """Run a single SVGD experiment.
        
        Args:
            temperature: Posterior temperature
            replicate: Replicate index (for different random seeds)
            pretrain_epochs: Epochs for feature extractor pretraining
            svgd_epochs: Epochs for SVGD training
            
        Returns:
            ExperimentResult with all metrics
        """
        # Set random seed
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        print(f"\n{'='*60}")
        print(f"SVGD Experiment: T={temperature}, Replicate={replicate+1}/3")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create SVGD trainer
        trainer = SVGDTrainer(
            self.config.network,
            self.config.svgd,
            device=self.device
        )
        
        # Pretrain feature extractor
        print("Pretraining feature extractor...")
        val_acc = trainer.pretrain_feature_extractor(
            self.data_manager.train_loader,
            self.data_manager.test_loader,
            num_epochs=pretrain_epochs
        )
        print(f"Pretrain validation accuracy: {val_acc:.4f}")
        
        # Run SVGD training
        print(f"Running SVGD with temperature T={temperature}...")
        trainer.train(
            self.data_manager.train_loader,
            temperature=temperature,
            num_epochs=svgd_epochs
        )
        
        # Evaluate using SVGD ensemble predictions
        print("Evaluating...")
        metrics = self._evaluate_svgd(trainer)
        
        training_time = time.time() - start_time
        
        result = ExperimentResult(
            method="svgd",
            temperature=temperature,
            replicate=replicate,
            metrics=metrics,
            training_time=training_time
        )
        
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        
        return result
    
    def _evaluate_svgd(self, trainer: SVGDTrainer) -> Dict[str, float]:
        """Evaluate SVGD ensemble on all metrics.
        
        Args:
            trainer: Trained SVGD trainer
            
        Returns:
            Dictionary of metric values
        """
        # Compute predictions on test set
        all_probs = []
        all_labels = []
        
        for batch_x, batch_y in self.data_manager.test_loader:
            batch_x = batch_x.to(self.device)
            probs = trainer.predict_proba(batch_x)
            all_probs.append(probs.cpu())
            all_labels.append(batch_y)
        
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Error
        preds = all_probs.argmax(dim=-1)
        error = (preds != all_labels).float().mean().item()
        
        # NLL
        nll = F.cross_entropy(torch.log(all_probs + 1e-10), all_labels).item()
        
        # ECE
        ece = self._compute_ece(all_probs, all_labels)
        
        # OOD AUROC
        ood_auroc = self._compute_ood_auroc_svgd(trainer)
        
        return {
            "error": error,
            "nll": nll,
            "ece": ece,
            "ood_auroc": ood_auroc
        }
    
    def _compute_ece(
        self, 
        probs: torch.Tensor, 
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """Compute Expected Calibration Error."""
        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == labels).float()
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += torch.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return ece.item()
    
    def _compute_ood_auroc_svgd(self, trainer: SVGDTrainer) -> float:
        """Compute OOD detection AUROC using predictive entropy."""
        # In-distribution entropy
        in_dist_entropy = []
        for batch_x, _ in self.data_manager.test_loader:
            batch_x = batch_x.to(self.device)
            _, _, entropy = trainer.ensemble.predict_with_uncertainty(batch_x)
            in_dist_entropy.append(entropy.cpu())
        in_dist_entropy = torch.cat(in_dist_entropy)
        
        # OOD entropy
        ood_entropy = []
        for batch_x, _ in self.data_manager.ood_loader:
            batch_x = batch_x.to(self.device)
            _, _, entropy = trainer.ensemble.predict_with_uncertainty(batch_x)
            ood_entropy.append(entropy.cpu())
        ood_entropy = torch.cat(ood_entropy)
        
        # Compute AUROC (higher entropy -> OOD)
        from sklearn.metrics import roc_auc_score
        
        labels = np.concatenate([
            np.zeros(len(in_dist_entropy)),  # In-distribution
            np.ones(len(ood_entropy))         # OOD
        ])
        scores = np.concatenate([
            in_dist_entropy.numpy(),
            ood_entropy.numpy()
        ])
        
        return roc_auc_score(labels, scores)
    
    def run_mfvi_experiment(
        self,
        temperature: float,
        replicate: int,
        pretrain_epochs: int = 50,
        mfvi_epochs: int = 500
    ) -> ExperimentResult:
        """Run a single MFVI experiment.
        
        Args:
            temperature: Temperature (λ) for KL term
            replicate: Replicate index
            pretrain_epochs: Epochs for feature extractor pretraining
            mfvi_epochs: Epochs for MFVI training
            
        Returns:
            ExperimentResult with all metrics
        """
        # Set random seed
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        print(f"\n{'='*60}")
        print(f"MFVI Experiment: λ={temperature}, Replicate={replicate+1}/3")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model
        model = self._create_model("mfvi")
        
        # Create trainer
        trainer = MFVITrainer(
            model,
            self.config.mfvi,
            device=self.device
        )
        
        # Pretrain feature extractor
        print("Pretraining feature extractor...")
        val_acc = trainer.pretrain_feature_extractor(
            self.data_manager.train_loader,
            self.data_manager.test_loader,
            num_epochs=pretrain_epochs
        )
        print(f"Pretrain validation accuracy: {val_acc:.4f}")
        
        # Train MFVI
        print(f"Training MFVI with λ={temperature}...")
        trainer.train(
            self.data_manager.train_loader,
            temperature=temperature,
            num_epochs=mfvi_epochs
        )
        
        # Evaluate
        print("Evaluating...")
        metrics = self.metrics_computer.compute_all_metrics(
            model,
            self.data_manager.test_loader,
            self.data_manager.ood_loader,
            num_samples=self.config.mfvi.num_test_samples,
            is_mcmc=False
        )
        
        training_time = time.time() - start_time
        
        result = ExperimentResult(
            method="mfvi",
            temperature=temperature,
            replicate=replicate,
            metrics=metrics.to_dict(),
            training_time=training_time
        )
        
        print(f"Results: Error={metrics.error:.4f}, NLL={metrics.nll:.4f}, "
              f"ECE={metrics.ece:.4f}, OOD AUROC={metrics.ood_auroc:.4f}")
        
        return result
    
    def run_all_experiments(
        self,
        methods: List[str] = ["svgd", "mfvi"],
        temperatures: Optional[List[float]] = None,
        num_replicates: Optional[int] = None,
        pretrain_epochs: int = 50,
        mfvi_epochs: int = 500,
        svgd_epochs: int = 200
    ) -> List[ExperimentResult]:
        """Run all experiment configurations.
        
        Args:
            methods: List of methods to test ("svgd" and/or "mfvi")
            temperatures: List of temperatures (uses config default if None)
            num_replicates: Number of replicates (uses config default if None)
            pretrain_epochs: Epochs for pretraining
            mfvi_epochs: Epochs for MFVI training
            svgd_epochs: Epochs for SVGD training
            
        Returns:
            List of all experiment results
        """
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        self.results = []
        
        total_experiments = len(methods) * len(temperatures) * num_replicates
        current_exp = 0
        
        print(f"\n{'#'*70}")
        print(f"Starting {total_experiments} experiments")
        print(f"Methods: {methods}")
        print(f"Temperatures: {temperatures}")
        print(f"Replicates per configuration: {num_replicates}")
        print(f"{'#'*70}")
        
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    current_exp += 1
                    print(f"\n[Experiment {current_exp}/{total_experiments}]")
                    
                    if method == "svgd":
                        result = self.run_svgd_experiment(
                            temp, rep, pretrain_epochs, svgd_epochs
                        )
                    elif method == "mfvi":
                        result = self.run_mfvi_experiment(
                            temp, rep, pretrain_epochs, mfvi_epochs
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}. Choose from ['svgd', 'mfvi']")
                    
                    self.results.append(result)
        
        return self.results
    
    def aggregate_results(self) -> List[AggregatedResults]:
        """Aggregate results across replicates.
        
        Returns:
            List of aggregated results per method and temperature
        """
        aggregated = []
        
        # Group by method and temperature
        groups: Dict[Tuple[str, float], List[ExperimentResult]] = {}
        for result in self.results:
            key = (result.method, result.temperature)
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        # Compute statistics
        for (method, temp), results in groups.items():
            metrics_arrays = {key: [] for key in results[0].metrics.keys()}
            
            for result in results:
                for key, value in result.metrics.items():
                    metrics_arrays[key].append(value)
            
            metrics_mean = {key: np.mean(vals) for key, vals in metrics_arrays.items()}
            metrics_std = {key: np.std(vals) for key, vals in metrics_arrays.items()}
            
            aggregated.append(AggregatedResults(
                method=method,
                temperature=temp,
                metrics_mean=metrics_mean,
                metrics_std=metrics_std,
                num_replicates=len(results)
            ))
        
        return aggregated
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        data = {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> List[ExperimentResult]:
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = [
            ExperimentResult(**r) for r in data["results"]
        ]
        return self.results


def run_quick_experiment():
    """Run a quick experiment for testing purposes."""
    # Reduced configuration for testing
    config = ExperimentConfig(
        data=DataConfig(batch_size=64),
        network=NetworkConfig(hidden_dim=256),
        svgd=SVGDConfig(
            n_particles=5,
            num_epochs=50
        ),
        mfvi=MFVIConfig(num_epochs=100),
        temperature=TemperatureConfig(temperatures=[0.1, 1.0]),
        num_replicates=3  # Always use 3 replicates for statistics
    )
    
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments(
        methods=["mfvi"],  # MFVI is faster for testing
        pretrain_epochs=10,
        mfvi_epochs=50,
        svgd_epochs=30
    )
    
    aggregated = runner.aggregate_results()
    
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (mean ± std over 3 replicates)")
    print("="*60)
    for agg in aggregated:
        print(f"\n{agg.method.upper()}, Temperature={agg.temperature}")
        for metric in ["error", "nll", "ece", "ood_auroc"]:
            mean = agg.metrics_mean[metric]
            std = agg.metrics_std[metric]
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    return runner, results, aggregated


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_experiment_result():
    """Test ExperimentResult dataclass."""
    result = ExperimentResult(
        method="svgd",
        temperature=1.0,
        replicate=0,
        metrics={"error": 0.1, "nll": 0.5, "ece": 0.05, "ood_auroc": 0.9},
        training_time=100.0
    )
    
    assert result.method == "svgd"
    assert result.temperature == 1.0
    assert result.metrics["error"] == 0.1
    
    print("✓ ExperimentResult test passed")


def test_aggregated_results():
    """Test result aggregation with 3 replicates."""
    results = [
        ExperimentResult("svgd", 1.0, 0, {"error": 0.10}, 100.0),
        ExperimentResult("svgd", 1.0, 1, {"error": 0.12}, 100.0),
        ExperimentResult("svgd", 1.0, 2, {"error": 0.14}, 100.0),
    ]
    
    # Create dummy runner
    config = ExperimentConfig()
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.results = results
    
    aggregated = runner.aggregate_results()
    
    assert len(aggregated) == 1
    agg = aggregated[0]
    assert agg.method == "svgd"
    assert agg.temperature == 1.0
    assert agg.num_replicates == 3
    assert abs(agg.metrics_mean["error"] - 0.12) < 0.01
    # Std should be ~0.0163
    assert agg.metrics_std["error"] > 0
    
    print("✓ Aggregated results test passed")


def test_experiment_runner_initialization():
    """Test ExperimentRunner initialization."""
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    
    assert runner.device is not None
    assert runner.data_manager is not None
    assert runner.metrics_computer is not None
    
    print("✓ ExperimentRunner initialization test passed")


def test_model_creation():
    """Test model creation in runner."""
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    
    # Test MFVI model
    mfvi_model = runner._create_model("mfvi")
    assert hasattr(mfvi_model.last_layer, 'kl_divergence')
    
    print("✓ Model creation test passed")


def test_svgd_trainer_creation():
    """Test SVGD trainer can be created."""
    config = NetworkConfig()
    svgd_config = SVGDConfig(n_particles=3)
    
    trainer = SVGDTrainer(config, svgd_config, DEVICE)
    assert trainer.ensemble is not None
    assert len(trainer.ensemble.particles) == 3
    
    print("✓ SVGD trainer creation test passed")


if __name__ == "__main__":
    test_experiment_result()
    test_aggregated_results()
    test_experiment_runner_initialization()
    test_model_creation()
    test_svgd_trainer_creation()
    
    print("\n" + "="*60)
    print("Running quick experiment...")
    print("="*60)
    # Uncomment to run actual experiment:
    # run_quick_experiment()
    
    print("\nAll experiment runner tests passed!")
