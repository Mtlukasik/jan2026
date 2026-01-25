"""
Step-by-Step Training Pipeline for SVGD vs MFVI Comparison.

This module provides a manual step-by-step process:

STEP 1: Train deterministic model
    - Trains full network (feature extractor + last layer) with SGD
    - Saves weights to: {save_dir}/deterministic_model/
    - Saves performance metrics
    
STEP 2: Train Bayesian last layers
    - Loads deterministic model weights
    - Freezes feature extractor
    - Trains Bayesian last layer (SVGD or MFVI)
    - Runs all 36 configurations (2 methods × 6 temperatures × 3 replicates)
    - Skips already completed runs

Directory Structure:
    {save_dir}/
    ├── deterministic_model/
    │   ├── model_weights.pt          # Saved model weights
    │   ├── training_history.json     # Loss/accuracy per epoch
    │   ├── metrics.json              # Final test metrics
    │   ├── training_curves.png       # Loss/accuracy plots
    │   └── COMPLETED                 # Marker file
    │
    ├── last_layer_svgd_T0.001_replicate_1/
    │   ├── results.json
    │   ├── hyperparameters.json
    │   ├── training_curves.png
    │   └── COMPLETED
    │
    ├── last_layer_mfvi_T0.001_replicate_1/
    │   └── ...
    │
    └── aggregated_results.json

Usage:
    # Step 1: Train deterministic model
    python pipeline.py --step 1 --save-dir /path/to/results
    
    # Step 2: Train all Bayesian models (loads from step 1)
    python pipeline.py --step 2 --save-dir /path/to/results
    
    # Check status
    python pipeline.py --status --save-dir /path/to/results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from config import (
    ExperimentConfig, DataConfig, NetworkConfig,
    MFVIConfig, SVGDConfig, TemperatureConfig,
    DEVICE, SEED
)
from data_loading import DataLoaderManager
from networks import FullNetwork, FeatureExtractor, DeterministicLastLayer
from metrics import MetricsComputer
from mfvi_inference import MFVITrainer
from svgd_inference import SVGDTrainer


# =============================================================================
# STEP 1: Deterministic Model Training
# =============================================================================

class DeterministicTrainer:
    """Trainer for the deterministic baseline model."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        save_dir: str,
        device: str = None
    ):
        self.config = config
        self.save_dir = save_dir
        self.device = device or DEVICE
        self.model_dir = os.path.join(save_dir, "deterministic_model")
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize data
        self.data_manager = DataLoaderManager(
            config.data,
            config.ood,
            config.calibration,
            flatten=True
        )
        
        # Initialize metrics
        self.metrics_computer = MetricsComputer(
            num_bins=config.calibration.num_bins,
            device=self.device
        )
        
        # Create model
        self.model = FullNetwork(
            config.network,
            last_layer_type="deterministic"
        ).to(self.device)
        
        # Training history
        self.history = {
            "epochs": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
    
    def is_completed(self) -> bool:
        """Check if deterministic training is already done."""
        return os.path.exists(os.path.join(self.model_dir, "COMPLETED"))
    
    def get_weights_path(self) -> str:
        """Get path to saved weights."""
        return os.path.join(self.model_dir, "model_weights.pt")
    
    def train(
        self,
        num_epochs: int = 100,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        val_frequency: int = 5
    ) -> Dict:
        """Train the deterministic model.
        
        Args:
            num_epochs: Number of training epochs
            lr: Learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization
            val_frequency: How often to compute validation metrics
            
        Returns:
            Dictionary with final metrics
        """
        if self.is_completed():
            print(f"[SKIP] Deterministic model already trained at {self.model_dir}")
            return self.load_metrics()
        
        print("="*60)
        print("STEP 1: Training Deterministic Model")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Save directory: {self.model_dir}")
        print("="*60)
        
        start_time = time.time()
        
        # Optimizer and scheduler
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = F.cross_entropy(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_y)
                train_correct += (output.argmax(dim=-1) == batch_y).sum().item()
                train_total += len(batch_y)
            
            scheduler.step()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Record history
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            
            # Validation
            if (epoch + 1) % val_frequency == 0 or epoch == num_epochs - 1:
                val_loss, val_acc = self._evaluate()
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_weights()
            else:
                # Interpolate for history
                if self.history["val_loss"]:
                    self.history["val_loss"].append(self.history["val_loss"][-1])
                    self.history["val_accuracy"].append(self.history["val_accuracy"][-1])
        
        training_time = time.time() - start_time
        
        # Final evaluation
        print("\nFinal Evaluation...")
        self._load_weights()  # Load best weights
        final_metrics = self._compute_final_metrics()
        final_metrics["training_time"] = training_time
        final_metrics["best_val_accuracy"] = best_val_acc
        
        # Save everything
        self._save_history()
        self._save_metrics(final_metrics)
        self._plot_training_curves()
        self._mark_completed()
        
        print(f"\n✓ Deterministic model training complete!")
        print(f"  Error: {final_metrics['error']:.4f}")
        print(f"  NLL: {final_metrics['nll']:.4f}")
        print(f"  ECE: {final_metrics['ece']:.4f}")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Saved to: {self.model_dir}")
        
        return final_metrics
    
    def _evaluate(self) -> Tuple[float, float]:
        """Compute validation loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x)
                loss = F.cross_entropy(output, batch_y)
                
                total_loss += loss.item() * len(batch_y)
                total_correct += (output.argmax(dim=-1) == batch_y).sum().item()
                total_samples += len(batch_y)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def _compute_final_metrics(self) -> Dict[str, float]:
        """Compute all final metrics."""
        self.model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                probs = F.softmax(self.model(batch_x), dim=-1)
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
        
        return {
            "error": error,
            "nll": nll,
            "ece": ece,
            "accuracy": 1 - error
        }
    
    def _compute_ece(self, probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
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
    
    def _save_weights(self):
        """Save model weights."""
        torch.save(self.model.state_dict(), self.get_weights_path())
    
    def _load_weights(self):
        """Load model weights."""
        self.model.load_state_dict(torch.load(self.get_weights_path(), map_location=self.device))
    
    def _save_history(self):
        """Save training history."""
        path = os.path.join(self.model_dir, "training_history.json")
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _save_metrics(self, metrics: Dict):
        """Save final metrics."""
        path = os.path.join(self.model_dir, "metrics.json")
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_metrics(self) -> Dict:
        """Load saved metrics."""
        path = os.path.join(self.model_dir, "metrics.json")
        with open(path, 'r') as f:
            return json.load(f)
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Deterministic Model Training', fontsize=14, fontweight='bold')
        
        epochs = self.history["epochs"]
        
        # Loss
        axes[0].plot(epochs, self.history["train_loss"], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"], 'r--', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history["train_accuracy"], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history["val_accuracy"], 'r--', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _mark_completed(self):
        """Mark training as complete."""
        path = os.path.join(self.model_dir, "COMPLETED")
        with open(path, 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")


# =============================================================================
# STEP 2: Bayesian Last Layer Training
# =============================================================================

class BayesianLastLayerTrainer:
    """Trainer for Bayesian last layer models using pretrained features."""
    
    RUN_NAME_TEMPLATE = "last_layer_{method}_T{temperature}_replicate_{replicate}"
    
    def __init__(
        self,
        config: ExperimentConfig,
        save_dir: str,
        device: str = None
    ):
        self.config = config
        self.save_dir = save_dir
        self.device = device or DEVICE
        self.deterministic_dir = os.path.join(save_dir, "deterministic_model")
        
        # Check deterministic model exists
        if not self._deterministic_model_exists():
            raise RuntimeError(
                f"No deterministic model found at {self.deterministic_dir}. "
                "Run Step 1 first: python pipeline.py --step 1"
            )
        
        # Initialize data
        self.data_manager = DataLoaderManager(
            config.data,
            config.ood,
            config.calibration,
            flatten=True
        )
        
        # Initialize metrics
        self.metrics_computer = MetricsComputer(
            num_bins=config.calibration.num_bins,
            device=self.device
        )
        
        # Load deterministic weights
        self.deterministic_weights = self._load_deterministic_weights()
        
        # Scan completed runs
        self.completed_runs = self._scan_completed_runs()
    
    def _deterministic_model_exists(self) -> bool:
        """Check if deterministic model is trained."""
        weights_path = os.path.join(self.deterministic_dir, "model_weights.pt")
        completed_path = os.path.join(self.deterministic_dir, "COMPLETED")
        return os.path.exists(weights_path) and os.path.exists(completed_path)
    
    def _load_deterministic_weights(self) -> Dict:
        """Load pretrained deterministic model weights."""
        weights_path = os.path.join(self.deterministic_dir, "model_weights.pt")
        weights = torch.load(weights_path, map_location=self.device)
        print(f"Loaded deterministic weights from: {weights_path}")
        return weights
    
    def _get_feature_extractor_weights(self) -> Dict:
        """Extract only feature extractor weights."""
        return {
            k: v for k, v in self.deterministic_weights.items()
            if k.startswith("feature_extractor.")
        }
    
    @staticmethod
    def get_run_name(method: str, temperature: float, replicate: int) -> str:
        """Generate consistent run name."""
        return BayesianLastLayerTrainer.RUN_NAME_TEMPLATE.format(
            method=method,
            temperature=temperature,
            replicate=replicate + 1
        )
    
    def _get_run_dir(self, method: str, temperature: float, replicate: int) -> str:
        """Get full path to run directory."""
        run_name = self.get_run_name(method, temperature, replicate)
        return os.path.join(self.save_dir, run_name)
    
    def _is_run_completed(self, method: str, temperature: float, replicate: int) -> bool:
        """Check if a run is completed."""
        run_dir = self._get_run_dir(method, temperature, replicate)
        return os.path.exists(os.path.join(run_dir, "COMPLETED"))
    
    def _scan_completed_runs(self) -> Set[str]:
        """Scan for completed runs."""
        completed = set()
        if os.path.exists(self.save_dir):
            for item in os.listdir(self.save_dir):
                if item.startswith("last_layer_"):
                    completed_marker = os.path.join(self.save_dir, item, "COMPLETED")
                    if os.path.exists(completed_marker):
                        completed.add(item)
        return completed
    
    def _mark_completed(self, method: str, temperature: float, replicate: int):
        """Mark run as completed."""
        run_dir = self._get_run_dir(method, temperature, replicate)
        with open(os.path.join(run_dir, "COMPLETED"), 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")
        self.completed_runs.add(self.get_run_name(method, temperature, replicate))
    
    def get_pending_runs(
        self,
        methods: List[str] = ["svgd", "mfvi"],
        temperatures: List[float] = None,
        num_replicates: int = None
    ) -> List[Tuple[str, float, int]]:
        """Get list of pending runs."""
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        pending = []
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    if not self._is_run_completed(method, temp, rep):
                        pending.append((method, temp, rep))
        return pending
    
    def train_all(
        self,
        methods: List[str] = ["svgd", "mfvi"],
        temperatures: List[float] = None,
        num_replicates: int = None,
        svgd_epochs: int = 200,
        mfvi_epochs: int = 500
    ):
        """Train all pending Bayesian last layer models."""
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        # Get pending runs
        pending = self.get_pending_runs(methods, temperatures, num_replicates)
        total_runs = len(methods) * len(temperatures) * num_replicates
        completed_count = total_runs - len(pending)
        
        print("="*60)
        print("STEP 2: Training Bayesian Last Layer Models")
        print("="*60)
        print(f"Using pretrained features from: {self.deterministic_dir}")
        print(f"Total configurations: {total_runs}")
        print(f"Already completed: {completed_count}")
        print(f"Pending: {len(pending)}")
        print("="*60)
        
        if completed_count > 0:
            print("\nCompleted runs:")
            for name in sorted(self.completed_runs):
                if name.startswith("last_layer_"):
                    print(f"  ✓ {name}")
        
        if len(pending) == 0:
            print("\n✓ All Bayesian models already trained!")
            return
        
        print("\nPending runs:")
        for method, temp, rep in pending:
            print(f"  ○ {self.get_run_name(method, temp, rep)}")
        
        print(f"\n{'='*60}")
        print(f"Starting training...")
        print(f"{'='*60}")
        
        for i, (method, temp, rep) in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}]")
            
            if method == "svgd":
                self._train_svgd(temp, rep, svgd_epochs)
            elif method == "mfvi":
                self._train_mfvi(temp, rep, mfvi_epochs)
        
        print("\n" + "="*60)
        print("✓ Step 2 complete!")
        print("="*60)
    
    def _train_svgd(self, temperature: float, replicate: int, num_epochs: int = 200):
        """Train SVGD model with frozen feature extractor."""
        run_name = self.get_run_name("svgd", temperature, replicate)
        run_dir = self._get_run_dir("svgd", temperature, replicate)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        start_time = time.time()
        
        # Create SVGD trainer
        trainer = SVGDTrainer(
            self.config.network,
            self.config.svgd,
            device=self.device
        )
        
        # Load pretrained feature extractor weights into all particles
        feature_weights = self._get_feature_extractor_weights()
        for particle in trainer.ensemble.particles:
            # Map weights (remove 'feature_extractor.' prefix)
            particle_state = particle.state_dict()
            for key, value in feature_weights.items():
                # Convert from FullNetwork format to SVGDParticle format
                new_key = key.replace("feature_extractor.", "feature_extractor.")
                if new_key in particle_state:
                    particle_state[new_key] = value
            particle.load_state_dict(particle_state)
            
            # Freeze feature extractor
            for param in particle.feature_extractor.parameters():
                param.requires_grad = False
            
            # Add small perturbation to last layer for diversity
            with torch.no_grad():
                particle.last_layer.weight.add_(
                    torch.randn_like(particle.last_layer.weight) * 0.01
                )
                particle.last_layer.bias.add_(
                    torch.randn_like(particle.last_layer.bias) * 0.01
                )
        
        print("Feature extractor loaded and frozen")
        
        # Training with tracking
        history = {"epochs": [], "train_loss": [], "train_accuracy": [], 
                   "val_loss": [], "val_accuracy": [], "diversity": []}
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_diversity = 0.0
            epoch_correct = 0
            epoch_total = 0
            num_batches = 0
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                loss, diversity = trainer.ensemble.train_step(batch_x, batch_y, temperature)
                
                epoch_loss += loss
                epoch_diversity += diversity
                num_batches += 1
                
                with torch.no_grad():
                    mean_probs, _, _ = trainer.ensemble.predict_with_uncertainty(batch_x)
                    preds = mean_probs.argmax(dim=-1)
                    epoch_correct += (preds == batch_y).sum().item()
                    epoch_total += len(batch_y)
            
            # Don't step schedulers since feature extractor is frozen
            
            train_loss = epoch_loss / num_batches
            train_acc = epoch_correct / epoch_total
            
            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["diversity"].append(epoch_diversity / num_batches)
            
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                val_loss, val_acc = self._evaluate_svgd(trainer)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, "
                      f"Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            else:
                if history["val_loss"]:
                    history["val_loss"].append(history["val_loss"][-1])
                    history["val_accuracy"].append(history["val_accuracy"][-1])
        
        training_time = time.time() - start_time
        
        # Final metrics
        metrics = self._compute_svgd_metrics(trainer)
        
        # Save everything
        self._save_run(run_dir, "svgd", temperature, replicate, 
                       metrics, history, training_time, num_epochs)
        self._plot_curves(history, run_name, run_dir, "svgd")
        self._mark_completed("svgd", temperature, replicate)
        
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        print(f"✓ {run_name} COMPLETED ({training_time:.1f}s)")
    
    def _train_mfvi(self, temperature: float, replicate: int, num_epochs: int = 500):
        """Train MFVI model with frozen feature extractor."""
        run_name = self.get_run_name("mfvi", temperature, replicate)
        run_dir = self._get_run_dir("mfvi", temperature, replicate)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        start_time = time.time()
        
        # Create MFVI model
        model = FullNetwork(
            self.config.network,
            last_layer_type="mfvi",
            prior_log_var=self.config.mfvi.prior_log_var
        ).to(self.device)
        
        # Load pretrained feature extractor weights
        feature_weights = self._get_feature_extractor_weights()
        model_state = model.state_dict()
        for key, value in feature_weights.items():
            if key in model_state:
                model_state[key] = value
        model.load_state_dict(model_state)
        
        # Freeze feature extractor
        model.freeze_feature_extractor()
        print("Feature extractor loaded and frozen")
        
        # Create trainer
        trainer = MFVITrainer(model, self.config.mfvi, device=self.device)
        
        # Train (trainer handles the optimization)
        mfvi_history = trainer.train(
            self.data_manager.train_loader,
            temperature=temperature,
            num_epochs=num_epochs
        )
        
        training_time = time.time() - start_time
        
        # Convert history
        history = {
            "epochs": [s.epoch for s in mfvi_history],
            "train_loss": [s.loss for s in mfvi_history],
            "kl_loss": [s.kl_loss for s in mfvi_history],
            "nll_loss": [s.nll_loss for s in mfvi_history],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Compute final validation
        val_loss, val_acc = self._evaluate_mfvi(model)
        history["val_loss"] = [val_loss]
        history["val_accuracy"] = [val_acc]
        history["train_accuracy"] = [val_acc]
        
        # Final metrics
        metrics_result = self.metrics_computer.compute_all_metrics(
            model,
            self.data_manager.test_loader,
            self.data_manager.ood_loader,
            num_samples=self.config.mfvi.num_test_samples,
            is_mcmc=False
        )
        metrics = metrics_result.to_dict()
        
        # Save
        self._save_run(run_dir, "mfvi", temperature, replicate,
                       metrics, history, training_time, num_epochs)
        self._plot_curves(history, run_name, run_dir, "mfvi")
        self._mark_completed("mfvi", temperature, replicate)
        
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        print(f"✓ {run_name} COMPLETED ({training_time:.1f}s)")
    
    def _evaluate_svgd(self, trainer) -> Tuple[float, float]:
        """Evaluate SVGD model."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                probs = trainer.predict_proba(batch_x)
                loss = F.cross_entropy(torch.log(probs + 1e-10), batch_y)
                
                total_loss += loss.item() * len(batch_y)
                total_correct += (probs.argmax(dim=-1) == batch_y).sum().item()
                total_samples += len(batch_y)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def _evaluate_mfvi(self, model) -> Tuple[float, float]:
        """Evaluate MFVI model."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = model(batch_x, num_samples=10)
                if logits.dim() == 3:
                    probs = F.softmax(logits, dim=-1).mean(dim=0)
                else:
                    probs = F.softmax(logits, dim=-1)
                
                loss = F.cross_entropy(torch.log(probs + 1e-10), batch_y)
                
                total_loss += loss.item() * len(batch_y)
                total_correct += (probs.argmax(dim=-1) == batch_y).sum().item()
                total_samples += len(batch_y)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def _compute_svgd_metrics(self, trainer) -> Dict[str, float]:
        """Compute all metrics for SVGD."""
        from sklearn.metrics import roc_auc_score
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                probs = trainer.predict_proba(batch_x)
                all_probs.append(probs.cpu())
                all_labels.append(batch_y)
        
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        preds = all_probs.argmax(dim=-1)
        error = (preds != all_labels).float().mean().item()
        nll = F.cross_entropy(torch.log(all_probs + 1e-10), all_labels).item()
        
        # ECE
        confidences, predictions = all_probs.max(dim=-1)
        accuracies = (predictions == all_labels).float()
        bin_boundaries = torch.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                ece += torch.abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * prop_in_bin
        
        # OOD AUROC
        in_entropy = []
        for batch_x, _ in self.data_manager.test_loader:
            batch_x = batch_x.to(self.device)
            _, _, entropy = trainer.ensemble.predict_with_uncertainty(batch_x)
            in_entropy.append(entropy.cpu())
        in_entropy = torch.cat(in_entropy)
        
        ood_entropy = []
        for batch_x, _ in self.data_manager.ood_loader:
            batch_x = batch_x.to(self.device)
            _, _, entropy = trainer.ensemble.predict_with_uncertainty(batch_x)
            ood_entropy.append(entropy.cpu())
        ood_entropy = torch.cat(ood_entropy)
        
        labels = np.concatenate([np.zeros(len(in_entropy)), np.ones(len(ood_entropy))])
        scores = np.concatenate([in_entropy.numpy(), ood_entropy.numpy()])
        ood_auroc = roc_auc_score(labels, scores)
        
        return {"error": error, "nll": nll, "ece": ece.item(), "ood_auroc": ood_auroc}
    
    def _save_run(self, run_dir, method, temperature, replicate, 
                  metrics, history, training_time, num_epochs):
        """Save run results."""
        # Results
        results = {
            "method": method,
            "temperature": temperature,
            "replicate": replicate + 1,
            "metrics": metrics,
            "training_time": training_time,
            "training_history": history,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Hyperparameters
        if method == "svgd":
            hyperparams = asdict(self.config.svgd)
        else:
            hyperparams = asdict(self.config.mfvi)
        hyperparams["temperature"] = temperature
        hyperparams["num_epochs"] = num_epochs
        hyperparams["network"] = asdict(self.config.network)
        
        with open(os.path.join(run_dir, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def _plot_curves(self, history, run_name, run_dir, method):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{run_name}', fontsize=14, fontweight='bold')
        
        epochs = history["epochs"]
        
        axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if method == "svgd" and "diversity" in history:
            axes[1].plot(epochs, history["diversity"], 'g-', label='Diversity', linewidth=2)
            axes[1].set_ylabel('Particle Diversity')
            axes[1].set_title('SVGD Diversity')
        elif method == "mfvi" and "kl_loss" in history:
            axes[1].plot(epochs, history["kl_loss"], 'g-', label='KL Loss', linewidth=2)
            axes[1].set_ylabel('KL Divergence')
            axes[1].set_title('MFVI KL Loss')
        
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# STATUS CHECK
# =============================================================================

def print_status(save_dir: str, config: ExperimentConfig):
    """Print experiment status."""
    print("="*60)
    print("EXPERIMENT STATUS")
    print("="*60)
    print(f"Save directory: {save_dir}")
    print()
    
    # Check deterministic model
    det_dir = os.path.join(save_dir, "deterministic_model")
    det_completed = os.path.exists(os.path.join(det_dir, "COMPLETED"))
    
    print("STEP 1: Deterministic Model")
    if det_completed:
        print(f"  ✓ COMPLETED")
        metrics_path = os.path.join(det_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"    Error: {metrics.get('error', 'N/A'):.4f}")
    else:
        print(f"  ○ NOT STARTED")
        print(f"    Run: python pipeline.py --step 1 --save-dir {save_dir}")
    
    print()
    print("STEP 2: Bayesian Last Layer Models")
    
    if not det_completed:
        print("  (Requires Step 1 to complete first)")
        return
    
    methods = ["svgd", "mfvi"]
    temperatures = config.temperature.temperatures
    num_replicates = config.num_replicates
    
    total = len(methods) * len(temperatures) * num_replicates
    completed = 0
    
    for method in methods:
        for temp in temperatures:
            for rep in range(num_replicates):
                run_name = BayesianLastLayerTrainer.get_run_name(method, temp, rep)
                run_dir = os.path.join(save_dir, run_name)
                if os.path.exists(os.path.join(run_dir, "COMPLETED")):
                    completed += 1
    
    print(f"  Completed: {completed}/{total}")
    
    if completed < total:
        print(f"  Pending: {total - completed}")
        print(f"    Run: python pipeline.py --step 2 --save-dir {save_dir}")
    else:
        print("  ✓ ALL COMPLETED")
    
    print()
    print("="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step BNN training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check status
    python pipeline.py --status --save-dir ./results
    
    # Step 1: Train deterministic model
    python pipeline.py --step 1 --save-dir ./results
    
    # Step 2: Train all Bayesian models
    python pipeline.py --step 2 --save-dir ./results
    
    # For Google Drive (Colab):
    python pipeline.py --step 1 --save-dir /content/drive/MyDrive/bnn_results
    python pipeline.py --step 2 --save-dir /content/drive/MyDrive/bnn_results
        """
    )
    
    parser.add_argument("--step", type=int, choices=[1, 2],
                        help="Step to run: 1=deterministic, 2=bayesian")
    parser.add_argument("--status", action="store_true",
                        help="Print experiment status")
    parser.add_argument("--save-dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--det-epochs", type=int, default=100,
                        help="Epochs for deterministic training")
    parser.add_argument("--svgd-epochs", type=int, default=200,
                        help="Epochs for SVGD training")
    parser.add_argument("--mfvi-epochs", type=int, default=500,
                        help="Epochs for MFVI training")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    config = ExperimentConfig()
    
    if args.status:
        print_status(args.save_dir, config)
        return
    
    if args.step is None:
        parser.print_help()
        print("\n" + "="*60)
        print_status(args.save_dir, config)
        return
    
    if args.step == 1:
        trainer = DeterministicTrainer(config, args.save_dir)
        trainer.train(num_epochs=args.det_epochs)
    
    elif args.step == 2:
        trainer = BayesianLastLayerTrainer(config, args.save_dir)
        trainer.train_all(
            svgd_epochs=args.svgd_epochs,
            mfvi_epochs=args.mfvi_epochs
        )


if __name__ == "__main__":
    main()
