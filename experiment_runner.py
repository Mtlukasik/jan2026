"""
Enhanced Experiment Runner for SVGD vs MFVI Comparison.

This module provides comprehensive experiment tracking including:
1. Training and validation loss/accuracy curves
2. Individual experiment results saved to JSON with consistent naming
3. Model parameters saved for reproducibility
4. Automatic plotting of learning curves
5. Google Drive integration for saving results
6. RESUMABLE TRAINING: Only runs experiments that haven't been completed yet

Naming Convention:
    last_layer_{method}_T{temperature}_replicate_{rep}/
        ├── results.json          # Full results with metrics and history
        ├── hyperparameters.json  # All hyperparameters
        ├── training_curves.png   # Loss/accuracy plots
        └── COMPLETED             # Marker file indicating successful completion

Example:
    If drive contains:
        last_layer_mfvi_T0.1_replicate_1/COMPLETED
        last_layer_mfvi_T0.1_replicate_2/COMPLETED
    
    Running main will skip these and only train:
        - last_layer_mfvi_T0.1_replicate_3
        - last_layer_svgd_T0.1_replicate_1
        - last_layer_svgd_T0.1_replicate_2
        - last_layer_svgd_T0.1_replicate_3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import shutil
import glob
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from config import (
    ExperimentConfig, DataConfig, NetworkConfig,
    MFVIConfig, SVGDConfig, TemperatureConfig, OODConfig, CalibrationConfig,
    DEVICE, SEED
)
from data_loading import DataLoaderManager
from networks import FullNetwork
from metrics import MetricsComputer, MetricResults
from mfvi_inference import MFVITrainer
from svgd_inference import SVGDTrainer, SVGDEnsemble


@dataclass
class TrainingHistory:
    """Container for training history."""
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    # SVGD-specific
    diversity: List[float] = field(default_factory=list)
    # MFVI-specific
    kl_loss: List[float] = field(default_factory=list)
    nll_loss: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Container for single experiment run results."""
    method: str  # "svgd" or "mfvi"
    temperature: float
    replicate: int
    metrics: Dict[str, float]
    training_time: float  # seconds
    training_history: Optional[Dict] = None
    hyperparameters: Optional[Dict] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AggregatedResults:
    """Container for aggregated results across replicates."""
    method: str
    temperature: float
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    num_replicates: int


class EnhancedExperimentRunner:
    """Enhanced experiment runner with comprehensive logging, saving, and resume capability.
    
    Key Features:
    - Consistent naming: last_layer_{method}_T{temp}_replicate_{rep}/
    - Resume capability: Checks for COMPLETED marker file to skip finished runs
    - Full tracking: Training curves, metrics, hyperparameters all saved
    - Google Drive support: Auto-sync to Drive in Colab
    """
    
    # Naming convention for experiment directories
    RUN_NAME_TEMPLATE = "last_layer_{method}_T{temperature}_replicate_{replicate}"
    
    def __init__(
        self,
        config: ExperimentConfig,
        save_dir: str = "./results",
        save_to_drive: bool = False,
        drive_path: str = "/content/drive/MyDrive/bnn_experiments"
    ):
        self.config = config
        self.device = DEVICE
        self.save_dir = save_dir
        self.save_to_drive = save_to_drive
        self.drive_path = drive_path
        
        # Determine primary save location
        if save_to_drive and os.path.exists("/content/drive"):
            self.primary_save_dir = drive_path
        else:
            self.primary_save_dir = save_dir
            if save_to_drive and not os.path.exists("/content/drive"):
                print("Warning: Google Drive not mounted. Saving locally only.")
                self.save_to_drive = False
        
        # Create directories
        self._setup_directories()
        
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
        
        # Track completed experiments
        self.completed_runs: Set[str] = self._scan_completed_runs()
        
        # Experiment timestamp for this session
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def get_run_name(method: str, temperature: float, replicate: int) -> str:
        """Generate consistent run name."""
        return EnhancedExperimentRunner.RUN_NAME_TEMPLATE.format(
            method=method,
            temperature=temperature,
            replicate=replicate + 1  # 1-indexed for human readability
        )
    
    def _get_run_dir(self, method: str, temperature: float, replicate: int) -> str:
        """Get full path to run directory."""
        run_name = self.get_run_name(method, temperature, replicate)
        return os.path.join(self.primary_save_dir, run_name)
    
    def _is_run_completed(self, method: str, temperature: float, replicate: int) -> bool:
        """Check if a specific run has been completed."""
        run_dir = self._get_run_dir(method, temperature, replicate)
        completed_marker = os.path.join(run_dir, "COMPLETED")
        return os.path.exists(completed_marker)
    
    def _scan_completed_runs(self) -> Set[str]:
        """Scan save directory for completed runs."""
        completed = set()
        
        # Check primary save directory
        if os.path.exists(self.primary_save_dir):
            for item in os.listdir(self.primary_save_dir):
                item_path = os.path.join(self.primary_save_dir, item)
                if os.path.isdir(item_path):
                    completed_marker = os.path.join(item_path, "COMPLETED")
                    if os.path.exists(completed_marker):
                        completed.add(item)
        
        return completed
    
    def _mark_run_completed(self, method: str, temperature: float, replicate: int):
        """Mark a run as completed by creating marker file."""
        run_dir = self._get_run_dir(method, temperature, replicate)
        completed_marker = os.path.join(run_dir, "COMPLETED")
        
        with open(completed_marker, 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")
        
        # Update tracking
        run_name = self.get_run_name(method, temperature, replicate)
        self.completed_runs.add(run_name)
    
    def _setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.primary_save_dir, exist_ok=True)
    
    def _ensure_run_dir(self, method: str, temperature: float, replicate: int) -> str:
        """Ensure run directory exists and return path."""
        run_dir = self._get_run_dir(method, temperature, replicate)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def _create_model(self, last_layer_type: str) -> FullNetwork:
        """Create a new model instance."""
        return FullNetwork(
            self.config.network,
            last_layer_type=last_layer_type,
            prior_log_var=self.config.mfvi.prior_log_var
        ).to(self.device)
    
    def _compute_validation_metrics(
        self,
        model_or_trainer,
        is_svgd: bool = False
    ) -> Tuple[float, float]:
        """Compute validation loss and accuracy."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if is_svgd:
                    probs = model_or_trainer.predict_proba(batch_x)
                    loss = F.cross_entropy(torch.log(probs + 1e-10), batch_y)
                    preds = probs.argmax(dim=-1)
                else:
                    # MFVI model
                    model_or_trainer.eval()
                    logits = model_or_trainer(batch_x, num_samples=10)
                    if logits.dim() == 3:
                        probs = F.softmax(logits, dim=-1).mean(dim=0)
                    else:
                        probs = F.softmax(logits, dim=-1)
                    loss = F.cross_entropy(torch.log(probs + 1e-10), batch_y)
                    preds = probs.argmax(dim=-1)
                    model_or_trainer.train()
                
                total_loss += loss.item() * len(batch_y)
                total_correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def _plot_training_curves(
        self,
        history: TrainingHistory,
        method: str,
        temperature: float,
        replicate: int,
        run_dir: str
    ) -> str:
        """Plot and save training curves to run directory."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        run_name = self.get_run_name(method, temperature, replicate)
        fig.suptitle(f'{run_name} Training Curves', fontsize=14, fontweight='bold')
        
        epochs = history.epochs
        
        # Plot 1: Training Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, history.train_loss, 'b-', label='Train Loss', linewidth=2)
        if history.val_loss:
            ax1.plot(epochs, history.val_loss, 'r--', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2 = axes[0, 1]
        ax2.plot(epochs, history.train_accuracy, 'b-', label='Train Acc', linewidth=2)
        if history.val_accuracy:
            ax2.plot(epochs, history.val_accuracy, 'r--', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: Method-specific metrics
        ax3 = axes[1, 0]
        if method == "svgd" and history.diversity:
            ax3.plot(epochs, history.diversity, 'g-', label='Diversity', linewidth=2)
            ax3.set_ylabel('Particle Diversity')
            ax3.set_title('SVGD Particle Diversity')
        elif method == "mfvi" and history.kl_loss:
            ax3.plot(epochs, history.kl_loss, 'g-', label='KL Loss', linewidth=2)
            ax3.set_ylabel('KL Divergence')
            ax3.set_title('MFVI KL Divergence')
        ax3.set_xlabel('Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning rate or NLL (if available)
        ax4 = axes[1, 1]
        if method == "mfvi" and history.nll_loss:
            ax4.plot(epochs, history.nll_loss, 'm-', label='NLL Loss', linewidth=2)
            ax4.set_ylabel('NLL')
            ax4.set_title('Negative Log-Likelihood')
        else:
            # Plot loss on log scale
            ax4.semilogy(epochs, history.train_loss, 'b-', label='Train Loss (log)', linewidth=2)
            ax4.set_ylabel('Loss (log scale)')
            ax4.set_title('Loss (Log Scale)')
        ax4.set_xlabel('Epoch')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(run_dir, "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _save_run_results(
        self,
        result: ExperimentResult,
        method: str,
        temperature: float,
        replicate: int,
        run_dir: str
    ):
        """Save all results for a run to its directory."""
        # Save full results
        results_path = os.path.join(run_dir, "results.json")
        result_dict = {
            "method": result.method,
            "temperature": result.temperature,
            "replicate": result.replicate,
            "metrics": result.metrics,
            "training_time": result.training_time,
            "training_history": result.training_history,
            "timestamp": result.timestamp
        }
        with open(results_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        # Save hyperparameters separately for easy access
        hyperparams_path = os.path.join(run_dir, "hyperparameters.json")
        with open(hyperparams_path, 'w') as f:
            json.dump(result.hyperparameters, f, indent=2)
        
        run_name = self.get_run_name(method, temperature, replicate)
        print(f"  → Saved to: {run_dir}")
    
    def run_svgd_experiment(
        self,
        temperature: float,
        replicate: int,
        pretrain_epochs: int = 50,
        svgd_epochs: int = 200,
        val_frequency: int = 10
    ) -> ExperimentResult:
        """Run a single SVGD experiment with full tracking."""
        run_name = self.get_run_name("svgd", temperature, replicate)
        
        # Check if already completed
        if self._is_run_completed("svgd", temperature, replicate):
            print(f"\n[SKIP] {run_name} already completed")
            return None
        
        # Set random seed
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        # Ensure run directory exists
        run_dir = self._ensure_run_dir("svgd", temperature, replicate)
        
        start_time = time.time()
        
        # Create SVGD trainer
        trainer = SVGDTrainer(
            self.config.network,
            self.config.svgd,
            device=self.device
        )
        
        # Pretrain feature extractor
        print("Pretraining feature extractor...")
        pretrain_acc = trainer.pretrain_feature_extractor(
            self.data_manager.train_loader,
            self.data_manager.test_loader,
            num_epochs=pretrain_epochs
        )
        print(f"Pretrain validation accuracy: {pretrain_acc:.4f}")
        
        # Training with tracking
        print(f"Running SVGD with temperature T={temperature}...")
        history = TrainingHistory()
        
        num_epochs = svgd_epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_diversity = 0.0
            epoch_correct = 0
            epoch_total = 0
            num_batches = 0
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                loss, diversity = trainer.ensemble.train_step(
                    batch_x, batch_y, temperature
                )
                
                epoch_loss += loss
                epoch_diversity += diversity
                num_batches += 1
                
                # Compute training accuracy
                with torch.no_grad():
                    mean_probs, _, _ = trainer.ensemble.predict_with_uncertainty(batch_x)
                    preds = mean_probs.argmax(dim=-1)
                    epoch_correct += (preds == batch_y).sum().item()
                    epoch_total += len(batch_y)
            
            trainer.ensemble.step_schedulers()
            
            # Record training metrics
            train_loss = epoch_loss / num_batches
            train_acc = epoch_correct / epoch_total
            
            history.epochs.append(epoch)
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)
            history.diversity.append(epoch_diversity / num_batches)
            
            # Compute validation metrics periodically
            if (epoch + 1) % val_frequency == 0 or epoch == num_epochs - 1:
                val_loss, val_acc = self._compute_validation_metrics(trainer, is_svgd=True)
                history.val_loss.append(val_loss)
                history.val_accuracy.append(val_acc)
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                # Interpolate validation metrics
                if history.val_loss:
                    history.val_loss.append(history.val_loss[-1])
                    history.val_accuracy.append(history.val_accuracy[-1])
                else:
                    history.val_loss.append(train_loss)
                    history.val_accuracy.append(train_acc)
        
        # Plot training curves
        plot_path = self._plot_training_curves(history, "svgd", temperature, replicate, run_dir)
        print(f"  → Saved plot: {plot_path}")
        
        # Evaluate final metrics
        print("Evaluating...")
        metrics = self._evaluate_svgd(trainer)
        
        training_time = time.time() - start_time
        
        # Collect hyperparameters
        hyperparams = {
            "method": "svgd",
            "temperature": temperature,
            "replicate": replicate + 1,
            "n_particles": self.config.svgd.n_particles,
            "svgd_lr": self.config.svgd.svgd_lr,
            "feature_lr": self.config.svgd.feature_lr,
            "prior_std": self.config.svgd.prior_std,
            "use_laplace_prior": self.config.svgd.use_laplace_prior,
            "num_epochs": svgd_epochs,
            "pretrain_epochs": pretrain_epochs,
            "hidden_dim": self.config.network.hidden_dim,
            "dropout_rate": self.config.network.dropout_rate,
            "batch_size": self.config.data.batch_size,
        }
        
        result = ExperimentResult(
            method="svgd",
            temperature=temperature,
            replicate=replicate,
            metrics=metrics,
            training_time=training_time,
            training_history=history.to_dict(),
            hyperparameters=hyperparams
        )
        
        # Save results to run directory
        self._save_run_results(result, "svgd", temperature, replicate, run_dir)
        
        # Mark as completed
        self._mark_run_completed("svgd", temperature, replicate)
        
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        print(f"Training time: {training_time:.1f}s")
        print(f"✓ {run_name} COMPLETED")
        
        return result
    
    def run_mfvi_experiment(
        self,
        temperature: float,
        replicate: int,
        pretrain_epochs: int = 50,
        mfvi_epochs: int = 500,
        val_frequency: int = 20
    ) -> ExperimentResult:
        """Run a single MFVI experiment with full tracking."""
        run_name = self.get_run_name("mfvi", temperature, replicate)
        
        # Check if already completed
        if self._is_run_completed("mfvi", temperature, replicate):
            print(f"\n[SKIP] {run_name} already completed")
            return None
        
        # Set random seed
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        # Ensure run directory exists
        run_dir = self._ensure_run_dir("mfvi", temperature, replicate)
        
        start_time = time.time()
        
        # Create model and trainer
        model = self._create_model("mfvi")
        trainer = MFVITrainer(model, self.config.mfvi, device=self.device)
        
        # Pretrain feature extractor
        print("Pretraining feature extractor...")
        pretrain_acc = trainer.pretrain_feature_extractor(
            self.data_manager.train_loader,
            self.data_manager.test_loader,
            num_epochs=pretrain_epochs
        )
        print(f"Pretrain validation accuracy: {pretrain_acc:.4f}")
        
        # Training with tracking
        print(f"Training MFVI with λ={temperature}...")
        history = TrainingHistory()
        
        # Get MFVI training history from trainer
        mfvi_history = trainer.train(
            self.data_manager.train_loader,
            temperature=temperature,
            num_epochs=mfvi_epochs
        )
        
        # Convert MFVI history to our format
        for state in mfvi_history:
            history.epochs.append(state.epoch)
            history.train_loss.append(state.loss)
            history.kl_loss.append(state.kl_loss)
            history.nll_loss.append(state.nll_loss)
            
            # Compute train accuracy from loss (approximate)
            # We'll compute actual accuracy periodically
            if state.epoch % val_frequency == 0 or state.epoch == mfvi_epochs - 1:
                val_loss, val_acc = self._compute_validation_metrics(model, is_svgd=False)
                history.val_loss.append(val_loss)
                history.val_accuracy.append(val_acc)
                history.train_accuracy.append(val_acc)  # Approximate
            else:
                if history.val_loss:
                    history.val_loss.append(history.val_loss[-1])
                    history.val_accuracy.append(history.val_accuracy[-1])
                    history.train_accuracy.append(history.train_accuracy[-1])
                else:
                    history.val_loss.append(state.loss)
                    history.val_accuracy.append(0.5)
                    history.train_accuracy.append(0.5)
        
        # Plot training curves
        plot_path = self._plot_training_curves(history, "mfvi", temperature, replicate, run_dir)
        print(f"  → Saved plot: {plot_path}")
        
        # Evaluate final metrics
        print("Evaluating...")
        metrics_result = self.metrics_computer.compute_all_metrics(
            model,
            self.data_manager.test_loader,
            self.data_manager.ood_loader,
            num_samples=self.config.mfvi.num_test_samples,
            is_mcmc=False
        )
        metrics = metrics_result.to_dict()
        
        training_time = time.time() - start_time
        
        # Collect hyperparameters
        hyperparams = {
            "method": "mfvi",
            "temperature": temperature,
            "replicate": replicate + 1,
            "num_epochs": mfvi_epochs,
            "pretrain_epochs": pretrain_epochs,
            "learning_rate_init": self.config.mfvi.learning_rate_init,
            "kl_annealing_epochs": self.config.mfvi.kl_annealing_epochs,
            "num_train_samples": self.config.mfvi.num_train_samples,
            "num_test_samples": self.config.mfvi.num_test_samples,
            "prior_log_var": self.config.mfvi.prior_log_var,
            "hidden_dim": self.config.network.hidden_dim,
            "dropout_rate": self.config.network.dropout_rate,
            "batch_size": self.config.data.batch_size,
        }
        
        result = ExperimentResult(
            method="mfvi",
            temperature=temperature,
            replicate=replicate,
            metrics=metrics,
            training_time=training_time,
            training_history=history.to_dict(),
            hyperparameters=hyperparams
        )
        
        # Save results to run directory
        self._save_run_results(result, "mfvi", temperature, replicate, run_dir)
        
        # Mark as completed
        self._mark_run_completed("mfvi", temperature, replicate)
        
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        print(f"Training time: {training_time:.1f}s")
        print(f"✓ {run_name} COMPLETED")
        
        return result
    
    def _evaluate_svgd(self, trainer: SVGDTrainer) -> Dict[str, float]:
        """Evaluate SVGD ensemble on all metrics."""
        from sklearn.metrics import roc_auc_score
        
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
        from sklearn.metrics import roc_auc_score
        
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
        
        # Compute AUROC
        labels = np.concatenate([
            np.zeros(len(in_dist_entropy)),
            np.ones(len(ood_entropy))
        ])
        scores = np.concatenate([
            in_dist_entropy.numpy(),
            ood_entropy.numpy()
        ])
        
        return roc_auc_score(labels, scores)
    
    def run_all_experiments(
        self,
        methods: List[str] = ["svgd", "mfvi"],
        temperatures: Optional[List[float]] = None,
        num_replicates: Optional[int] = None,
        pretrain_epochs: int = 50,
        mfvi_epochs: int = 500,
        svgd_epochs: int = 200
    ) -> List[ExperimentResult]:
        """Run all experiment configurations, skipping completed ones.
        
        Scans the save directory for completed runs and only trains
        experiments that haven't been finished yet.
        """
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        # Build list of all required runs
        all_runs = []
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    run_name = self.get_run_name(method, temp, rep)
                    all_runs.append((method, temp, rep, run_name))
        
        # Identify pending runs
        pending_runs = [
            (method, temp, rep, name) 
            for method, temp, rep, name in all_runs 
            if name not in self.completed_runs
        ]
        
        completed_count = len(all_runs) - len(pending_runs)
        
        print(f"\n{'#'*70}")
        print(f"EXPERIMENT STATUS")
        print(f"{'#'*70}")
        print(f"Total experiments: {len(all_runs)}")
        print(f"Already completed: {completed_count}")
        print(f"Pending: {len(pending_runs)}")
        print(f"Save directory: {self.primary_save_dir}")
        print(f"{'#'*70}")
        
        if completed_count > 0:
            print(f"\nCompleted runs found:")
            for name in sorted(self.completed_runs):
                print(f"  ✓ {name}")
        
        if len(pending_runs) == 0:
            print("\nAll experiments already completed!")
            # Load existing results
            self._load_completed_results(methods, temperatures, num_replicates)
            return self.results
        
        print(f"\nPending runs to train:")
        for method, temp, rep, name in pending_runs:
            print(f"  ○ {name}")
        
        print(f"\n{'#'*70}")
        print(f"Starting {len(pending_runs)} experiments...")
        print(f"{'#'*70}")
        
        self.results = []
        current_exp = 0
        
        for method, temp, rep, name in pending_runs:
            current_exp += 1
            print(f"\n[Experiment {current_exp}/{len(pending_runs)}]")
            
            if method == "svgd":
                result = self.run_svgd_experiment(
                    temp, rep, pretrain_epochs, svgd_epochs
                )
            elif method == "mfvi":
                result = self.run_mfvi_experiment(
                    temp, rep, pretrain_epochs, mfvi_epochs
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if result is not None:
                self.results.append(result)
        
        # Load all results (including previously completed)
        self._load_completed_results(methods, temperatures, num_replicates)
        
        # Save aggregated results
        self._save_aggregated_results()
        
        return self.results
    
    def _load_completed_results(
        self,
        methods: List[str],
        temperatures: List[float],
        num_replicates: int
    ):
        """Load results from completed runs."""
        self.results = []
        
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    run_dir = self._get_run_dir(method, temp, rep)
                    results_file = os.path.join(run_dir, "results.json")
                    
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                            
                            result = ExperimentResult(
                                method=data["method"],
                                temperature=data["temperature"],
                                replicate=data["replicate"],
                                metrics=data["metrics"],
                                training_time=data["training_time"],
                                training_history=data.get("training_history"),
                                hyperparameters=data.get("hyperparameters"),
                                timestamp=data.get("timestamp", "")
                            )
                            self.results.append(result)
                        except Exception as e:
                            print(f"Warning: Could not load {results_file}: {e}")
    
    def aggregate_results(self) -> List[AggregatedResults]:
        """Aggregate results across replicates."""
        aggregated = []
        
        groups: Dict[Tuple[str, float], List[ExperimentResult]] = {}
        for result in self.results:
            key = (result.method, result.temperature)
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        for (method, temp), results in groups.items():
            metrics_arrays = {key: [] for key in results[0].metrics.keys()}
            
            for result in results:
                for key, value in result.metrics.items():
                    metrics_arrays[key].append(value)
            
            metrics_mean = {key: float(np.mean(vals)) for key, vals in metrics_arrays.items()}
            metrics_std = {key: float(np.std(vals)) for key, vals in metrics_arrays.items()}
            
            aggregated.append(AggregatedResults(
                method=method,
                temperature=temp,
                metrics_mean=metrics_mean,
                metrics_std=metrics_std,
                num_replicates=len(results)
            ))
        
        return aggregated
    
    def _save_aggregated_results(self):
        """Save aggregated results to JSON in primary save directory."""
        aggregated = self.aggregate_results()
        
        # Convert to serializable format
        results_dict = {
            "experiment_timestamp": self.experiment_timestamp,
            "config": {
                "temperatures": self.config.temperature.temperatures,
                "num_replicates": self.config.num_replicates,
                "network": asdict(self.config.network),
                "svgd": asdict(self.config.svgd),
                "mfvi": asdict(self.config.mfvi),
            },
            "aggregated_results": [
                {
                    "method": a.method,
                    "temperature": a.temperature,
                    "metrics_mean": a.metrics_mean,
                    "metrics_std": a.metrics_std,
                    "num_replicates": a.num_replicates
                }
                for a in aggregated
            ],
            "individual_results": [
                {
                    "method": r.method,
                    "temperature": r.temperature,
                    "replicate": r.replicate,
                    "metrics": r.metrics,
                    "training_time": r.training_time,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        filename = f"aggregated_results.json"
        filepath = os.path.join(self.primary_save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n→ Saved aggregated results: {filepath}")
    
    def save_results(self, filepath: str):
        """Save all results to JSON file."""
        data = {
            "config": asdict(self.config),
            "results": [
                {
                    "method": r.method,
                    "temperature": r.temperature,
                    "replicate": r.replicate,
                    "metrics": r.metrics,
                    "training_time": r.training_time,
                    "training_history": r.training_history,
                    "hyperparameters": r.hyperparameters,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def plot_final_comparison(self, save_path: Optional[str] = None):
        """Generate final comparison plots."""
        from visualization import ResultsPlotter, convert_aggregated_results_to_plot_format
        
        aggregated = self.aggregate_results()
        plot_data = convert_aggregated_results_to_plot_format(aggregated)
        
        svgd_data = plot_data.get("svgd", {})
        mfvi_data = plot_data.get("mfvi", {})
        
        plotter = ResultsPlotter()
        
        if save_path is None:
            save_path = os.path.join(self.primary_save_dir, "final_comparison.png")
        
        fig = plotter.plot_comparison_grid(
            svgd_data, mfvi_data,
            title="SVGD vs MFVI Last Layer Inference on CIFAR-10",
            save_path=save_path
        )
        
        print(f"→ Saved final comparison: {save_path}")
        
        return fig
    
    def list_completed_runs(self) -> List[str]:
        """Return list of completed run names."""
        return sorted(list(self.completed_runs))
    
    def list_pending_runs(
        self,
        methods: List[str] = ["svgd", "mfvi"],
        temperatures: Optional[List[float]] = None,
        num_replicates: Optional[int] = None
    ) -> List[str]:
        """Return list of pending run names."""
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        pending = []
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    run_name = self.get_run_name(method, temp, rep)
                    if run_name not in self.completed_runs:
                        pending.append(run_name)
        
        return pending


def run_quick_experiment_enhanced(save_dir: str = "./results_quick"):
    """Run a quick experiment with enhanced tracking."""
    config = ExperimentConfig(
        data=DataConfig(batch_size=64),
        network=NetworkConfig(hidden_dim=256),
        svgd=SVGDConfig(n_particles=5, num_epochs=50),
        mfvi=MFVIConfig(num_epochs=100),
        temperature=TemperatureConfig(temperatures=[0.1, 1.0]),
        num_replicates=3
    )
    
    runner = EnhancedExperimentRunner(
        config,
        save_dir=save_dir,
        save_to_drive=False
    )
    
    # Show what's pending
    print("\nPending runs:")
    for run in runner.list_pending_runs(methods=["mfvi"]):
        print(f"  ○ {run}")
    
    results = runner.run_all_experiments(
        methods=["mfvi"],
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


if __name__ == "__main__":
    run_quick_experiment_enhanced()
