"""
ResNet-based Training Pipeline for SVGD vs MFVI Comparison.

Three-step process:

    STEP 1: Train ResNet-18 on CIFAR-10 with standard settings
        - SGD + momentum 0.9, weight decay 5e-4
        - LR: 0.1 with CosineAnnealing
        - Augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
        - Train 200 epochs
        - Save weights to deterministic_model/
    
    STEP 2: Train Bayesian last layers (frozen features)
        - Load pretrained ResNet feature extractor → FREEZE
        - Initialize particles from pretrained last layer
        - Train only last layer with SVGD or MFVI
        - 36 configurations: 2 methods × 6 temperatures × 3 replicates
        - Save to last_layer_{method}_T{temp}_replicate_{rep}/
    
    STEP 3: Joint training (optional, unfreeze features)
        - Load Step 2 results (trained Bayesian last layer)
        - Unfreeze feature extractor with lower LR
        - Continue training both features + Bayesian last layer
        - Fine-tunes features for better uncertainty estimation
        - Save to joint_{method}_T{temp}_replicate_{rep}/

Usage:
    python pipeline_resnet.py --step 1 --save-dir ./results   # Train ResNet
    python pipeline_resnet.py --step 2 --save-dir ./results   # Bayesian last layer
    python pipeline_resnet.py --step 3 --save-dir ./results   # Joint training
    python pipeline_resnet.py --status --save-dir ./results   # Check progress
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

from config import ExperimentConfig, DEVICE, SEED
from data_loading_resnet import DataLoaderManager
from resnet import ResNetForBayesianLastLayer, ResNetFeatureExtractor


# =============================================================================
# STEP 1: Deterministic ResNet Training
# =============================================================================

class DeterministicResNetTrainer:
    """Trainer for deterministic ResNet-18 on CIFAR-10."""
    
    def __init__(self, config: ExperimentConfig, save_dir: str, device: str = None):
        self.config = config
        self.save_dir = save_dir
        self.device = device or DEVICE
        self.model_dir = os.path.join(save_dir, "deterministic_model")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Data with augmentation
        self.data_manager = DataLoaderManager(
            config.data, config.ood, config.calibration,
            flatten=False  # Keep 3D for ResNet
        )
        
        # Model
        self.model = ResNetForBayesianLastLayer(
            num_classes=10, last_layer_type="deterministic"
        ).to(self.device)
        
        # History
        self.history = {
            "epochs": [], "train_loss": [], "train_accuracy": [],
            "val_epochs": [], "val_loss": [], "val_accuracy": [], "lr": []
        }
    
    def is_completed(self) -> bool:
        return os.path.exists(os.path.join(self.model_dir, "COMPLETED"))
    
    def get_weights_path(self) -> str:
        return os.path.join(self.model_dir, "model_weights.pt")
    
    def train(
        self,
        num_epochs: int = 200,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        val_frequency: int = 5
    ) -> Dict:
        """Train ResNet-18 with standard CIFAR-10 settings."""
        
        if self.is_completed():
            print(f"[SKIP] Deterministic model already trained at {self.model_dir}")
            return self._load_metrics()
        
        print("=" * 70)
        print("STEP 1: Training ResNet-18 on CIFAR-10")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"LR: {lr}, Momentum: {momentum}, Weight Decay: {weight_decay}")
        print(f"Batch Size: {self.config.data.batch_size}")
        print(f"Augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip")
        print("=" * 70)
        
        start_time = time.time()
        
        # Optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Cosine annealing scheduler
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
            
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["lr"].append(current_lr)
            
            # Validation
            if (epoch + 1) % val_frequency == 0 or epoch == num_epochs - 1:
                val_loss, val_acc = self._evaluate()
                self.history["val_epochs"].append(epoch)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_weights()
        
        training_time = time.time() - start_time
        
        # Final evaluation with best weights
        print("\nFinal Evaluation with best weights...")
        self._load_weights()
        final_metrics = self._compute_final_metrics()
        final_metrics["training_time"] = training_time
        final_metrics["best_val_accuracy"] = best_val_acc
        
        # Save
        self._save_history()
        self._save_metrics(final_metrics)
        self._plot_training_curves()
        self._mark_completed()
        
        print(f"\n✓ ResNet-18 training complete!")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        print(f"  Test Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Test Error: {final_metrics['error']:.4f}")
        print(f"  Training time: {training_time/60:.1f} min")
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
        """Compute all final test metrics."""
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
        
        preds = all_probs.argmax(dim=-1)
        error = (preds != all_labels).float().mean().item()
        nll = F.nll_loss(torch.log(all_probs + 1e-10), all_labels).item()
        
        # ECE
        confidences, predictions = all_probs.max(dim=-1)
        accuracies = (predictions == all_labels).float()
        bin_boundaries = torch.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                ece += torch.abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * in_bin.float().mean()
        
        return {"error": error, "nll": nll, "ece": ece.item(), "accuracy": 1 - error}
    
    def _save_weights(self):
        torch.save(self.model.state_dict(), self.get_weights_path())
    
    def _load_weights(self):
        self.model.load_state_dict(torch.load(self.get_weights_path(), map_location=self.device))
    
    def _save_history(self):
        with open(os.path.join(self.model_dir, "training_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _save_metrics(self, metrics: Dict):
        with open(os.path.join(self.model_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _load_metrics(self) -> Dict:
        with open(os.path.join(self.model_dir, "metrics.json"), 'r') as f:
            return json.load(f)
    
    def _plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('ResNet-18 Training on CIFAR-10', fontsize=14, fontweight='bold')
        
        epochs = self.history["epochs"]
        val_epochs = self.history["val_epochs"]
        
        # Loss
        axes[0].plot(epochs, self.history["train_loss"], 'b-', label='Train', linewidth=2)
        axes[0].plot(val_epochs, self.history["val_loss"], 'ro--', label='Val', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history["train_accuracy"], 'b-', label='Train', linewidth=2)
        axes[1].plot(val_epochs, self.history["val_accuracy"], 'ro--', label='Val', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Learning Rate
        axes[2].plot(epochs, self.history["lr"], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _mark_completed(self):
        with open(os.path.join(self.model_dir, "COMPLETED"), 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")


# =============================================================================
# STEP 2: Bayesian Last Layer Training
# =============================================================================

class BayesianLastLayerTrainer:
    """Train Bayesian last layers using pretrained ResNet features."""
    
    RUN_NAME_TEMPLATE = "last_layer_{method}_T{temperature}_replicate_{replicate}"
    
    def __init__(self, config: ExperimentConfig, save_dir: str, device: str = None):
        self.config = config
        self.save_dir = save_dir
        self.device = device or DEVICE
        self.deterministic_dir = os.path.join(save_dir, "deterministic_model")
        
        if not self._deterministic_exists():
            raise RuntimeError(
                f"No deterministic model found. Run Step 1 first:\n"
                f"  python pipeline_resnet.py --step 1 --save-dir {save_dir}"
            )
        
        # Load data (no augmentation for Bayesian training)
        self.data_manager = DataLoaderManager(
            config.data, config.ood, config.calibration, flatten=False
        )
        
        # Load pretrained weights
        self.pretrained_weights = torch.load(
            os.path.join(self.deterministic_dir, "model_weights.pt"),
            map_location=self.device
        )
        print(f"Loaded pretrained weights from {self.deterministic_dir}")
        
        self.completed_runs = self._scan_completed()
    
    def _deterministic_exists(self) -> bool:
        return os.path.exists(os.path.join(self.deterministic_dir, "COMPLETED"))
    
    def _scan_completed(self) -> Set[str]:
        completed = set()
        if os.path.exists(self.save_dir):
            for item in os.listdir(self.save_dir):
                if item.startswith("last_layer_"):
                    if os.path.exists(os.path.join(self.save_dir, item, "COMPLETED")):
                        completed.add(item)
        return completed
    
    @staticmethod
    def get_run_name(method: str, temperature: float, replicate: int) -> str:
        return BayesianLastLayerTrainer.RUN_NAME_TEMPLATE.format(
            method=method, temperature=temperature, replicate=replicate + 1
        )
    
    def _get_run_dir(self, method: str, temperature: float, replicate: int) -> str:
        return os.path.join(self.save_dir, self.get_run_name(method, temperature, replicate))
    
    def _is_completed(self, method: str, temperature: float, replicate: int) -> bool:
        run_name = self.get_run_name(method, temperature, replicate)
        return run_name in self.completed_runs
    
    def get_pending(self, methods=["svgd", "mfvi"], temperatures=None, num_replicates=None):
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        pending = []
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    if not self._is_completed(method, temp, rep):
                        pending.append((method, temp, rep))
        return pending
    
    def train_all(
        self,
        methods=["svgd", "mfvi"],
        temperatures=None,
        num_replicates=None,
        svgd_epochs=100,
        mfvi_epochs=200
    ):
        """Train all pending Bayesian models."""
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        pending = self.get_pending(methods, temperatures, num_replicates)
        total = len(methods) * len(temperatures) * num_replicates
        
        print("=" * 70)
        print("STEP 2: Training Bayesian Last Layers")
        print("=" * 70)
        print(f"Pretrained features from: {self.deterministic_dir}")
        print(f"Total: {total}, Completed: {total - len(pending)}, Pending: {len(pending)}")
        print("=" * 70)
        
        if len(pending) == 0:
            print("✓ All Bayesian models already trained!")
            return
        
        for i, (method, temp, rep) in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}]")
            if method == "svgd":
                self._train_svgd(temp, rep, svgd_epochs)
            else:
                self._train_mfvi(temp, rep, mfvi_epochs)
    
    def _train_svgd(self, temperature: float, replicate: int, num_epochs: int):
        """
        Train SVGD with frozen ResNet features.
        
        Training Process:
            1. Load pretrained weights (feature extractor + last layer)
            2. Initialize all particles from pretrained last layer (+ small perturbation)
            3. Freeze feature extractor
            4. Phase 1: Train only last layer particles via SVGD
            5. (Optional) Phase 2: Unfreeze features for joint training
        
        SVGD Update:
            For each particle θᵢ:
            θᵢ ← θᵢ + ε · (1/n) Σⱼ [k(θⱼ,θᵢ)·∇log p(θⱼ|D) + ∇k(θⱼ,θᵢ)]
                                    \_____attractive______/   \_repulsive_/
        """
        run_name = self.get_run_name("svgd", temperature, replicate)
        run_dir = self._get_run_dir("svgd", temperature, replicate)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")
        
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        start_time = time.time()
        
        # =====================================================================
        # Step 1: Create shared feature extractor from pretrained weights
        # =====================================================================
        feature_extractor = ResNetFeatureExtractor().to(self.device)
        
        # Load pretrained feature extractor weights
        fe_state = {
            k.replace("feature_extractor.", ""): v 
            for k, v in self.pretrained_weights.items() 
            if k.startswith("feature_extractor.")
        }
        feature_extractor.load_state_dict(fe_state)
        
        # Freeze feature extractor
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        
        print(f"Loaded and froze pretrained feature extractor")
        
        # =====================================================================
        # Step 2: Get pretrained last layer weights
        # =====================================================================
        pretrained_weight = self.pretrained_weights["last_layer.fc.weight"].clone()
        pretrained_bias = self.pretrained_weights["last_layer.fc.bias"].clone()
        
        print(f"Pretrained last layer shape: weight={pretrained_weight.shape}, bias={pretrained_bias.shape}")
        
        # =====================================================================
        # Step 3: Create particles initialized from pretrained last layer
        # =====================================================================
        n_particles = self.config.svgd.n_particles
        particles = []
        
        for i in range(n_particles):
            particle = nn.Linear(512, 10).to(self.device)
            
            # Initialize with pretrained weights
            particle.weight.data.copy_(pretrained_weight)
            particle.bias.data.copy_(pretrained_bias)
            
            # Add small perturbation for diversity (except particle 0)
            if i > 0:
                with torch.no_grad():
                    particle.weight.data.add_(torch.randn_like(particle.weight) * 0.01)
                    particle.bias.data.add_(torch.randn_like(particle.bias) * 0.01)
            
            particles.append(particle)
        
        print(f"Created {n_particles} particles (particle 0 = exact pretrained, rest perturbed)")
        
        # =====================================================================
        # Helper functions for SVGD
        # =====================================================================
        def get_params(particle):
            """Flatten particle parameters."""
            return torch.cat([particle.weight.flatten(), particle.bias.flatten()])
        
        def set_params(particle, params):
            """Set particle parameters from flattened vector."""
            w_size = particle.weight.numel()
            particle.weight.data.copy_(params[:w_size].view_as(particle.weight))
            particle.bias.data.copy_(params[w_size:].view_as(particle.bias))
        
        def rbf_kernel(theta):
            """RBF kernel with median heuristic."""
            dists_sq = torch.cdist(theta, theta, p=2) ** 2
            h = torch.median(dists_sq) / np.log(n_particles + 1)
            h = torch.clamp(h, min=1e-5)
            return torch.exp(-dists_sq / (2 * h)), h
        
        def log_prior(params):
            """Laplace prior: -|θ|/σ"""
            if self.config.svgd.use_laplace_prior:
                return -torch.sum(torch.abs(params)) / self.config.svgd.prior_std
            else:
                return -0.5 * torch.sum(params ** 2) / (self.config.svgd.prior_std ** 2)
        
        # =====================================================================
        # Training loop
        # =====================================================================
        history = {"epochs": [], "train_loss": [], "diversity": [], "val_accuracy": []}
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_diversity = 0.0
            num_batches = 0
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # ---------------------------------------------------------
                # Extract features (frozen, no gradient)
                # ---------------------------------------------------------
                with torch.no_grad():
                    features = feature_extractor(batch_x)
                
                # ---------------------------------------------------------
                # Collect current particle parameters
                # ---------------------------------------------------------
                theta = torch.stack([get_params(p) for p in particles])
                
                # ---------------------------------------------------------
                # Compute RBF kernel
                # ---------------------------------------------------------
                kernel, bandwidth = rbf_kernel(theta)
                
                # ---------------------------------------------------------
                # Compute gradient of log posterior for each particle
                # ---------------------------------------------------------
                grad_log_post = []
                batch_loss = 0.0
                
                for particle in particles:
                    particle.zero_grad()
                    
                    # Log likelihood (tempered)
                    logits = particle(features)
                    log_lik = -F.cross_entropy(logits, batch_y, reduction='sum') / temperature
                    
                    # Log prior
                    params = get_params(particle)
                    log_p = log_prior(params)
                    
                    # Log posterior
                    log_post = log_lik + log_p
                    log_post.backward()
                    
                    # Collect gradient
                    grad = torch.cat([particle.weight.grad.flatten(), 
                                     particle.bias.grad.flatten()])
                    grad_log_post.append(grad)
                    
                    # Track loss
                    with torch.no_grad():
                        batch_loss += F.cross_entropy(logits, batch_y).item()
                
                grad_log_post = torch.stack(grad_log_post)
                
                # ---------------------------------------------------------
                # Compute SVGD gradient: φ(θᵢ) = (1/n)Σⱼ[k·∇log p + ∇k]
                # ---------------------------------------------------------
                svgd_grad = torch.zeros_like(theta)
                
                for i in range(n_particles):
                    # Attractive: Σⱼ k(θⱼ,θᵢ) · ∇θⱼ log p(θⱼ|D)
                    attractive = (kernel[:, i].unsqueeze(-1) * grad_log_post).sum(dim=0)
                    
                    # Repulsive: Σⱼ ∇θⱼ k(θⱼ,θᵢ) = Σⱼ k·(θᵢ-θⱼ)/h
                    repulsive = (kernel[:, i].unsqueeze(-1) * (theta[i] - theta) / bandwidth).sum(dim=0)
                    
                    svgd_grad[i] = (attractive + repulsive) / n_particles
                
                # ---------------------------------------------------------
                # Update particles: θ ← θ + ε·φ(θ)
                # ---------------------------------------------------------
                with torch.no_grad():
                    for i, particle in enumerate(particles):
                        new_params = theta[i] + self.config.svgd.svgd_lr * svgd_grad[i]
                        set_params(particle, new_params)
                
                epoch_loss += batch_loss / n_particles
                epoch_diversity += torch.cdist(theta, theta).mean().item()
                num_batches += 1
            
            history["epochs"].append(epoch)
            history["train_loss"].append(epoch_loss / num_batches)
            history["diversity"].append(epoch_diversity / num_batches)
            
            # Evaluate periodically
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                val_acc = self._eval_svgd_particles(feature_extractor, particles)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={history['train_loss'][-1]:.4f}, "
                      f"Diversity={history['diversity'][-1]:.4f}, Val Acc={val_acc:.4f}")
        
        training_time = time.time() - start_time
        
        # Compute final metrics
        metrics = self._compute_svgd_metrics_particles(feature_extractor, particles)
        
        self._save_run(run_dir, "svgd", temperature, replicate, metrics, history, training_time)
        self._plot_curves(history, run_name, run_dir, "svgd")
        self._mark_completed(run_dir)
        
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        print(f"✓ {run_name} COMPLETED ({training_time:.1f}s)")
    
    def _eval_svgd_particles(self, feature_extractor, particles) -> float:
        """Evaluate SVGD ensemble accuracy."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                
                # Average predictions across particles
                probs = torch.zeros(len(batch_x), 10, device=self.device)
                for particle in particles:
                    probs += F.softmax(particle(features), dim=-1)
                probs /= len(particles)
                
                correct += (probs.argmax(dim=-1).cpu() == batch_y).sum().item()
                total += len(batch_y)
        
        return correct / total
    
    def _compute_svgd_metrics_particles(self, feature_extractor, particles) -> Dict:
        """Compute all metrics for SVGD ensemble."""
        from sklearn.metrics import roc_auc_score
        
        all_probs, all_labels = [], []
        in_entropy, ood_entropy = [], []
        
        # Test set predictions
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                
                probs = torch.zeros(len(batch_x), 10, device=self.device)
                for particle in particles:
                    probs += F.softmax(particle(features), dim=-1)
                probs /= len(particles)
                
                all_probs.append(probs.cpu())
                all_labels.append(batch_y)
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                in_entropy.append(entropy.cpu())
            
            # OOD entropy
            for batch_x, _ in self.data_manager.ood_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                
                probs = torch.zeros(len(batch_x), 10, device=self.device)
                for particle in particles:
                    probs += F.softmax(particle(features), dim=-1)
                probs /= len(particles)
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                ood_entropy.append(entropy.cpu())
        
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        in_entropy = torch.cat(in_entropy)
        ood_entropy = torch.cat(ood_entropy)
        
        # Error
        error = (all_probs.argmax(-1) != all_labels).float().mean().item()
        
        # NLL
        nll = F.nll_loss(torch.log(all_probs + 1e-10), all_labels).item()
        
        # ECE
        conf, pred = all_probs.max(-1)
        acc = (pred == all_labels).float()
        bins = torch.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            mask = (conf > bins[i]) & (conf <= bins[i+1])
            if mask.sum() > 0:
                ece += torch.abs(acc[mask].mean() - conf[mask].mean()) * mask.float().mean()
        
        # OOD AUROC
        labels = np.concatenate([np.zeros(len(in_entropy)), np.ones(len(ood_entropy))])
        scores = np.concatenate([in_entropy.numpy(), ood_entropy.numpy()])
        ood_auroc = roc_auc_score(labels, scores)
        
        return {"error": error, "nll": nll, "ece": ece.item() if isinstance(ece, torch.Tensor) else ece, "ood_auroc": ood_auroc}
    
    def _train_mfvi(self, temperature: float, replicate: int, num_epochs: int):
        """Train MFVI with frozen ResNet features."""
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
        model = ResNetForBayesianLastLayer(
            num_classes=10, last_layer_type="mfvi",
            prior_log_var=self.config.mfvi.prior_log_var
        ).to(self.device)
        
        # Load pretrained feature extractor
        model_state = model.state_dict()
        for key, value in self.pretrained_weights.items():
            if key.startswith("feature_extractor."):
                model_state[key] = value
        model.load_state_dict(model_state)
        
        # Freeze feature extractor
        model.freeze_feature_extractor()
        
        # Optimizer for last layer only
        optimizer = optim.Adam(model.last_layer.parameters(), lr=self.config.mfvi.learning_rate_init)
        
        history = {"epochs": [], "train_loss": [], "kl_loss": [], "nll_loss": [], "val_accuracy": []}
        
        n_train = len(self.data_manager.train_dataset)
        kl_weight = temperature  # Use temperature as KL weight
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_nll = 0.0
            num_batches = 0
            
            # KL annealing
            if epoch < self.config.mfvi.kl_annealing_epochs:
                current_kl_weight = kl_weight * (epoch + 1) / self.config.mfvi.kl_annealing_epochs
            else:
                current_kl_weight = kl_weight
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                output = model(batch_x, num_samples=1)
                nll = F.cross_entropy(output, batch_y)
                kl = model.last_layer.kl_divergence() / n_train
                loss = nll + current_kl_weight * kl
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_kl += kl.item()
                epoch_nll += nll.item()
                num_batches += 1
            
            history["epochs"].append(epoch)
            history["train_loss"].append(epoch_loss / num_batches)
            history["kl_loss"].append(epoch_kl / num_batches)
            history["nll_loss"].append(epoch_nll / num_batches)
            
            if (epoch + 1) % 40 == 0 or epoch == num_epochs - 1:
                val_acc = self._eval_mfvi(model)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={history['train_loss'][-1]:.4f}, Val Acc={val_acc:.4f}")
        
        training_time = time.time() - start_time
        metrics = self._compute_mfvi_metrics(model)
        
        self._save_run(run_dir, "mfvi", temperature, replicate, metrics, history, training_time)
        self._plot_curves(history, run_name, run_dir, "mfvi")
        self._mark_completed(run_dir)
        
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        print(f"✓ {run_name} COMPLETED ({training_time:.1f}s)")
    
    def _eval_mfvi(self, model) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x, num_samples=10)
                probs = F.softmax(logits, dim=-1).mean(dim=0)
                correct += (probs.argmax(dim=-1).cpu() == batch_y).sum().item()
                total += len(batch_y)
        return correct / total
    
    def _compute_mfvi_metrics(self, model) -> Dict:
        from sklearn.metrics import roc_auc_score
        
        model.eval()
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x, num_samples=10)
                probs = F.softmax(logits, dim=-1).mean(dim=0)
                all_probs.append(probs.cpu())
                all_labels.append(batch_y)
        
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        error = (all_probs.argmax(-1) != all_labels).float().mean().item()
        nll = F.nll_loss(torch.log(all_probs + 1e-10), all_labels).item()
        
        # ECE
        conf, pred = all_probs.max(-1)
        acc = (pred == all_labels).float()
        bins = torch.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            mask = (conf > bins[i]) & (conf <= bins[i+1])
            if mask.sum() > 0:
                ece += torch.abs(acc[mask].mean() - conf[mask].mean()) * mask.float().mean()
        
        # OOD AUROC using entropy
        def get_entropy(loader):
            entropies = []
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x, num_samples=10)
                probs = F.softmax(logits, dim=-1).mean(dim=0)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                entropies.append(ent.cpu())
            return torch.cat(entropies)
        
        in_ent = get_entropy(self.data_manager.test_loader)
        ood_ent = get_entropy(self.data_manager.ood_loader)
        
        labels = np.concatenate([np.zeros(len(in_ent)), np.ones(len(ood_ent))])
        scores = np.concatenate([in_ent.numpy(), ood_ent.numpy()])
        ood_auroc = roc_auc_score(labels, scores)
        
        return {"error": error, "nll": nll, "ece": ece.item() if isinstance(ece, torch.Tensor) else ece, "ood_auroc": ood_auroc}
    
    def _save_run(self, run_dir, method, temperature, replicate, metrics, history, training_time):
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump({
                "method": method, "temperature": temperature, "replicate": replicate + 1,
                "metrics": metrics, "training_time": training_time,
                "training_history": history, "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        hyperparams = asdict(self.config.svgd if method == "svgd" else self.config.mfvi)
        hyperparams["temperature"] = temperature
        with open(os.path.join(run_dir, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def _plot_curves(self, history, run_name, run_dir, method):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(run_name, fontsize=14, fontweight='bold')
        
        ax[0].plot(history["epochs"], history["train_loss"], 'b-', linewidth=2)
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')
        ax[0].grid(True, alpha=0.3)
        
        if method == "svgd":
            ax[1].plot(history["epochs"], history["diversity"], 'g-', linewidth=2)
            ax[1].set_ylabel('Diversity')
            ax[1].set_title('Particle Diversity')
        else:
            ax[1].plot(history["epochs"], history["kl_loss"], 'g-', linewidth=2)
            ax[1].set_ylabel('KL Loss')
            ax[1].set_title('KL Divergence')
        ax[1].set_xlabel('Epoch')
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _mark_completed(self, run_dir):
        with open(os.path.join(run_dir, "COMPLETED"), 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")


# =============================================================================
# STEP 3: Joint Training (Unfreeze Features)
# =============================================================================

class JointTrainer:
    """
    Step 3: Joint training of features + Bayesian last layer.
    
    This step:
        1. Loads trained Bayesian last layer from Step 2
        2. Unfreezes the feature extractor
        3. Continues training with:
           - Feature extractor: SGD with lower LR (e.g., 1e-4)
           - Last layer: SVGD updates (same as Step 2)
        4. Fine-tunes features to improve uncertainty estimates
    
    Why Joint Training Helps:
        - Step 2 keeps features fixed from deterministic training
        - Those features are optimized for point estimates, not uncertainty
        - Joint training allows features to adapt to Bayesian last layer
        - Can improve calibration and OOD detection
    
    Directory Structure:
        {save_dir}/
        ├── deterministic_model/           (from Step 1)
        ├── last_layer_svgd_T0.1_rep_1/    (from Step 2)
        └── joint_svgd_T0.1_replicate_1/   (from Step 3) ← NEW
    """
    
    RUN_NAME_TEMPLATE = "joint_{method}_T{temperature}_replicate_{replicate}"
    
    def __init__(self, config: ExperimentConfig, save_dir: str, device: str = None):
        self.config = config
        self.save_dir = save_dir
        self.device = device or DEVICE
        self.deterministic_dir = os.path.join(save_dir, "deterministic_model")
        
        # Check prerequisites
        if not self._step1_completed():
            raise RuntimeError(
                f"Step 1 not completed. Run Step 1 first:\n"
                f"  python pipeline_resnet.py --step 1 --save-dir {save_dir}"
            )
        
        # Load data (with augmentation for joint training)
        self.data_manager = DataLoaderManager(
            config.data, config.ood, config.calibration, flatten=False
        )
        
        # Load pretrained weights (for feature extractor initialization)
        self.pretrained_weights = torch.load(
            os.path.join(self.deterministic_dir, "model_weights.pt"),
            map_location=self.device
        )
        
        self.completed_runs = self._scan_completed()
        print(f"JointTrainer initialized. Step 2 runs available for joint training.")
    
    def _step1_completed(self) -> bool:
        return os.path.exists(os.path.join(self.deterministic_dir, "COMPLETED"))
    
    def _step2_run_completed(self, method: str, temperature: float, replicate: int) -> bool:
        """Check if Step 2 run exists to use as initialization."""
        step2_name = f"last_layer_{method}_T{temperature}_replicate_{replicate + 1}"
        step2_dir = os.path.join(self.save_dir, step2_name)
        return os.path.exists(os.path.join(step2_dir, "COMPLETED"))
    
    def _scan_completed(self) -> Set[str]:
        """Scan for completed Step 3 runs."""
        completed = set()
        if os.path.exists(self.save_dir):
            for item in os.listdir(self.save_dir):
                if item.startswith("joint_"):
                    if os.path.exists(os.path.join(self.save_dir, item, "COMPLETED")):
                        completed.add(item)
        return completed
    
    @staticmethod
    def get_run_name(method: str, temperature: float, replicate: int) -> str:
        return JointTrainer.RUN_NAME_TEMPLATE.format(
            method=method, temperature=temperature, replicate=replicate + 1
        )
    
    def _get_run_dir(self, method: str, temperature: float, replicate: int) -> str:
        return os.path.join(self.save_dir, self.get_run_name(method, temperature, replicate))
    
    def _is_completed(self, method: str, temperature: float, replicate: int) -> bool:
        run_name = self.get_run_name(method, temperature, replicate)
        return run_name in self.completed_runs
    
    def get_pending(self, methods=["svgd", "mfvi"], temperatures=None, num_replicates=None):
        """Get pending runs (requires Step 2 to be completed first)."""
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        pending = []
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    # Check Step 2 is done AND Step 3 is not done
                    if self._step2_run_completed(method, temp, rep):
                        if not self._is_completed(method, temp, rep):
                            pending.append((method, temp, rep))
        return pending
    
    def get_missing_step2(self, methods=["svgd", "mfvi"], temperatures=None, num_replicates=None):
        """Get Step 2 runs that need to be completed first."""
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        missing = []
        for method in methods:
            for temp in temperatures:
                for rep in range(num_replicates):
                    if not self._step2_run_completed(method, temp, rep):
                        missing.append((method, temp, rep))
        return missing
    
    def train_all(
        self,
        methods=["svgd", "mfvi"],
        temperatures=None,
        num_replicates=None,
        svgd_epochs=50,
        mfvi_epochs=100
    ):
        """Train all pending joint training runs."""
        if temperatures is None:
            temperatures = self.config.temperature.temperatures
        if num_replicates is None:
            num_replicates = self.config.num_replicates
        
        # Check for missing Step 2 runs
        missing = self.get_missing_step2(methods, temperatures, num_replicates)
        if missing:
            print("=" * 70)
            print("WARNING: Some Step 2 runs are not completed yet!")
            print("=" * 70)
            for method, temp, rep in missing[:10]:  # Show first 10
                print(f"  Missing: last_layer_{method}_T{temp}_replicate_{rep+1}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            print("\nRun Step 2 first:")
            print(f"  python pipeline_resnet.py --step 2 --save-dir {self.save_dir}")
            print()
        
        pending = self.get_pending(methods, temperatures, num_replicates)
        total = len(methods) * len(temperatures) * num_replicates
        
        print("=" * 70)
        print("STEP 3: Joint Training (Unfrozen Features)")
        print("=" * 70)
        print(f"Total configurations: {total}")
        print(f"Step 2 completed: {total - len(missing)}")
        print(f"Step 3 completed: {len(self.completed_runs)}")
        print(f"Step 3 pending: {len(pending)}")
        print("=" * 70)
        
        if len(pending) == 0:
            if len(missing) > 0:
                print("Complete Step 2 first before running Step 3.")
            else:
                print("✓ All joint training runs completed!")
            return
        
        print("\nPending joint training runs:")
        for method, temp, rep in pending[:10]:
            print(f"  ○ {self.get_run_name(method, temp, rep)}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        
        for i, (method, temp, rep) in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}]")
            if method == "svgd":
                self._train_joint_svgd(temp, rep, svgd_epochs)
            else:
                self._train_joint_mfvi(temp, rep, mfvi_epochs)
    
    def _train_joint_svgd(self, temperature: float, replicate: int, num_epochs: int):
        """
        Joint training for SVGD: unfreeze features and continue training.
        
        Training Process:
            1. Load Step 2 results (trained particles)
            2. Create fresh feature extractor from Step 1 weights
            3. UNFREEZE feature extractor
            4. Train both:
               - Features: SGD with low LR
               - Particles: SVGD updates
            5. Backprop from particle losses through features
        """
        run_name = self.get_run_name("svgd", temperature, replicate)
        run_dir = self._get_run_dir("svgd", temperature, replicate)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Joint Training: {run_name}")
        print(f"{'='*60}")
        
        torch.manual_seed(SEED + replicate + 1000)  # Different seed from Step 2
        np.random.seed(SEED + replicate + 1000)
        
        start_time = time.time()
        
        # =====================================================================
        # Step 1: Load Step 2 particle weights
        # =====================================================================
        step2_name = f"last_layer_svgd_T{temperature}_replicate_{replicate + 1}"
        step2_dir = os.path.join(self.save_dir, step2_name)
        
        with open(os.path.join(step2_dir, "results.json"), 'r') as f:
            step2_results = json.load(f)
        
        print(f"Loaded Step 2 results from: {step2_name}")
        print(f"  Step 2 final accuracy: {step2_results['metrics'].get('accuracy', 1 - step2_results['metrics']['error']):.4f}")
        
        # =====================================================================
        # Step 2: Create feature extractor (UNFROZEN this time)
        # =====================================================================
        feature_extractor = ResNetFeatureExtractor().to(self.device)
        
        fe_state = {
            k.replace("feature_extractor.", ""): v 
            for k, v in self.pretrained_weights.items() 
            if k.startswith("feature_extractor.")
        }
        feature_extractor.load_state_dict(fe_state)
        
        # IMPORTANT: Do NOT freeze - this is joint training
        feature_extractor.train()
        
        # Create optimizer for feature extractor
        feature_optimizer = optim.SGD(
            feature_extractor.parameters(),
            lr=self.config.svgd.feature_lr,
            momentum=0.9,
            weight_decay=self.config.svgd.weight_decay
        )
        feature_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            feature_optimizer, T_max=num_epochs
        )
        
        print(f"Feature extractor UNFROZEN, LR={self.config.svgd.feature_lr}")
        
        # =====================================================================
        # Step 3: Create particles (initialized from pretrained, will diverge)
        # =====================================================================
        n_particles = self.config.svgd.n_particles
        
        pretrained_weight = self.pretrained_weights["last_layer.fc.weight"].clone()
        pretrained_bias = self.pretrained_weights["last_layer.fc.bias"].clone()
        
        particles = []
        for i in range(n_particles):
            particle = nn.Linear(512, 10).to(self.device)
            particle.weight.data.copy_(pretrained_weight)
            particle.bias.data.copy_(pretrained_bias)
            if i > 0:
                with torch.no_grad():
                    particle.weight.data.add_(torch.randn_like(particle.weight) * self.config.svgd.init_std)
                    particle.bias.data.add_(torch.randn_like(particle.bias) * self.config.svgd.init_std)
            particles.append(particle)
        
        print(f"Created {n_particles} particles")
        
        # =====================================================================
        # Helper functions
        # =====================================================================
        def get_params(particle):
            return torch.cat([particle.weight.flatten(), particle.bias.flatten()])
        
        def set_params(particle, params):
            w_size = particle.weight.numel()
            particle.weight.data.copy_(params[:w_size].view_as(particle.weight))
            particle.bias.data.copy_(params[w_size:].view_as(particle.bias))
        
        def rbf_kernel(theta):
            dists_sq = torch.cdist(theta, theta, p=2) ** 2
            h = torch.median(dists_sq) / np.log(n_particles + 1)
            h = torch.clamp(h, min=1e-5)
            return torch.exp(-dists_sq / (2 * h)), h
        
        def log_prior(params):
            if self.config.svgd.use_laplace_prior:
                return -torch.sum(torch.abs(params)) / self.config.svgd.prior_std
            else:
                return -0.5 * torch.sum(params ** 2) / (self.config.svgd.prior_std ** 2)
        
        # =====================================================================
        # Joint Training Loop
        # =====================================================================
        history = {"epochs": [], "train_loss": [], "diversity": [], "val_accuracy": [], "feature_lr": []}
        
        for epoch in range(num_epochs):
            feature_extractor.train()  # Training mode (BatchNorm updates)
            epoch_loss = 0.0
            epoch_diversity = 0.0
            num_batches = 0
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # ---------------------------------------------------------
                # Forward through feature extractor (WITH gradient)
                # ---------------------------------------------------------
                feature_optimizer.zero_grad()
                features = feature_extractor(batch_x)  # Gradient flows through
                
                # ---------------------------------------------------------
                # Collect particle parameters
                # ---------------------------------------------------------
                theta = torch.stack([get_params(p) for p in particles])
                kernel, bandwidth = rbf_kernel(theta)
                
                # ---------------------------------------------------------
                # Compute gradients for SVGD + feature loss
                # ---------------------------------------------------------
                grad_log_post = []
                total_particle_loss = 0.0
                
                for particle in particles:
                    particle.zero_grad()
                    
                    logits = particle(features)
                    log_lik = -F.cross_entropy(logits, batch_y, reduction='sum') / temperature
                    params = get_params(particle)
                    log_p = log_prior(params)
                    log_post = log_lik + log_p
                    
                    # Backward for particle gradient
                    log_post.backward(retain_graph=True)
                    
                    grad = torch.cat([particle.weight.grad.flatten(), 
                                     particle.bias.grad.flatten()])
                    grad_log_post.append(grad)
                    
                    with torch.no_grad():
                        total_particle_loss += F.cross_entropy(logits, batch_y).item()
                
                grad_log_post = torch.stack(grad_log_post)
                
                # ---------------------------------------------------------
                # SVGD update on particles
                # ---------------------------------------------------------
                svgd_grad = torch.zeros_like(theta)
                for i in range(n_particles):
                    attractive = (kernel[:, i].unsqueeze(-1) * grad_log_post).sum(dim=0)
                    repulsive = (kernel[:, i].unsqueeze(-1) * (theta[i] - theta) / bandwidth).sum(dim=0)
                    svgd_grad[i] = (attractive + repulsive) / n_particles
                
                with torch.no_grad():
                    for i, particle in enumerate(particles):
                        new_params = theta[i] + self.config.svgd.svgd_lr * svgd_grad[i]
                        set_params(particle, new_params)
                
                # ---------------------------------------------------------
                # Update feature extractor
                # ---------------------------------------------------------
                # Compute fresh loss for feature gradient
                features = feature_extractor(batch_x)
                feature_loss = 0.0
                for particle in particles:
                    feature_loss += F.cross_entropy(particle(features), batch_y)
                feature_loss = feature_loss / n_particles
                
                feature_loss.backward()
                torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
                feature_optimizer.step()
                
                epoch_loss += total_particle_loss / n_particles
                epoch_diversity += torch.cdist(theta, theta).mean().item()
                num_batches += 1
            
            feature_scheduler.step()
            
            history["epochs"].append(epoch)
            history["train_loss"].append(epoch_loss / num_batches)
            history["diversity"].append(epoch_diversity / num_batches)
            history["feature_lr"].append(feature_scheduler.get_last_lr()[0])
            
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                feature_extractor.eval()
                val_acc = self._eval_joint_svgd(feature_extractor, particles)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={history['train_loss'][-1]:.4f}, "
                      f"Diversity={history['diversity'][-1]:.4f}, Val Acc={val_acc:.4f}, "
                      f"FE_LR={history['feature_lr'][-1]:.6f}")
        
        training_time = time.time() - start_time
        
        feature_extractor.eval()
        metrics = self._compute_joint_svgd_metrics(feature_extractor, particles)
        
        # Save
        self._save_run(run_dir, "svgd", temperature, replicate, metrics, history, training_time, 
                      step2_accuracy=step2_results['metrics'].get('accuracy', 1 - step2_results['metrics']['error']))
        self._plot_curves(history, run_name, run_dir, "svgd")
        self._mark_completed(run_dir)
        
        improvement = metrics['accuracy'] - step2_results['metrics'].get('accuracy', 1 - step2_results['metrics']['error'])
        print(f"Results: Error={metrics['error']:.4f}, NLL={metrics['nll']:.4f}, "
              f"ECE={metrics['ece']:.4f}, OOD AUROC={metrics['ood_auroc']:.4f}")
        print(f"Accuracy improvement from Step 2: {improvement:+.4f}")
        print(f"✓ {run_name} COMPLETED ({training_time:.1f}s)")
    
    def _train_joint_mfvi(self, temperature: float, replicate: int, num_epochs: int):
        """Joint training for MFVI: unfreeze features and continue training."""
        run_name = self.get_run_name("mfvi", temperature, replicate)
        run_dir = self._get_run_dir("mfvi", temperature, replicate)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Joint Training: {run_name}")
        print(f"{'='*60}")
        
        torch.manual_seed(SEED + replicate + 1000)
        np.random.seed(SEED + replicate + 1000)
        
        start_time = time.time()
        
        # Load Step 2 results
        step2_name = f"last_layer_mfvi_T{temperature}_replicate_{replicate + 1}"
        step2_dir = os.path.join(self.save_dir, step2_name)
        
        with open(os.path.join(step2_dir, "results.json"), 'r') as f:
            step2_results = json.load(f)
        
        print(f"Loaded Step 2 results: accuracy={step2_results['metrics'].get('accuracy', 1 - step2_results['metrics']['error']):.4f}")
        
        # Create MFVI model (UNFROZEN)
        model = ResNetForBayesianLastLayer(
            num_classes=10, last_layer_type="mfvi",
            prior_log_var=self.config.mfvi.prior_log_var
        ).to(self.device)
        
        # Load pretrained feature extractor
        model_state = model.state_dict()
        for key, value in self.pretrained_weights.items():
            if key.startswith("feature_extractor."):
                model_state[key] = value
        model.load_state_dict(model_state)
        
        # Do NOT freeze - joint training
        model.train()
        
        # Separate optimizers for features and last layer
        feature_optimizer = optim.SGD(
            model.feature_extractor.parameters(),
            lr=self.config.svgd.feature_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        last_layer_optimizer = optim.Adam(
            model.last_layer.parameters(),
            lr=self.config.mfvi.learning_rate_init
        )
        
        feature_scheduler = optim.lr_scheduler.CosineAnnealingLR(feature_optimizer, T_max=num_epochs)
        
        print(f"Feature extractor UNFROZEN, LR={self.config.svgd.feature_lr}")
        
        history = {"epochs": [], "train_loss": [], "kl_loss": [], "nll_loss": [], "val_accuracy": []}
        
        n_train = len(self.data_manager.train_dataset)
        kl_weight = temperature
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_nll = 0.0
            num_batches = 0
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                feature_optimizer.zero_grad()
                last_layer_optimizer.zero_grad()
                
                output = model(batch_x, num_samples=1)
                nll = F.cross_entropy(output, batch_y)
                kl = model.last_layer.kl_divergence() / n_train
                loss = nll + kl_weight * kl
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                feature_optimizer.step()
                last_layer_optimizer.step()
                
                epoch_loss += loss.item()
                epoch_kl += kl.item()
                epoch_nll += nll.item()
                num_batches += 1
            
            feature_scheduler.step()
            
            history["epochs"].append(epoch)
            history["train_loss"].append(epoch_loss / num_batches)
            history["kl_loss"].append(epoch_kl / num_batches)
            history["nll_loss"].append(epoch_nll / num_batches)
            
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                val_acc = self._eval_mfvi(model)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={history['train_loss'][-1]:.4f}, Val Acc={val_acc:.4f}")
        
        training_time = time.time() - start_time
        metrics = self._compute_mfvi_metrics(model)
        metrics['accuracy'] = 1 - metrics['error']
        
        self._save_run(run_dir, "mfvi", temperature, replicate, metrics, history, training_time,
                      step2_accuracy=step2_results['metrics'].get('accuracy', 1 - step2_results['metrics']['error']))
        self._plot_curves(history, run_name, run_dir, "mfvi")
        self._mark_completed(run_dir)
        
        improvement = metrics['accuracy'] - step2_results['metrics'].get('accuracy', 1 - step2_results['metrics']['error'])
        print(f"Results: Error={metrics['error']:.4f}, Improvement from Step 2: {improvement:+.4f}")
        print(f"✓ {run_name} COMPLETED ({training_time:.1f}s)")
    
    def _eval_joint_svgd(self, feature_extractor, particles) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                probs = torch.zeros(len(batch_x), 10, device=self.device)
                for particle in particles:
                    probs += F.softmax(particle(features), dim=-1)
                probs /= len(particles)
                correct += (probs.argmax(dim=-1).cpu() == batch_y).sum().item()
                total += len(batch_y)
        return correct / total
    
    def _eval_mfvi(self, model) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x, num_samples=10)
                probs = F.softmax(logits, dim=-1).mean(dim=0)
                correct += (probs.argmax(dim=-1).cpu() == batch_y).sum().item()
                total += len(batch_y)
        return correct / total
    
    def _compute_joint_svgd_metrics(self, feature_extractor, particles) -> Dict:
        from sklearn.metrics import roc_auc_score
        
        all_probs, all_labels = [], []
        in_entropy, ood_entropy = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                probs = torch.zeros(len(batch_x), 10, device=self.device)
                for particle in particles:
                    probs += F.softmax(particle(features), dim=-1)
                probs /= len(particles)
                all_probs.append(probs.cpu())
                all_labels.append(batch_y)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                in_entropy.append(entropy.cpu())
            
            for batch_x, _ in self.data_manager.ood_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                probs = torch.zeros(len(batch_x), 10, device=self.device)
                for particle in particles:
                    probs += F.softmax(particle(features), dim=-1)
                probs /= len(particles)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                ood_entropy.append(entropy.cpu())
        
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        in_entropy = torch.cat(in_entropy)
        ood_entropy = torch.cat(ood_entropy)
        
        error = (all_probs.argmax(-1) != all_labels).float().mean().item()
        nll = F.nll_loss(torch.log(all_probs + 1e-10), all_labels).item()
        
        conf, pred = all_probs.max(-1)
        acc = (pred == all_labels).float()
        bins = torch.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            mask = (conf > bins[i]) & (conf <= bins[i+1])
            if mask.sum() > 0:
                ece += torch.abs(acc[mask].mean() - conf[mask].mean()) * mask.float().mean()
        
        labels = np.concatenate([np.zeros(len(in_entropy)), np.ones(len(ood_entropy))])
        scores = np.concatenate([in_entropy.numpy(), ood_entropy.numpy()])
        ood_auroc = roc_auc_score(labels, scores)
        
        return {"error": error, "nll": nll, "ece": ece.item() if isinstance(ece, torch.Tensor) else ece, 
                "ood_auroc": ood_auroc, "accuracy": 1 - error}
    
    def _compute_mfvi_metrics(self, model) -> Dict:
        from sklearn.metrics import roc_auc_score
        
        model.eval()
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x, num_samples=10)
                probs = F.softmax(logits, dim=-1).mean(dim=0)
                all_probs.append(probs.cpu())
                all_labels.append(batch_y)
        
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        error = (all_probs.argmax(-1) != all_labels).float().mean().item()
        nll = F.nll_loss(torch.log(all_probs + 1e-10), all_labels).item()
        
        conf, pred = all_probs.max(-1)
        acc = (pred == all_labels).float()
        bins = torch.linspace(0, 1, 16)
        ece = 0.0
        for i in range(15):
            mask = (conf > bins[i]) & (conf <= bins[i+1])
            if mask.sum() > 0:
                ece += torch.abs(acc[mask].mean() - conf[mask].mean()) * mask.float().mean()
        
        def get_entropy(loader):
            entropies = []
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x, num_samples=10)
                probs = F.softmax(logits, dim=-1).mean(dim=0)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                entropies.append(ent.cpu())
            return torch.cat(entropies)
        
        in_ent = get_entropy(self.data_manager.test_loader)
        ood_ent = get_entropy(self.data_manager.ood_loader)
        
        labels = np.concatenate([np.zeros(len(in_ent)), np.ones(len(ood_ent))])
        scores = np.concatenate([in_ent.numpy(), ood_ent.numpy()])
        ood_auroc = roc_auc_score(labels, scores)
        
        return {"error": error, "nll": nll, "ece": ece.item() if isinstance(ece, torch.Tensor) else ece, "ood_auroc": ood_auroc}
    
    def _save_run(self, run_dir, method, temperature, replicate, metrics, history, training_time, step2_accuracy=None):
        results = {
            "method": method, "temperature": temperature, "replicate": replicate + 1,
            "metrics": metrics, "training_time": training_time,
            "training_history": history, "timestamp": datetime.now().isoformat(),
            "step2_accuracy": step2_accuracy,
            "improvement_from_step2": metrics.get('accuracy', 1 - metrics['error']) - step2_accuracy if step2_accuracy else None
        }
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        hyperparams = asdict(self.config.svgd if method == "svgd" else self.config.mfvi)
        hyperparams["temperature"] = temperature
        hyperparams["joint_training"] = True
        with open(os.path.join(run_dir, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def _plot_curves(self, history, run_name, run_dir, method):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{run_name} (Joint Training)', fontsize=14, fontweight='bold')
        
        ax[0].plot(history["epochs"], history["train_loss"], 'b-', linewidth=2)
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')
        ax[0].grid(True, alpha=0.3)
        
        if method == "svgd" and "diversity" in history:
            ax[1].plot(history["epochs"], history["diversity"], 'g-', linewidth=2)
            ax[1].set_ylabel('Diversity')
            ax[1].set_title('Particle Diversity')
        else:
            ax[1].plot(history["epochs"], history["kl_loss"], 'g-', linewidth=2)
            ax[1].set_ylabel('KL Loss')
            ax[1].set_title('KL Divergence')
        ax[1].set_xlabel('Epoch')
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _mark_completed(self, run_dir):
        with open(os.path.join(run_dir, "COMPLETED"), 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")


# =============================================================================
# Status
# =============================================================================

def print_status(save_dir: str, config: ExperimentConfig):
    print("=" * 70)
    print("EXPERIMENT STATUS")
    print("=" * 70)
    
    det_dir = os.path.join(save_dir, "deterministic_model")
    det_done = os.path.exists(os.path.join(det_dir, "COMPLETED"))
    
    print("\nSTEP 1: ResNet-18 Training")
    if det_done:
        print("  ✓ COMPLETED")
        try:
            with open(os.path.join(det_dir, "metrics.json")) as f:
                m = json.load(f)
            print(f"    Test Accuracy: {m.get('accuracy', 'N/A'):.4f}")
        except:
            pass
    else:
        print("  ○ NOT STARTED")
        print(f"    Run: python pipeline_resnet.py --step 1 --save-dir {save_dir}")
    
    print("\nSTEP 2: Bayesian Last Layer (Frozen Features)")
    if not det_done:
        print("  (Requires Step 1 first)")
    else:
        temps = config.temperature.temperatures
        reps = config.num_replicates
        total = 2 * len(temps) * reps
        
        step2_completed = 0
        for method in ["svgd", "mfvi"]:
            for t in temps:
                for r in range(reps):
                    name = f"last_layer_{method}_T{t}_replicate_{r+1}"
                    if os.path.exists(os.path.join(save_dir, name, "COMPLETED")):
                        step2_completed += 1
        
        print(f"  Completed: {step2_completed}/{total}")
        if step2_completed < total:
            print(f"    Run: python pipeline_resnet.py --step 2 --save-dir {save_dir}")
        else:
            print("  ✓ ALL COMPLETED")
    
    print("\nSTEP 3: Joint Training (Unfrozen Features)")
    if not det_done:
        print("  (Requires Step 1 first)")
    else:
        temps = config.temperature.temperatures
        reps = config.num_replicates
        total = 2 * len(temps) * reps
        
        step3_completed = 0
        step2_available = 0
        for method in ["svgd", "mfvi"]:
            for t in temps:
                for r in range(reps):
                    step2_name = f"last_layer_{method}_T{t}_replicate_{r+1}"
                    step3_name = f"joint_{method}_T{t}_replicate_{r+1}"
                    
                    if os.path.exists(os.path.join(save_dir, step2_name, "COMPLETED")):
                        step2_available += 1
                    if os.path.exists(os.path.join(save_dir, step3_name, "COMPLETED")):
                        step3_completed += 1
        
        print(f"  Step 2 runs available: {step2_available}/{total}")
        print(f"  Step 3 completed: {step3_completed}/{total}")
        if step3_completed < step2_available:
            print(f"    Run: python pipeline_resnet.py --step 3 --save-dir {save_dir}")
        elif step2_available == total and step3_completed == total:
            print("  ✓ ALL COMPLETED")
        else:
            print("  (Complete Step 2 first)")
    
    print()
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ResNet-based BNN pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs for Step 1")
    parser.add_argument("--svgd-epochs", type=int, default=100, help="Epochs for Step 2 SVGD")
    parser.add_argument("--mfvi-epochs", type=int, default=200, help="Epochs for Step 2 MFVI")
    parser.add_argument("--joint-svgd-epochs", type=int, default=50, help="Epochs for Step 3 SVGD")
    parser.add_argument("--joint-mfvi-epochs", type=int, default=100, help="Epochs for Step 3 MFVI")
    
    args = parser.parse_args()
    
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
        print()
        print_status(args.save_dir, config)
        return
    
    if args.step == 1:
        trainer = DeterministicResNetTrainer(config, args.save_dir)
        trainer.train(num_epochs=args.epochs)
    
    elif args.step == 2:
        trainer = BayesianLastLayerTrainer(config, args.save_dir)
        trainer.train_all(svgd_epochs=args.svgd_epochs, mfvi_epochs=args.mfvi_epochs)
    
    elif args.step == 3:
        trainer = JointTrainer(config, args.save_dir)
        trainer.train_all(svgd_epochs=args.joint_svgd_epochs, mfvi_epochs=args.joint_mfvi_epochs)


if __name__ == "__main__":
    main()
