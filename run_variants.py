"""
Variant-Based Pipeline Runner

This script runs the BNN training pipeline using model variants defined in model_variants.py.
It provides more flexible configuration than the original pipeline.

Usage:
    # List all variants
    python run_variants.py --list
    
    # Run all variants for Step 2
    python run_variants.py --step 2 --save-dir ./results
    
    # Run specific variants
    python run_variants.py --step 2 --save-dir ./results --variants svgd_laplace_std1 svgd_gauss_std1
    
    # Run single configuration
    python run_variants.py --step 2 --save-dir ./results --variant svgd_laplace_std1 --temperature 0.1 --replicate 1
    
    # Check status
    python run_variants.py --status --save-dir ./results
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional

# Local imports
from config import ExperimentConfig, DEVICE, SEED
from data_loading_resnet import DataLoaderManager
from resnet import (
    ResNetFeatureExtractor, 
    DeterministicLastLayer, 
    BayesianLastLayerMFVI,
    ResNetForBayesianLastLayer
)
from model_variants import (
    ModelVariant, 
    MODEL_VARIANTS, 
    get_variant, 
    get_active_variants,
    get_variant_names,
    print_variants,
    TEMPERATURES,
    NUM_REPLICATES
)


# =============================================================================
# Variant Trainer
# =============================================================================

class VariantTrainer:
    """
    Train models based on ModelVariant configurations.
    """
    
    def __init__(self, save_dir: str, device: str = None):
        self.save_dir = save_dir
        self.device = device or DEVICE
        self.deterministic_dir = os.path.join(save_dir, "deterministic_model")
        
        # Check deterministic model exists
        if not os.path.exists(os.path.join(self.deterministic_dir, "COMPLETED")):
            raise RuntimeError(
                f"No deterministic model found. Run Step 1 first:\n"
                f"  python pipeline_resnet.py --step 1 --save-dir {save_dir}"
            )
        
        # Load config for data
        self.config = ExperimentConfig()
        
        # Load data
        self.data_manager = DataLoaderManager(
            self.config.data, self.config.ood, self.config.calibration,
            flatten=False, save_dir=save_dir
        )
        
        # Load pretrained weights
        self.pretrained_weights = torch.load(
            os.path.join(self.deterministic_dir, "model_weights.pt"),
            map_location=self.device
        )
        print(f"Loaded pretrained weights from {self.deterministic_dir}")
        
        # Scan completed runs
        self.completed_runs = self._scan_completed()
    
    def _scan_completed(self) -> Set[str]:
        """Scan for completed runs."""
        completed = set()
        if os.path.exists(self.save_dir):
            for item in os.listdir(self.save_dir):
                completed_path = os.path.join(self.save_dir, item, "COMPLETED")
                if os.path.exists(completed_path):
                    completed.add(item)
        return completed
    
    def _is_completed(self, run_name: str) -> bool:
        return run_name in self.completed_runs
    
    def _mark_completed(self, run_dir: str):
        with open(os.path.join(run_dir, "COMPLETED"), 'w') as f:
            f.write(datetime.now().isoformat())
    
    def get_pending_runs(
        self, 
        variants: List[ModelVariant] = None,
        temperatures: List[float] = None,
        num_replicates: int = None
    ) -> List[Tuple[ModelVariant, float, int]]:
        """Get list of pending (variant, temperature, replicate) tuples."""
        
        if variants is None:
            variants = get_active_variants()
        if temperatures is None:
            temperatures = TEMPERATURES
        if num_replicates is None:
            num_replicates = NUM_REPLICATES
        
        pending = []
        for variant in variants:
            for temp in temperatures:
                for rep in range(1, num_replicates + 1):
                    run_name = variant.get_run_name(temp, rep)
                    if not self._is_completed(run_name):
                        pending.append((variant, temp, rep))
        
        return pending
    
    def train_variant(
        self, 
        variant: ModelVariant, 
        temperature: float, 
        replicate: int,
        num_epochs: int = None
    ):
        """Train a single variant configuration."""
        
        run_name = variant.get_run_name(temperature, replicate)
        run_dir = os.path.join(self.save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Training: {run_name}")
        print(f"  Method: {variant.method}")
        print(f"  Prior: {variant.prior_type} (std={variant.prior_std})")
        if variant.method == 'svgd':
            print(f"  Particles: {variant.n_particles}")
        print(f"  Temperature: {temperature}")
        print(f"{'='*70}")
        
        # Set seeds
        torch.manual_seed(SEED + replicate)
        np.random.seed(SEED + replicate)
        
        start_time = time.time()
        
        if variant.method == 'svgd':
            metrics, history = self._train_svgd_variant(variant, temperature, num_epochs)
        else:
            metrics, history = self._train_mfvi_variant(variant, temperature, num_epochs)
        
        training_time = time.time() - start_time
        
        # Save results
        self._save_results(run_dir, variant, temperature, replicate, metrics, history, training_time)
        self._mark_completed(run_dir)
        
        # Print results
        print(f"\nResults:")
        print(f"  Error: {metrics['error']*100:.2f}%")
        print(f"  NLL: {metrics['nll']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  OOD AUROC: {metrics['ood_auroc']:.4f}")
        if 'mean_in_epistemic_entropy' in metrics:
            print(f"  Epistemic (in-dist): {metrics['mean_in_epistemic_entropy']:.4f}")
            print(f"  Epistemic (OOD): {metrics['mean_ood_epistemic_entropy']:.4f}")
        print(f"✓ Completed in {training_time:.1f}s")
        
        return metrics
    
    def _train_svgd_variant(
        self, 
        variant: ModelVariant, 
        temperature: float,
        num_epochs: int = None
    ) -> Tuple[Dict, Dict]:
        """Train SVGD with variant configuration."""
        
        num_epochs = num_epochs or variant.step2_epochs
        n_particles = variant.n_particles
        
        # Create feature extractor (frozen)
        feature_extractor = ResNetFeatureExtractor().to(self.device)
        fe_weights = {k.replace('feature_extractor.', ''): v 
                      for k, v in self.pretrained_weights.items() 
                      if k.startswith('feature_extractor.')}
        feature_extractor.load_state_dict(fe_weights)
        feature_extractor.eval()
        for p in feature_extractor.parameters():
            p.requires_grad = False
        
        # Initialize particles from pretrained last layer
        pretrained_ll = {k.replace('last_layer.', ''): v 
                        for k, v in self.pretrained_weights.items() 
                        if k.startswith('last_layer.')}
        
        particles = []
        for i in range(n_particles):
            particle = DeterministicLastLayer(512, 10).to(self.device)
            particle.load_state_dict(pretrained_ll)
            # Add small perturbation
            with torch.no_grad():
                particle.weight.data += torch.randn_like(particle.weight) * 0.01
                particle.bias.data += torch.randn_like(particle.bias) * 0.01
            particles.append(particle)
        
        # Helper functions
        def get_particle_params(particle):
            return torch.cat([particle.weight.view(-1), particle.bias.view(-1)])
        
        def set_particle_params(particle, params):
            w_size = particle.weight.numel()
            particle.weight.data.copy_(params[:w_size].view_as(particle.weight))
            particle.bias.data.copy_(params[w_size:].view_as(particle.bias))
        
        def rbf_kernel(theta):
            dists_sq = torch.cdist(theta, theta, p=2) ** 2
            h = torch.median(dists_sq) / np.log(n_particles + 1)
            h = torch.clamp(h, min=1e-5)
            return torch.exp(-dists_sq / (2 * h)), h
        
        def log_prior(params):
            if variant.prior_type == "laplace":
                return -torch.sum(torch.abs(params)) / variant.prior_std
            else:  # gaussian
                return -0.5 * torch.sum(params ** 2) / (variant.prior_std ** 2)
        
        # Training loop
        history = {"epochs": [], "train_loss": [], "diversity": [], "val_accuracy": []}
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_diversity = 0.0
            n_batches = 0
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                with torch.no_grad():
                    features = feature_extractor(batch_x)
                
                # Get current particle positions
                theta = torch.stack([get_particle_params(p) for p in particles])
                theta.requires_grad = True
                
                # Compute log posterior gradients
                grad_log_p = []
                for i, particle in enumerate(particles):
                    set_particle_params(particle, theta[i])
                    logits = particle(features)
                    log_lik = -F.cross_entropy(logits, batch_y, reduction='sum')
                    log_p = log_lik / temperature + log_prior(theta[i])
                    
                    grad = torch.autograd.grad(log_p, theta, retain_graph=True)[0][i]
                    grad_log_p.append(grad)
                
                grad_log_p = torch.stack(grad_log_p)
                
                # RBF kernel
                K, h = rbf_kernel(theta.detach())
                
                # Kernel gradients
                grad_K = []
                for i in range(n_particles):
                    diff = theta.detach()[i:i+1] - theta.detach()
                    grad_k_i = (K[i:i+1, :].T * diff / h).sum(dim=0)
                    grad_K.append(grad_k_i)
                grad_K = torch.stack(grad_K)
                
                # SVGD update
                svgd_grad = (K @ grad_log_p + grad_K) / n_particles
                
                # Apply update
                with torch.no_grad():
                    for i, particle in enumerate(particles):
                        new_params = theta[i] + variant.svgd_lr * svgd_grad[i]
                        new_params = torch.clamp(new_params, -10, 10)
                        set_particle_params(particle, new_params)
                
                # Track metrics
                with torch.no_grad():
                    avg_logits = sum(p(features) for p in particles) / n_particles
                    loss = F.cross_entropy(avg_logits, batch_y)
                    epoch_loss += loss.item()
                    epoch_diversity += h.item()
                    n_batches += 1
            
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(epoch_loss / n_batches)
            history["diversity"].append(epoch_diversity / n_batches)
            
            if (epoch + 1) % 10 == 0:
                val_acc = self._eval_svgd(feature_extractor, particles)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={history['train_loss'][-1]:.4f}, "
                      f"Diversity={history['diversity'][-1]:.4f}, Val Acc={val_acc:.4f}")
        
        # Compute final metrics
        metrics = self._compute_svgd_metrics(feature_extractor, particles)
        
        # Save particles
        run_name = variant.get_run_name(0, 0)  # temp name, will be overwritten
        run_dir = os.path.join(self.save_dir, run_name)
        particles_state = [p.state_dict() for p in particles]
        # Note: actual save happens in train_variant
        self._temp_particles = particles_state
        
        return metrics, history
    
    def _train_mfvi_variant(
        self, 
        variant: ModelVariant, 
        temperature: float,
        num_epochs: int = None
    ) -> Tuple[Dict, Dict]:
        """Train MFVI with variant configuration."""
        
        num_epochs = num_epochs or variant.step2_epochs
        
        # Create model
        model = ResNetForBayesianLastLayer(
            feature_extractor=ResNetFeatureExtractor(),
            bayesian_layer=BayesianLastLayerMFVI(512, 10, prior_log_var=2*np.log(variant.prior_std))
        ).to(self.device)
        
        # Load pretrained features
        fe_weights = {k.replace('feature_extractor.', ''): v 
                      for k, v in self.pretrained_weights.items() 
                      if k.startswith('feature_extractor.')}
        model.feature_extractor.load_state_dict(fe_weights)
        model.freeze_feature_extractor()
        
        # Initialize MFVI from pretrained
        pretrained_ll = {k.replace('last_layer.', ''): v 
                        for k, v in self.pretrained_weights.items() 
                        if k.startswith('last_layer.')}
        model.last_layer.weight_mu.data.copy_(pretrained_ll['weight'])
        model.last_layer.bias_mu.data.copy_(pretrained_ll['bias'])
        
        optimizer = torch.optim.Adam(model.last_layer.parameters(), lr=variant.mfvi_lr)
        
        history = {"epochs": [], "train_loss": [], "kl": [], "val_accuracy": []}
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_kl = 0.0
            n_batches = 0
            
            # KL annealing
            kl_weight = min(1.0, (epoch + 1) / variant.kl_annealing_epochs)
            
            for batch_x, batch_y in self.data_manager.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                logits = model(batch_x, num_samples=1)
                nll = F.cross_entropy(logits, batch_y)
                kl = model.last_layer.kl_divergence() / len(self.data_manager.train_loader.dataset)
                
                loss = nll + temperature * kl_weight * kl
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_kl += kl.item()
                n_batches += 1
            
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(epoch_loss / n_batches)
            history["kl"].append(epoch_kl / n_batches)
            
            if (epoch + 1) % 20 == 0:
                val_acc = self._eval_mfvi(model)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={history['train_loss'][-1]:.4f}, "
                      f"KL={history['kl'][-1]:.6f}, Val Acc={val_acc:.4f}")
        
        # Compute metrics
        metrics = self._compute_mfvi_metrics(model)
        
        # Save MFVI layer
        self._temp_mfvi_layer = model.last_layer.state_dict()
        
        return metrics, history
    
    def _eval_svgd(self, feature_extractor, particles) -> float:
        """Evaluate SVGD on validation set."""
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.val_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                probs = sum(F.softmax(p(features), dim=-1) for p in particles) / len(particles)
                correct += (probs.argmax(dim=-1).cpu() == batch_y).sum().item()
                total += len(batch_y)
        return correct / total
    
    def _eval_mfvi(self, model) -> float:
        """Evaluate MFVI on validation set."""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_manager.val_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x, num_samples=10)
                probs = F.softmax(logits, dim=-1).mean(dim=0)
                correct += (probs.argmax(dim=-1).cpu() == batch_y).sum().item()
                total += len(batch_y)
        return correct / total
    
    def _compute_svgd_metrics(self, feature_extractor, particles) -> Dict:
        """Compute full metrics for SVGD."""
        from sklearn.metrics import roc_auc_score
        
        all_probs, all_labels = [], []
        in_total, in_aleatoric, in_epistemic = [], [], []
        ood_total, ood_aleatoric, ood_epistemic = [], [], []
        
        n_particles = len(particles)
        
        with torch.no_grad():
            # In-distribution
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                
                particle_probs = torch.stack([F.softmax(p(features), dim=-1) for p in particles])
                mean_probs = particle_probs.mean(dim=0)
                
                total_ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
                indiv_ent = -torch.sum(particle_probs * torch.log(particle_probs + 1e-10), dim=-1)
                aleat_ent = indiv_ent.mean(dim=0)
                epist_ent = total_ent - aleat_ent
                
                all_probs.append(mean_probs.cpu())
                all_labels.append(batch_y)
                in_total.append(total_ent.cpu())
                in_aleatoric.append(aleat_ent.cpu())
                in_epistemic.append(epist_ent.cpu())
            
            # OOD
            for batch_x, _ in self.data_manager.ood_loader:
                batch_x = batch_x.to(self.device)
                features = feature_extractor(batch_x)
                
                particle_probs = torch.stack([F.softmax(p(features), dim=-1) for p in particles])
                mean_probs = particle_probs.mean(dim=0)
                
                total_ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
                indiv_ent = -torch.sum(particle_probs * torch.log(particle_probs + 1e-10), dim=-1)
                aleat_ent = indiv_ent.mean(dim=0)
                epist_ent = total_ent - aleat_ent
                
                ood_total.append(total_ent.cpu())
                ood_aleatoric.append(aleat_ent.cpu())
                ood_epistemic.append(epist_ent.cpu())
        
        # Concatenate
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        in_total = torch.cat(in_total)
        in_aleatoric = torch.cat(in_aleatoric)
        in_epistemic = torch.cat(in_epistemic)
        ood_total = torch.cat(ood_total)
        ood_aleatoric = torch.cat(ood_aleatoric)
        ood_epistemic = torch.cat(ood_epistemic)
        
        # Metrics
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
        
        # AUROC
        labels = np.concatenate([np.zeros(len(in_total)), np.ones(len(ood_total))])
        
        auroc_total = roc_auc_score(labels, np.concatenate([in_total.numpy(), ood_total.numpy()]))
        auroc_aleatoric = roc_auc_score(labels, np.concatenate([in_aleatoric.numpy(), ood_aleatoric.numpy()]))
        auroc_epistemic = roc_auc_score(labels, np.concatenate([in_epistemic.numpy(), ood_epistemic.numpy()]))
        
        return {
            "error": error,
            "nll": nll,
            "ece": ece.item() if isinstance(ece, torch.Tensor) else ece,
            "ood_auroc": auroc_total,
            "ood_auroc_total": auroc_total,
            "ood_auroc_aleatoric": auroc_aleatoric,
            "ood_auroc_epistemic": auroc_epistemic,
            "mean_in_total_entropy": in_total.mean().item(),
            "mean_in_aleatoric_entropy": in_aleatoric.mean().item(),
            "mean_in_epistemic_entropy": in_epistemic.mean().item(),
            "mean_ood_total_entropy": ood_total.mean().item(),
            "mean_ood_aleatoric_entropy": ood_aleatoric.mean().item(),
            "mean_ood_epistemic_entropy": ood_epistemic.mean().item(),
        }
    
    def _compute_mfvi_metrics(self, model) -> Dict:
        """Compute full metrics for MFVI."""
        from sklearn.metrics import roc_auc_score
        
        model.eval()
        num_samples = 10
        
        all_probs, all_labels = [], []
        in_total, in_aleatoric, in_epistemic = [], [], []
        ood_total, ood_aleatoric, ood_epistemic = [], [], []
        
        with torch.no_grad():
            # In-distribution
            for batch_x, batch_y in self.data_manager.test_loader:
                batch_x = batch_x.to(self.device)
                
                sample_probs = torch.stack([
                    F.softmax(model(batch_x, num_samples=1), dim=-1) 
                    for _ in range(num_samples)
                ])
                mean_probs = sample_probs.mean(dim=0)
                
                total_ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
                indiv_ent = -torch.sum(sample_probs * torch.log(sample_probs + 1e-10), dim=-1)
                aleat_ent = indiv_ent.mean(dim=0)
                epist_ent = total_ent - aleat_ent
                
                all_probs.append(mean_probs.cpu())
                all_labels.append(batch_y)
                in_total.append(total_ent.cpu())
                in_aleatoric.append(aleat_ent.cpu())
                in_epistemic.append(epist_ent.cpu())
            
            # OOD
            for batch_x, _ in self.data_manager.ood_loader:
                batch_x = batch_x.to(self.device)
                
                sample_probs = torch.stack([
                    F.softmax(model(batch_x, num_samples=1), dim=-1) 
                    for _ in range(num_samples)
                ])
                mean_probs = sample_probs.mean(dim=0)
                
                total_ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
                indiv_ent = -torch.sum(sample_probs * torch.log(sample_probs + 1e-10), dim=-1)
                aleat_ent = indiv_ent.mean(dim=0)
                epist_ent = total_ent - aleat_ent
                
                ood_total.append(total_ent.cpu())
                ood_aleatoric.append(aleat_ent.cpu())
                ood_epistemic.append(epist_ent.cpu())
        
        # Concatenate and compute metrics (same as SVGD)
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        in_total = torch.cat(in_total)
        in_aleatoric = torch.cat(in_aleatoric)
        in_epistemic = torch.cat(in_epistemic)
        ood_total = torch.cat(ood_total)
        ood_aleatoric = torch.cat(ood_aleatoric)
        ood_epistemic = torch.cat(ood_epistemic)
        
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
        
        labels = np.concatenate([np.zeros(len(in_total)), np.ones(len(ood_total))])
        auroc_total = roc_auc_score(labels, np.concatenate([in_total.numpy(), ood_total.numpy()]))
        auroc_aleatoric = roc_auc_score(labels, np.concatenate([in_aleatoric.numpy(), ood_aleatoric.numpy()]))
        auroc_epistemic = roc_auc_score(labels, np.concatenate([in_epistemic.numpy(), ood_epistemic.numpy()]))
        
        return {
            "error": error,
            "nll": nll,
            "ece": ece.item() if isinstance(ece, torch.Tensor) else ece,
            "ood_auroc": auroc_total,
            "ood_auroc_total": auroc_total,
            "ood_auroc_aleatoric": auroc_aleatoric,
            "ood_auroc_epistemic": auroc_epistemic,
            "mean_in_total_entropy": in_total.mean().item(),
            "mean_in_aleatoric_entropy": in_aleatoric.mean().item(),
            "mean_in_epistemic_entropy": in_epistemic.mean().item(),
            "mean_ood_total_entropy": ood_total.mean().item(),
            "mean_ood_aleatoric_entropy": ood_aleatoric.mean().item(),
            "mean_ood_epistemic_entropy": ood_epistemic.mean().item(),
        }
    
    def _save_results(self, run_dir, variant, temperature, replicate, metrics, history, training_time):
        """Save all results."""
        # Save results.json
        results = {
            "variant": variant.name,
            "method": variant.method,
            "prior_type": variant.prior_type,
            "prior_std": variant.prior_std,
            "n_particles": variant.n_particles if variant.method == 'svgd' else None,
            "temperature": temperature,
            "replicate": replicate,
            "metrics": metrics,
            "training_time": training_time,
            "training_history": history,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save variant config
        with open(os.path.join(run_dir, "variant_config.json"), 'w') as f:
            json.dump(variant.to_dict(), f, indent=2)
        
        # Save model weights
        if hasattr(self, '_temp_particles'):
            torch.save(self._temp_particles, os.path.join(run_dir, "particles.pt"))
            del self._temp_particles
        
        if hasattr(self, '_temp_mfvi_layer'):
            torch.save(self._temp_mfvi_layer, os.path.join(run_dir, "mfvi_layer.pt"))
            del self._temp_mfvi_layer
    
    def train_all(
        self,
        variants: List[ModelVariant] = None,
        temperatures: List[float] = None,
        num_replicates: int = None
    ):
        """Train all pending configurations."""
        
        pending = self.get_pending_runs(variants, temperatures, num_replicates)
        
        print("\n" + "=" * 70)
        print("VARIANT-BASED TRAINING")
        print("=" * 70)
        print(f"Pending runs: {len(pending)}")
        print("=" * 70)
        
        if len(pending) == 0:
            print("✓ All configurations completed!")
            return
        
        for i, (variant, temp, rep) in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}]")
            try:
                self.train_variant(variant, temp, rep)
            except Exception as e:
                print(f"ERROR: {e}")
                continue


# =============================================================================
# Status Printing
# =============================================================================

def print_status(save_dir: str, variants: List[ModelVariant] = None):
    """Print status of all configurations."""
    
    if variants is None:
        variants = MODEL_VARIANTS
    
    print("\n" + "=" * 100)
    print("TRAINING STATUS")
    print("=" * 100)
    
    total = 0
    completed = 0
    
    for variant in variants:
        variant_completed = 0
        variant_total = 0
        
        for temp in TEMPERATURES:
            for rep in range(1, NUM_REPLICATES + 1):
                run_name = variant.get_run_name(temp, rep)
                run_dir = os.path.join(save_dir, run_name)
                variant_total += 1
                total += 1
                
                if os.path.exists(os.path.join(run_dir, "COMPLETED")):
                    variant_completed += 1
                    completed += 1
        
        status = "✓" if variant_completed == variant_total else f"{variant_completed}/{variant_total}"
        print(f"  {variant.name:<30} {status}")
    
    print("-" * 100)
    print(f"Total: {completed}/{total} completed ({100*completed/total:.1f}%)")
    print("=" * 100)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variant-based BNN training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--list", action="store_true", help="List all model variants")
    parser.add_argument("--status", action="store_true", help="Show training status")
    parser.add_argument("--step", type=int, choices=[2, 3], help="Training step")
    parser.add_argument("--save-dir", type=str, default="./results", help="Results directory")
    
    parser.add_argument("--variants", nargs="+", help="Specific variants to run")
    parser.add_argument("--variant", type=str, help="Single variant name")
    parser.add_argument("--temperature", type=float, help="Single temperature")
    parser.add_argument("--replicate", type=int, help="Single replicate")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    
    parser.add_argument("--force", action="store_true", help="Force re-run")
    
    args = parser.parse_args()
    
    # List variants
    if args.list:
        print_variants()
        return
    
    # Status
    if args.status:
        print_status(args.save_dir)
        return
    
    # Training
    if args.step is None:
        parser.print_help()
        return
    
    if args.step == 2:
        trainer = VariantTrainer(args.save_dir)
        
        # Single configuration
        if args.variant and args.temperature and args.replicate:
            variant = get_variant(args.variant)
            if variant is None:
                print(f"ERROR: Unknown variant: {args.variant}")
                print(f"Available: {get_variant_names()}")
                return
            
            run_name = variant.get_run_name(args.temperature, args.replicate)
            if trainer._is_completed(run_name) and not args.force:
                print(f"Already completed: {run_name}")
                print("Use --force to re-run")
                return
            
            trainer.train_variant(variant, args.temperature, args.replicate, args.epochs)
        
        # Specific variants
        elif args.variants:
            variants = [get_variant(name) for name in args.variants]
            variants = [v for v in variants if v is not None]
            trainer.train_all(variants=variants)
        
        # All variants
        else:
            trainer.train_all()
    
    elif args.step == 3:
        print("Step 3 (joint training) not yet implemented for variants")
        print("Use pipeline_resnet.py for Step 3")


if __name__ == "__main__":
    main()
