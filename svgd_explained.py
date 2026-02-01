"""
SVGD (Stein Variational Gradient Descent) for Bayesian Last Layer.

This implements the SVGD algorithm from:
    "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm"
    Liu & Wang, NeurIPS 2016

SVGD maintains an ensemble of "particles" (parameter vectors) that collectively
approximate the posterior distribution. Unlike MCMC which produces sequential samples,
SVGD updates all particles simultaneously using a deterministic gradient flow.

Key Insight: SVGD balances two forces:
    1. ATTRACTIVE: Push particles toward high posterior density (gradient of log prob)
    2. REPULSIVE: Push particles apart to maintain diversity (kernel gradient)

The update rule for particle θᵢ is:
    θᵢ ← θᵢ + ε · φ(θᵢ)
    
where:
    φ(θᵢ) = (1/n) Σⱼ [ k(θⱼ, θᵢ) · ∇θⱼ log p(θⱼ|D) + ∇θⱼ k(θⱼ, θᵢ) ]
                      \_____________attractive____________/   \___repulsive___/

The kernel k(θ, θ') measures similarity between particles. We use RBF kernel:
    k(θ, θ') = exp(-||θ - θ'||² / (2h))
    
with bandwidth h chosen by the median heuristic.

Training Phases:
    Phase 1: Train Bayesian last layer only (feature extractor frozen)
    Phase 2 (optional): Joint training (unfreeze feature extractor, lower LR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from resnet import ResNetFeatureExtractor


@dataclass
class SVGDConfig:
    """Configuration for SVGD training."""
    n_particles: int = 20           # Number of particles in ensemble
    svgd_lr: float = 1e-3           # Learning rate for SVGD updates
    feature_lr: float = 1e-4        # Learning rate for feature extractor (Phase 2)
    prior_std: float = 1.0          # Prior standard deviation
    use_laplace_prior: bool = True  # Laplace prior (sparsity) vs Gaussian
    bandwidth_scale: float = 1.0    # Multiplier for kernel bandwidth
    init_std: float = 0.01          # Std for particle initialization perturbation


class SVGDEnsemble:
    """
    SVGD Ensemble for Bayesian Last Layer Inference.
    
    Architecture:
        ┌─────────────────────────────────────────────────────┐
        │  Input Image (3, 32, 32)                            │
        │       │                                             │
        │       ▼                                             │
        │  ┌─────────────────────────────────┐                │
        │  │  Shared Feature Extractor       │ ← Frozen in    │
        │  │  (ResNet-18 backbone)           │   Phase 1      │
        │  │  Output: 512-dim features       │                │
        │  └─────────────────────────────────┘                │
        │       │                                             │
        │       ▼                                             │
        │  ┌─────────────────────────────────┐                │
        │  │  SVGD Particles (n=20)          │                │
        │  │  Each: Linear(512 → 10)         │ ← Trained via  │
        │  │                                 │   SVGD updates │
        │  │  θ₁, θ₂, ..., θ₂₀              │                │
        │  └─────────────────────────────────┘                │
        │       │                                             │
        │       ▼                                             │
        │  Ensemble Predictions                               │
        │  p(y|x) ≈ (1/n) Σᵢ softmax(fθᵢ(x))                 │
        └─────────────────────────────────────────────────────┘
    
    The particles are initialized from the pretrained deterministic last layer
    with small random perturbations for diversity.
    """
    
    def __init__(
        self,
        config: SVGDConfig,
        pretrained_state_dict: Dict[str, torch.Tensor],
        num_classes: int = 10,
        device: str = "cuda"
    ):
        """
        Initialize SVGD ensemble from pretrained weights.
        
        Args:
            config: SVGD hyperparameters
            pretrained_state_dict: State dict from trained ResNet model
                                   Must contain 'feature_extractor.*' and 'last_layer.*' keys
            num_classes: Number of output classes
            device: Device to use
        """
        self.config = config
        self.device = device
        self.n_particles = config.n_particles
        self.num_classes = num_classes
        
        # ===================================================================
        # Step 1: Create and load SHARED feature extractor (will be frozen)
        # ===================================================================
        self.feature_extractor = ResNetFeatureExtractor().to(device)
        
        # Extract feature extractor weights from pretrained model
        fe_state = {
            k.replace("feature_extractor.", ""): v 
            for k, v in pretrained_state_dict.items() 
            if k.startswith("feature_extractor.")
        }
        self.feature_extractor.load_state_dict(fe_state)
        
        # Freeze feature extractor (Phase 1)
        self._freeze_feature_extractor()
        
        print(f"Loaded pretrained feature extractor ({sum(p.numel() for p in self.feature_extractor.parameters()):,} params, frozen)")
        
        # ===================================================================
        # Step 2: Extract pretrained last layer weights
        # ===================================================================
        pretrained_weight = pretrained_state_dict["last_layer.fc.weight"]  # Shape: (10, 512)
        pretrained_bias = pretrained_state_dict["last_layer.fc.bias"]      # Shape: (10,)
        
        print(f"Pretrained last layer: weight {pretrained_weight.shape}, bias {pretrained_bias.shape}")
        
        # ===================================================================
        # Step 3: Create particles initialized from pretrained last layer
        # ===================================================================
        # Each particle is a copy of the pretrained last layer + small perturbation
        self.particles: List[nn.Linear] = []
        
        for i in range(self.n_particles):
            # Create linear layer
            particle = nn.Linear(512, num_classes).to(device)
            
            # Initialize with pretrained weights
            particle.weight.data.copy_(pretrained_weight)
            particle.bias.data.copy_(pretrained_bias)
            
            # Add small random perturbation for diversity (except first particle)
            # First particle keeps exact pretrained weights as reference
            if i > 0:
                with torch.no_grad():
                    particle.weight.data.add_(
                        torch.randn_like(particle.weight) * config.init_std
                    )
                    particle.bias.data.add_(
                        torch.randn_like(particle.bias) * config.init_std
                    )
            
            self.particles.append(particle)
        
        print(f"Created {self.n_particles} particles from pretrained last layer")
        print(f"  Particle 0: exact copy of pretrained")
        print(f"  Particles 1-{self.n_particles-1}: perturbed with std={config.init_std}")
        
        # Track training phase
        self.phase = 1  # 1 = frozen features, 2 = joint training
    
    def _freeze_feature_extractor(self):
        """Freeze feature extractor and set to eval mode (freezes BatchNorm stats)."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
    
    def _unfreeze_feature_extractor(self):
        """Unfreeze feature extractor for joint training (Phase 2)."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.feature_extractor.train()
    
    def start_joint_training(self, feature_lr: float = None):
        """
        Switch to Phase 2: Joint training of features + last layer.
        
        In Phase 2:
            - Feature extractor is unfrozen
            - Feature extractor trained with smaller LR
            - SVGD continues on last layer
        
        Args:
            feature_lr: Learning rate for feature extractor (default from config)
        """
        if feature_lr is None:
            feature_lr = self.config.feature_lr
        
        self._unfreeze_feature_extractor()
        self.phase = 2
        
        # Create optimizer for feature extractor
        self.feature_optimizer = optim.SGD(
            self.feature_extractor.parameters(),
            lr=feature_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        print(f"Switched to Phase 2: Joint training")
        print(f"  Feature extractor unfrozen, LR={feature_lr}")
    
    # =========================================================================
    # SVGD Core Algorithm
    # =========================================================================
    
    def get_particle_params(self, particle: nn.Linear) -> torch.Tensor:
        """
        Flatten particle parameters into a single vector.
        
        For SVGD, we need to treat all parameters as a single θ vector.
        
        Args:
            particle: Linear layer
            
        Returns:
            Flattened parameter vector of shape (512*10 + 10,) = (5130,)
        """
        return torch.cat([
            particle.weight.flatten(),  # (10, 512) → (5120,)
            particle.bias.flatten()     # (10,) → (10,)
        ])
    
    def set_particle_params(self, particle: nn.Linear, params: torch.Tensor):
        """
        Set particle parameters from flattened vector.
        
        Args:
            particle: Linear layer to update
            params: Flattened parameter vector
        """
        weight_numel = particle.weight.numel()  # 10 * 512 = 5120
        
        particle.weight.data.copy_(
            params[:weight_numel].view_as(particle.weight)
        )
        particle.bias.data.copy_(
            params[weight_numel:].view_as(particle.bias)
        )
    
    def compute_rbf_kernel(
        self, 
        theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RBF kernel matrix with median heuristic bandwidth.
        
        The RBF (Radial Basis Function) kernel measures similarity:
            k(θᵢ, θⱼ) = exp(-||θᵢ - θⱼ||² / (2h))
        
        Bandwidth h is chosen using the median heuristic:
            h = median(||θᵢ - θⱼ||²) / log(n+1)
        
        This adaptive bandwidth ensures the kernel is neither too peaked
        (particles don't interact) nor too flat (no local structure).
        
        Args:
            theta: Particle parameters, shape (n_particles, param_dim)
            
        Returns:
            kernel_matrix: Shape (n_particles, n_particles)
            bandwidth: Scalar bandwidth value
        """
        # Compute pairwise squared distances
        # ||θᵢ - θⱼ||² for all pairs
        pairwise_dists_sq = torch.cdist(theta, theta, p=2) ** 2
        
        # Median heuristic for bandwidth
        # Take median of non-zero distances
        h = torch.median(pairwise_dists_sq)
        h = h / np.log(self.n_particles + 1)
        h = torch.clamp(h, min=1e-5)  # Prevent division by zero
        h = h * self.config.bandwidth_scale  # Allow manual scaling
        
        # Compute RBF kernel
        kernel_matrix = torch.exp(-pairwise_dists_sq / (2 * h))
        
        return kernel_matrix, h
    
    def compute_log_prior(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute log prior probability of parameters.
        
        We support two priors:
        
        1. Laplace prior (default): p(θ) ∝ exp(-|θ|/σ)
           - Encourages sparsity (L1-like regularization)
           - log p(θ) = -Σ|θᵢ|/σ + const
        
        2. Gaussian prior: p(θ) ∝ exp(-θ²/(2σ²))
           - Standard L2 regularization
           - log p(θ) = -Σθᵢ²/(2σ²) + const
        
        Args:
            params: Parameter vector
            
        Returns:
            Log prior (scalar)
        """
        if self.config.use_laplace_prior:
            # Laplace: log p(θ) = -|θ|/σ (up to constant)
            return -torch.sum(torch.abs(params)) / self.config.prior_std
        else:
            # Gaussian: log p(θ) = -θ²/(2σ²) (up to constant)
            return -0.5 * torch.sum(params ** 2) / (self.config.prior_std ** 2)
    
    def compute_log_likelihood(
        self, 
        particle: nn.Linear, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute tempered log likelihood.
        
        The likelihood for classification is:
            p(y|x,θ) = softmax(f_θ(x))_y
        
        Log likelihood (negative cross-entropy):
            log p(y|x,θ) = log softmax(f_θ(x))_y = -CE(f_θ(x), y)
        
        Temperature scaling (cold/warm posterior):
            log p(y|x,θ)^(1/T) = -CE(f_θ(x), y) / T
        
        T < 1: "Cold" posterior - sharper, more confident
        T > 1: "Warm" posterior - flatter, more uncertain
        T = 1: Standard Bayesian posterior
        
        Args:
            particle: Linear layer
            features: Input features, shape (batch, 512)
            labels: True labels, shape (batch,)
            temperature: Posterior temperature
            
        Returns:
            Log likelihood (scalar, summed over batch)
        """
        logits = particle(features)  # (batch, 10)
        
        # Negative cross-entropy = log likelihood
        # reduction='sum' because we want total log prob, not average
        log_likelihood = -F.cross_entropy(logits, labels, reduction='sum')
        
        # Temperature scaling
        return log_likelihood / temperature
    
    def compute_log_posterior(
        self,
        particle: nn.Linear,
        features: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute (unnormalized) log posterior.
        
        By Bayes' rule:
            log p(θ|D) ∝ log p(D|θ) + log p(θ)
                       = log_likelihood + log_prior
        
        Args:
            particle: Linear layer
            features: Input features
            labels: True labels  
            temperature: Posterior temperature
            
        Returns:
            Log posterior (scalar)
        """
        params = self.get_particle_params(particle)
        
        log_likelihood = self.compute_log_likelihood(
            particle, features, labels, temperature
        )
        log_prior = self.compute_log_prior(params)
        
        return log_likelihood + log_prior
    
    def svgd_update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[float, float]:
        """
        Perform one SVGD update on all particles.
        
        The SVGD update rule:
            θᵢ ← θᵢ + ε · φ(θᵢ)
        
        where φ(θᵢ) is the "optimal perturbation direction":
            φ(θᵢ) = (1/n) Σⱼ [ k(θⱼ,θᵢ) · ∇θⱼ log p(θⱼ|D) + ∇θⱼ k(θⱼ,θᵢ) ]
        
        Breaking this down:
        
        Term 1: k(θⱼ,θᵢ) · ∇θⱼ log p(θⱼ|D)
            - Weighted average of posterior gradients
            - Particles contribute their gradients, weighted by kernel similarity
            - ATTRACTIVE: Pulls θᵢ toward high posterior regions
        
        Term 2: ∇θⱼ k(θⱼ,θᵢ)
            - Gradient of kernel w.r.t. θⱼ
            - For RBF: ∇θⱼ k = k(θⱼ,θᵢ) · (θᵢ - θⱼ) / h
            - REPULSIVE: Pushes particles apart to maintain diversity
        
        The balance between these terms is what makes SVGD special:
        - If particles collapse together, repulsive term increases
        - If particles spread too far, attractive term dominates
        
        Args:
            features: Batch of features, shape (batch, 512)
            labels: Batch of labels, shape (batch,)
            temperature: Posterior temperature
            
        Returns:
            Tuple of (average_loss, particle_diversity)
        """
        # =====================================================================
        # Step 1: Collect current particle parameters
        # =====================================================================
        # theta shape: (n_particles, param_dim) = (20, 5130)
        theta = torch.stack([
            self.get_particle_params(p) for p in self.particles
        ])
        
        # =====================================================================
        # Step 2: Compute RBF kernel matrix
        # =====================================================================
        # kernel shape: (n_particles, n_particles)
        # kernel[i,j] = k(θᵢ, θⱼ) = similarity between particles i and j
        kernel, bandwidth = self.compute_rbf_kernel(theta)
        
        # =====================================================================
        # Step 3: Compute gradient of log posterior for each particle
        # =====================================================================
        grad_log_posterior = []
        total_loss = 0.0
        
        for particle in self.particles:
            # Zero gradients
            particle.zero_grad()
            
            # Compute log posterior
            log_post = self.compute_log_posterior(
                particle, features, labels, temperature
            )
            
            # Backward to get gradients
            log_post.backward()
            
            # Collect gradient (∇θ log p(θ|D))
            grad = torch.cat([
                particle.weight.grad.flatten(),
                particle.bias.grad.flatten()
            ])
            grad_log_posterior.append(grad)
            
            # Track loss for monitoring
            with torch.no_grad():
                loss = F.cross_entropy(particle(features), labels)
                total_loss += loss.item()
        
        # Stack gradients: shape (n_particles, param_dim)
        grad_log_posterior = torch.stack(grad_log_posterior)
        
        # =====================================================================
        # Step 4: Compute SVGD gradient for each particle
        # =====================================================================
        svgd_gradient = torch.zeros_like(theta)
        
        for i in range(self.n_particles):
            # -------------------------------------------------------------
            # Term 1: Attractive force (weighted gradients)
            # Σⱼ k(θⱼ, θᵢ) · ∇θⱼ log p(θⱼ|D)
            # -------------------------------------------------------------
            # kernel[:, i] shape: (n_particles,) - similarity of all particles to particle i
            # grad_log_posterior shape: (n_particles, param_dim)
            # Result: weighted sum of gradients
            attractive = torch.sum(
                kernel[:, i].unsqueeze(-1) * grad_log_posterior,
                dim=0
            )
            
            # -------------------------------------------------------------
            # Term 2: Repulsive force (kernel gradient)
            # Σⱼ ∇θⱼ k(θⱼ, θᵢ)
            # 
            # For RBF kernel:
            # ∇θⱼ k(θⱼ,θᵢ) = k(θⱼ,θᵢ) · (θᵢ - θⱼ) / h
            # -------------------------------------------------------------
            # (theta[i] - theta) pushes away from all other particles
            repulsive = torch.sum(
                kernel[:, i].unsqueeze(-1) * (theta[i] - theta) / bandwidth,
                dim=0
            )
            
            # -------------------------------------------------------------
            # Combine: φ(θᵢ) = (1/n) * (attractive + repulsive)
            # -------------------------------------------------------------
            svgd_gradient[i] = (attractive + repulsive) / self.n_particles
        
        # =====================================================================
        # Step 5: Update particles
        # θᵢ ← θᵢ + ε · φ(θᵢ)
        # =====================================================================
        with torch.no_grad():
            for i, particle in enumerate(self.particles):
                new_params = theta[i] + self.config.svgd_lr * svgd_gradient[i]
                self.set_particle_params(particle, new_params)
        
        # =====================================================================
        # Compute diversity metric (average pairwise distance)
        # =====================================================================
        diversity = torch.cdist(theta, theta).mean().item()
        
        return total_loss / self.n_particles, diversity
    
    # =========================================================================
    # Training Step (combines feature extraction + SVGD)
    # =========================================================================
    
    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[float, float]:
        """
        One training step: extract features and update particles.
        
        Phase 1 (frozen features):
            - Features extracted with no gradient
            - Only SVGD update on last layer
        
        Phase 2 (joint training):
            - Features extracted with gradient
            - Feature extractor updated via backprop from particles
            - SVGD update on last layer
        
        Args:
            images: Batch of images, shape (batch, 3, 32, 32)
            labels: Batch of labels, shape (batch,)
            temperature: Posterior temperature
            
        Returns:
            Tuple of (loss, diversity)
        """
        if self.phase == 1:
            # Phase 1: Frozen features
            with torch.no_grad():
                features = self.feature_extractor(images)
        else:
            # Phase 2: Joint training - compute features with gradient
            features = self.feature_extractor(images)
        
        # SVGD update on last layer particles
        loss, diversity = self.svgd_update(features, labels, temperature)
        
        # Phase 2: Also update feature extractor
        if self.phase == 2:
            # Compute loss for feature extractor update
            # Use mean prediction from all particles
            self.feature_optimizer.zero_grad()
            
            features = self.feature_extractor(images)  # Recompute with grad
            
            # Average loss across particles
            total_loss = 0
            for particle in self.particles:
                logits = particle(features)
                total_loss += F.cross_entropy(logits, labels)
            total_loss = total_loss / self.n_particles
            
            total_loss.backward()
            self.feature_optimizer.step()
        
        return loss, diversity
    
    # =========================================================================
    # Prediction Methods
    # =========================================================================
    
    @torch.no_grad()
    def predict_all(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from all particles.
        
        Args:
            images: Input images, shape (batch, 3, 32, 32)
            
        Returns:
            Logits from all particles, shape (n_particles, batch, num_classes)
        """
        self.feature_extractor.eval()
        features = self.feature_extractor(images)
        
        predictions = []
        for particle in self.particles:
            logits = particle(features)
            predictions.append(logits)
        
        return torch.stack(predictions)
    
    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get mean probability predictions (Bayesian model averaging).
        
        p(y|x) ≈ (1/n) Σᵢ softmax(f_θᵢ(x))
        
        This is the key Bayesian prediction: average over posterior samples.
        
        Args:
            images: Input images
            
        Returns:
            Mean probabilities, shape (batch, num_classes)
        """
        logits = self.predict_all(images)  # (n_particles, batch, classes)
        probs = F.softmax(logits, dim=-1)   # Convert to probabilities
        return probs.mean(dim=0)            # Average over particles
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self, 
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty estimates.
        
        Uncertainty is measured by:
        1. Predictive entropy: H[p(y|x)] = -Σ p(y|x) log p(y|x)
           - High entropy = model is uncertain about prediction
        
        2. Standard deviation across particles
           - High std = particles disagree (epistemic uncertainty)
        
        Args:
            images: Input images
            
        Returns:
            Tuple of:
                - mean_probs: Mean probabilities, shape (batch, classes)
                - std_probs: Std across particles, shape (batch, classes)
                - entropy: Predictive entropy, shape (batch,)
        """
        logits = self.predict_all(images)
        probs = F.softmax(logits, dim=-1)
        
        mean_probs = probs.mean(dim=0)
        std_probs = probs.std(dim=0)
        
        # Predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        
        return mean_probs, std_probs, entropy


# =============================================================================
# Training Function
# =============================================================================

def train_svgd_two_phase(
    ensemble: SVGDEnsemble,
    train_loader,
    test_loader,
    temperature: float = 1.0,
    phase1_epochs: int = 100,
    phase2_epochs: int = 50,
    device: str = "cuda"
) -> Dict:
    """
    Two-phase SVGD training.
    
    Phase 1: Train Bayesian last layer only
        - Feature extractor frozen
        - SVGD updates on last layer particles
        - Faster convergence, stable features
    
    Phase 2: Joint training (optional)
        - Unfreeze feature extractor with small LR
        - Continue SVGD on last layer
        - Fine-tune features for uncertainty
    
    Args:
        ensemble: SVGDEnsemble instance
        train_loader: Training data
        test_loader: Test data for evaluation
        temperature: Posterior temperature
        phase1_epochs: Epochs for Phase 1
        phase2_epochs: Epochs for Phase 2 (0 to skip)
        device: Device to use
        
    Returns:
        Training history dict
    """
    history = {
        "phase": [], "epoch": [], "train_loss": [], 
        "diversity": [], "val_accuracy": []
    }
    
    # =========================================================================
    # Phase 1: Frozen feature extractor
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Training Bayesian Last Layer (frozen features)")
    print(f"{'='*60}")
    
    for epoch in range(phase1_epochs):
        epoch_loss = 0.0
        epoch_diversity = 0.0
        n_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            loss, diversity = ensemble.train_step(images, labels, temperature)
            
            epoch_loss += loss
            epoch_diversity += diversity
            n_batches += 1
        
        # Record history
        history["phase"].append(1)
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss / n_batches)
        history["diversity"].append(epoch_diversity / n_batches)
        
        # Evaluate periodically
        if (epoch + 1) % 20 == 0 or epoch == phase1_epochs - 1:
            val_acc = evaluate_ensemble(ensemble, test_loader, device)
            history["val_accuracy"].append(val_acc)
            print(f"Phase 1, Epoch {epoch+1}/{phase1_epochs}: "
                  f"Loss={epoch_loss/n_batches:.4f}, "
                  f"Diversity={epoch_diversity/n_batches:.4f}, "
                  f"Val Acc={val_acc:.4f}")
    
    # =========================================================================
    # Phase 2: Joint training (optional)
    # =========================================================================
    if phase2_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Phase 2: Joint Training (unfrozen features)")
        print(f"{'='*60}")
        
        ensemble.start_joint_training()
        
        for epoch in range(phase2_epochs):
            epoch_loss = 0.0
            epoch_diversity = 0.0
            n_batches = 0
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                loss, diversity = ensemble.train_step(images, labels, temperature)
                
                epoch_loss += loss
                epoch_diversity += diversity
                n_batches += 1
            
            history["phase"].append(2)
            history["epoch"].append(phase1_epochs + epoch)
            history["train_loss"].append(epoch_loss / n_batches)
            history["diversity"].append(epoch_diversity / n_batches)
            
            if (epoch + 1) % 10 == 0 or epoch == phase2_epochs - 1:
                val_acc = evaluate_ensemble(ensemble, test_loader, device)
                history["val_accuracy"].append(val_acc)
                print(f"Phase 2, Epoch {epoch+1}/{phase2_epochs}: "
                      f"Loss={epoch_loss/n_batches:.4f}, "
                      f"Val Acc={val_acc:.4f}")
    
    return history


def evaluate_ensemble(ensemble: SVGDEnsemble, test_loader, device: str) -> float:
    """Evaluate ensemble accuracy."""
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        probs = ensemble.predict_proba(images)
        preds = probs.argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += len(labels)
    
    return correct / total


# =============================================================================
# Unit Tests
# =============================================================================

def test_svgd_ensemble():
    """Test SVGD ensemble creation and basic operations."""
    print("Testing SVGD Ensemble...")
    
    # Create dummy pretrained weights
    from resnet import ResNetForBayesianLastLayer
    
    dummy_model = ResNetForBayesianLastLayer(num_classes=10, last_layer_type="deterministic")
    pretrained_state = dummy_model.state_dict()
    
    # Create ensemble
    config = SVGDConfig(n_particles=5)
    ensemble = SVGDEnsemble(config, pretrained_state, device="cpu")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    probs = ensemble.predict_proba(x)
    
    assert probs.shape == (4, 10), f"Expected (4, 10), got {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4)), "Probabilities don't sum to 1"
    
    print("✓ SVGD Ensemble test passed")


def test_svgd_update():
    """Test SVGD update step."""
    print("Testing SVGD update...")
    
    from resnet import ResNetForBayesianLastLayer
    
    dummy_model = ResNetForBayesianLastLayer(num_classes=10, last_layer_type="deterministic")
    pretrained_state = dummy_model.state_dict()
    
    config = SVGDConfig(n_particles=5, svgd_lr=0.01)
    ensemble = SVGDEnsemble(config, pretrained_state, device="cpu")
    
    # Get initial params
    initial_params = [ensemble.get_particle_params(p).clone() for p in ensemble.particles]
    
    # Run update
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    
    loss, diversity = ensemble.train_step(x, y, temperature=1.0)
    
    # Check params changed
    for i, particle in enumerate(ensemble.particles):
        new_params = ensemble.get_particle_params(particle)
        assert not torch.allclose(initial_params[i], new_params), f"Particle {i} didn't update"
    
    print(f"✓ SVGD update test passed (loss={loss:.4f}, diversity={diversity:.4f})")


if __name__ == "__main__":
    test_svgd_ensemble()
    test_svgd_update()
    print("\n✓ All SVGD tests passed!")
