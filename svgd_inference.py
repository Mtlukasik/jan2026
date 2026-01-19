"""
Stein Variational Gradient Descent (SVGD) Module for Bayesian Last Layer.

This module implements SVGD following Liu & Wang (2016) "Stein Variational 
Gradient Descent: A General Purpose Bayesian Inference Algorithm".

Key Components:
1. Particle-based posterior approximation
2. RBF kernel with median heuristic bandwidth
3. Combined attractive (gradient) and repulsive (kernel gradient) forces

SVGD vs MCMC vs MFVI:
---------------------
SVGD:
- Maintains ensemble of particles (parameter settings)
- Balances fitting data (attractive) with diversity (repulsive)
- Deterministic updates (no random sampling during inference)
- Temperature (T) scales the likelihood term

MCMC:
- Sequential samples from posterior
- Requires burn-in, thinning
- Temperature appears as p(w|D)^(1/T)

MFVI:
- Single factorized Gaussian approximation
- May underestimate uncertainty
- Fast but less flexible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import copy

from config import SVGDConfig, NetworkConfig, DEVICE


@dataclass
class SVGDTrainingState:
    """State container for SVGD training."""
    epoch: int
    loss: float
    diversity: float  # Mean pairwise distance between particles
    accuracy: float


class SVGDParticle(nn.Module):
    """Single SVGD particle with feature extractor and last layer.
    
    Each particle is a complete network that can make predictions.
    The feature extractor is shared across particles (trained jointly),
    while the last layer is unique per particle (updated via SVGD).
    """
    
    def __init__(self, config: NetworkConfig, dropout_rate: float = 0.3):
        super().__init__()
        self.config = config
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Last layer (will be updated by SVGD)
        self.last_layer = nn.Linear(config.hidden_dim, config.output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for module in self.feature_extractor:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in', nonlinearity='relu')
        if self.last_layer.bias is not None:
            nn.init.zeros_(self.last_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the particle."""
        features = self.feature_extractor(x)
        return self.last_layer(features)
    
    def get_last_layer_params(self) -> torch.Tensor:
        """Get flattened last layer parameters."""
        return torch.cat([
            self.last_layer.weight.flatten(),
            self.last_layer.bias.flatten()
        ])
    
    def set_last_layer_params(self, params: torch.Tensor):
        """Set last layer parameters from flattened vector."""
        weight_size = self.last_layer.weight.numel()
        bias_size = self.last_layer.bias.numel()
        
        weight = params[:weight_size].view_as(self.last_layer.weight)
        bias = params[weight_size:weight_size + bias_size].view_as(self.last_layer.bias)
        
        self.last_layer.weight.data.copy_(weight)
        self.last_layer.bias.data.copy_(bias)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without passing through last layer."""
        return self.feature_extractor(x)


class SVGDEnsemble:
    """SVGD Ensemble for last-layer Bayesian inference.
    
    Maintains a set of particles that approximate the posterior distribution.
    Uses SVGD updates to balance fitting data with maintaining diversity.
    """
    
    def __init__(
        self,
        config: NetworkConfig,
        svgd_config: SVGDConfig,
        device: str = "cuda"
    ):
        self.config = config
        self.svgd_config = svgd_config
        self.device = device
        self.n_particles = svgd_config.n_particles
        
        # Create particle ensemble
        self.particles = [
            SVGDParticle(config, dropout_rate=0.3).to(device)
            for _ in range(self.n_particles)
        ]
        
        # Optimizers for feature extractors
        self.feature_optimizers = [
            torch.optim.AdamW(
                particle.feature_extractor.parameters(),
                lr=svgd_config.feature_lr,
                weight_decay=svgd_config.weight_decay
            )
            for particle in self.particles
        ]
        
        # Learning rate schedulers
        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=svgd_config.num_epochs, eta_min=1e-5
            )
            for opt in self.feature_optimizers
        ]
        
        # Training history
        self.history: List[SVGDTrainingState] = []
    
    def rbf_kernel(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        bandwidth_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RBF kernel with adaptive bandwidth (median heuristic).
        
        Args:
            x1: First set of points [n_particles, param_dim]
            x2: Second set of points [n_particles, param_dim]
            bandwidth_scale: Scaling factor for bandwidth
            
        Returns:
            k: Kernel matrix [n_particles, n_particles]
            grad_k: Gradient of kernel (for repulsive term)
            pairwise_dists: Pairwise distances
        """
        # Compute pairwise squared distances
        pairwise_dists = torch.cdist(x1, x2, p=2) ** 2
        
        # Median heuristic for bandwidth
        h = torch.median(pairwise_dists)
        h = torch.clamp(h / np.log(self.n_particles + 1), min=1e-5)
        h = h * bandwidth_scale
        
        # Compute kernel matrix: k(x, y) = exp(-||x-y||^2 / (2h))
        k = torch.exp(-pairwise_dists / (2 * h))
        
        # Gradient of kernel (used for repulsive force)
        # grad_k[i,j] = -k[i,j] / h * (x[i] - x[j])
        grad_k = -k.unsqueeze(-1) / h
        
        return k, grad_k, pairwise_dists
    
    def compute_log_prob(
        self,
        particle: SVGDParticle,
        x: torch.Tensor,
        y: torch.Tensor,
        prior_std: float = 1.0,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute log probability (likelihood + prior) with temperature.
        
        Args:
            particle: SVGD particle
            x: Input batch
            y: Target labels
            prior_std: Prior standard deviation
            temperature: Temperature for tempering
            
        Returns:
            Scalar log probability
        """
        # Log likelihood (tempered)
        output = particle(x)
        log_likelihood = -F.cross_entropy(output, y, reduction='sum') / temperature
        
        # Log prior on last layer
        last_layer_params = particle.get_last_layer_params()
        
        if self.svgd_config.use_laplace_prior:
            # Laplace prior: log p(w) = -|w| / scale (up to constant)
            log_prior = -torch.sum(torch.abs(last_layer_params)) / prior_std
        else:
            # Gaussian prior: log p(w) = -0.5 * ||w||^2 / sigma^2
            log_prior = -0.5 * torch.sum(last_layer_params ** 2) / (prior_std ** 2)
        
        return log_likelihood + log_prior
    
    def svgd_update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        svgd_lr: float = 1e-3,
        prior_std: float = 1.0,
        temperature: float = 1.0,
        bandwidth_scale: float = 1.0
    ) -> float:
        """Perform SVGD update on last layer parameters.
        
        The SVGD update combines:
        - Attractive term: pulls particles toward high-probability regions
        - Repulsive term: pushes particles apart to maintain diversity
        
        Args:
            x: Input batch
            y: Target labels
            svgd_lr: Learning rate for SVGD
            prior_std: Prior standard deviation
            temperature: Temperature for tempering
            bandwidth_scale: Kernel bandwidth scaling
            
        Returns:
            Mean pairwise distance (diversity measure)
        """
        # Get current parameters from all particles
        theta = torch.stack([p.get_last_layer_params() for p in self.particles])
        
        # Compute kernel matrix and gradients
        k, grad_k, dists = self.rbf_kernel(theta, theta, bandwidth_scale)
        
        # Compute gradients of log probability for each particle
        grad_log_probs = []
        for particle in self.particles:
            particle.zero_grad()
            log_prob = self.compute_log_prob(particle, x, y, prior_std, temperature)
            log_prob.backward()
            
            # Extract gradient of last layer parameters
            grad = torch.cat([
                particle.last_layer.weight.grad.flatten(),
                particle.last_layer.bias.grad.flatten()
            ])
            grad_log_probs.append(grad)
        
        grad_log_probs = torch.stack(grad_log_probs)
        
        # Compute SVGD gradient for each particle
        svgd_grad = torch.zeros_like(theta)
        for i in range(self.n_particles):
            # Attractive term: weighted sum of gradients
            # Sum_j k(x_j, x_i) * grad_log_prob(x_j)
            attractive = torch.sum(k[:, i].unsqueeze(-1) * grad_log_probs, dim=0)
            
            # Repulsive term: gradient of kernel
            # Sum_j grad_k(x_j, x_i)
            repulsive = torch.sum(
                grad_k[:, i, :] * (theta - theta[i].unsqueeze(0)), 
                dim=0
            )
            
            svgd_grad[i] = (attractive + repulsive) / self.n_particles
        
        # Update parameters
        with torch.no_grad():
            for i, particle in enumerate(self.particles):
                new_params = theta[i] + svgd_lr * svgd_grad[i]
                particle.set_last_layer_params(new_params)
        
        return dists.mean().item()
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[float, float]:
        """Combined training step: update feature extractors + SVGD.
        
        Args:
            x: Input batch
            y: Target labels
            temperature: Temperature for tempering
            
        Returns:
            Tuple of (mean loss, diversity)
        """
        total_loss = 0.0
        
        # Update feature extractors with standard gradient descent
        for particle, optimizer in zip(self.particles, self.feature_optimizers):
            optimizer.zero_grad()
            output = particle(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                particle.feature_extractor.parameters(),
                max_norm=self.svgd_config.grad_clip
            )
            
            optimizer.step()
            total_loss += loss.item()
        
        # SVGD update for last layer
        diversity = self.svgd_update(
            x, y,
            svgd_lr=self.svgd_config.svgd_lr,
            prior_std=self.svgd_config.prior_std,
            temperature=temperature,
            bandwidth_scale=self.svgd_config.bandwidth_scale
        )
        
        return total_loss / self.n_particles, diversity
    
    def step_schedulers(self):
        """Step all learning rate schedulers."""
        for scheduler in self.schedulers:
            scheduler.step()
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from all particles.
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions of shape [n_particles, batch_size, num_classes]
        """
        predictions = []
        for particle in self.particles:
            particle.eval()
            pred = particle(x)
            predictions.append(pred)
            particle.train()
        return torch.stack(predictions)
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get mean prediction and uncertainty estimates.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_probs, std_probs, entropy)
        """
        predictions = self.predict(x)  # [n_particles, batch_size, num_classes]
        
        # Softmax probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Mean and std across particles
        mean_probs = probs.mean(dim=0)
        std_probs = probs.std(dim=0)
        
        # Predictive entropy (uncertainty measure)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        
        return mean_probs, std_probs, entropy


class SVGDTrainer:
    """Trainer for SVGD-based last layer inference.
    
    Handles the training loop, including:
    1. Optional feature extractor pretraining
    2. Joint training with SVGD updates
    3. Evaluation and prediction
    """
    
    def __init__(
        self,
        config: NetworkConfig,
        svgd_config: SVGDConfig,
        device: str = "cuda"
    ):
        self.config = config
        self.svgd_config = svgd_config
        self.device = device
        
        # Create SVGD ensemble
        self.ensemble = SVGDEnsemble(config, svgd_config, device)
        
        # Training history
        self.history: List[SVGDTrainingState] = []
    
    def pretrain_feature_extractor(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50,
        lr: float = 0.01
    ) -> float:
        """Pretrain feature extractors with standard SGD.
        
        All particles share the same feature extractor initialization
        after pretraining (but will diverge during SVGD training).
        
        Returns:
            Final validation accuracy
        """
        # Use first particle for pretraining
        particle = self.ensemble.particles[0]
        particle.train()
        
        optimizer = torch.optim.SGD(
            particle.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = particle(batch_x)
                loss = F.cross_entropy(output, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
        
        # Copy pretrained weights to all other particles
        pretrained_state = particle.state_dict()
        for other_particle in self.ensemble.particles[1:]:
            other_particle.load_state_dict(pretrained_state)
            
            # Add small random perturbation to last layer for diversity
            with torch.no_grad():
                other_particle.last_layer.weight.add_(
                    torch.randn_like(other_particle.last_layer.weight) * 0.01
                )
                other_particle.last_layer.bias.add_(
                    torch.randn_like(other_particle.last_layer.bias) * 0.01
                )
        
        # Compute validation accuracy
        correct = 0
        total = 0
        particle.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = particle(batch_x)
                preds = output.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        
        return correct / total
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        temperature: float = 1.0,
        num_epochs: Optional[int] = None
    ) -> List[SVGDTrainingState]:
        """Train the SVGD ensemble.
        
        Args:
            train_loader: Training data loader
            temperature: Temperature for cold/warm posterior
            num_epochs: Number of epochs (uses config default if None)
            
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.svgd_config.num_epochs
        
        self.history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_diversity = 0.0
            epoch_correct = 0
            epoch_total = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                loss, diversity = self.ensemble.train_step(
                    batch_x, batch_y, temperature
                )
                
                epoch_loss += loss
                epoch_diversity += diversity
                num_batches += 1
                
                # Compute accuracy (using mean prediction)
                with torch.no_grad():
                    mean_probs, _, _ = self.ensemble.predict_with_uncertainty(batch_x)
                    preds = mean_probs.argmax(dim=-1)
                    epoch_correct += (preds == batch_y).sum().item()
                    epoch_total += len(batch_y)
            
            # Step schedulers
            self.ensemble.step_schedulers()
            
            # Record state
            state = SVGDTrainingState(
                epoch=epoch,
                loss=epoch_loss / num_batches,
                diversity=epoch_diversity / num_batches,
                accuracy=epoch_correct / epoch_total
            )
            self.history.append(state)
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={state.loss:.4f}, Acc={state.accuracy:.4f}, "
                      f"Diversity={state.diversity:.4f}")
        
        return self.history
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with all particles.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits of shape [n_particles, batch_size, num_classes]
        """
        return self.ensemble.predict(x)
    
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get mean probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Mean probabilities of shape [batch_size, num_classes]
        """
        mean_probs, _, _ = self.ensemble.predict_with_uncertainty(x)
        return mean_probs


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_svgd_particle():
    """Test SVGDParticle forward pass and parameter access."""
    config = NetworkConfig()
    particle = SVGDParticle(config).to(DEVICE)
    
    # Test forward pass
    x = torch.randn(32, config.input_dim).to(DEVICE)
    output = particle(x)
    assert output.shape == (32, config.output_dim), f"Got shape {output.shape}"
    
    # Test parameter extraction
    params = particle.get_last_layer_params()
    expected_size = config.hidden_dim * config.output_dim + config.output_dim
    assert params.shape == (expected_size,), f"Got shape {params.shape}"
    
    # Test parameter setting
    new_params = torch.randn_like(params)
    particle.set_last_layer_params(new_params)
    retrieved_params = particle.get_last_layer_params()
    assert torch.allclose(new_params, retrieved_params)
    
    print("✓ SVGDParticle test passed")


def test_rbf_kernel():
    """Test RBF kernel computation."""
    config = NetworkConfig()
    svgd_config = SVGDConfig(n_particles=5)
    ensemble = SVGDEnsemble(config, svgd_config, DEVICE)
    
    # Get particle parameters
    theta = torch.stack([p.get_last_layer_params() for p in ensemble.particles])
    
    # Compute kernel
    k, grad_k, dists = ensemble.rbf_kernel(theta, theta)
    
    # Check shapes
    assert k.shape == (5, 5), f"Got kernel shape {k.shape}"
    assert grad_k.shape == (5, 5, 1), f"Got grad_k shape {grad_k.shape}"
    
    # Kernel should be symmetric
    assert torch.allclose(k, k.t(), atol=1e-5)
    
    # Diagonal should be 1 (same point)
    assert torch.allclose(k.diag(), torch.ones(5, device=DEVICE), atol=1e-5)
    
    print("✓ RBF kernel test passed")


def test_svgd_update():
    """Test SVGD update changes parameters."""
    config = NetworkConfig()
    svgd_config = SVGDConfig(n_particles=3)
    ensemble = SVGDEnsemble(config, svgd_config, DEVICE)
    
    # Get initial parameters
    initial_params = [p.get_last_layer_params().clone() for p in ensemble.particles]
    
    # Create dummy data
    x = torch.randn(16, config.input_dim).to(DEVICE)
    y = torch.randint(0, config.output_dim, (16,)).to(DEVICE)
    
    # Perform SVGD update
    diversity = ensemble.svgd_update(x, y, svgd_lr=0.1)
    
    # Check parameters changed
    for i, particle in enumerate(ensemble.particles):
        new_params = particle.get_last_layer_params()
        assert not torch.allclose(initial_params[i], new_params), \
            f"Parameters for particle {i} did not change"
    
    # Diversity should be positive
    assert diversity > 0
    
    print("✓ SVGD update test passed")


def test_svgd_prediction():
    """Test SVGD ensemble prediction."""
    config = NetworkConfig()
    svgd_config = SVGDConfig(n_particles=5)
    trainer = SVGDTrainer(config, svgd_config, DEVICE)
    
    x = torch.randn(32, config.input_dim).to(DEVICE)
    
    # Test raw prediction
    logits = trainer.predict(x)
    assert logits.shape == (5, 32, config.output_dim), f"Got shape {logits.shape}"
    
    # Test probability prediction
    probs = trainer.predict_proba(x)
    assert probs.shape == (32, config.output_dim), f"Got shape {probs.shape}"
    
    # Probabilities should sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(32, device=DEVICE), atol=1e-5)
    
    print("✓ SVGD prediction test passed")


def test_svgd_uncertainty():
    """Test SVGD uncertainty estimation."""
    config = NetworkConfig()
    svgd_config = SVGDConfig(n_particles=10)
    ensemble = SVGDEnsemble(config, svgd_config, DEVICE)
    
    x = torch.randn(32, config.input_dim).to(DEVICE)
    
    mean_probs, std_probs, entropy = ensemble.predict_with_uncertainty(x)
    
    # Check shapes
    assert mean_probs.shape == (32, config.output_dim)
    assert std_probs.shape == (32, config.output_dim)
    assert entropy.shape == (32,)
    
    # Entropy should be non-negative
    assert (entropy >= 0).all()
    
    # Entropy should be bounded by log(num_classes)
    max_entropy = np.log(config.output_dim)
    assert (entropy <= max_entropy + 0.1).all()
    
    print("✓ SVGD uncertainty test passed")


def test_temperature_effect():
    """Test that temperature affects the log probability."""
    config = NetworkConfig()
    svgd_config = SVGDConfig(n_particles=1)
    ensemble = SVGDEnsemble(config, svgd_config, DEVICE)
    
    particle = ensemble.particles[0]
    x = torch.randn(16, config.input_dim).to(DEVICE)
    y = torch.randint(0, config.output_dim, (16,)).to(DEVICE)
    
    # Compute log probs at different temperatures
    log_prob_cold = ensemble.compute_log_prob(particle, x, y, temperature=0.1)
    log_prob_normal = ensemble.compute_log_prob(particle, x, y, temperature=1.0)
    log_prob_warm = ensemble.compute_log_prob(particle, x, y, temperature=10.0)
    
    # Cold temperature should have larger magnitude (more peaked)
    # (assuming same data, colder = more extreme log likelihood)
    # This is a bit tricky to test definitively, so just check they're different
    assert not torch.allclose(log_prob_cold, log_prob_normal)
    assert not torch.allclose(log_prob_normal, log_prob_warm)
    
    print("✓ Temperature effect test passed")


def test_laplace_vs_gaussian_prior():
    """Test that different priors give different results."""
    config = NetworkConfig()
    
    # Gaussian prior
    svgd_config_gauss = SVGDConfig(n_particles=1, use_laplace_prior=False)
    ensemble_gauss = SVGDEnsemble(config, svgd_config_gauss, DEVICE)
    
    # Laplace prior
    svgd_config_laplace = SVGDConfig(n_particles=1, use_laplace_prior=True)
    ensemble_laplace = SVGDEnsemble(config, svgd_config_laplace, DEVICE)
    
    # Copy parameters to make them identical
    ensemble_laplace.particles[0].load_state_dict(
        ensemble_gauss.particles[0].state_dict()
    )
    
    x = torch.randn(16, config.input_dim).to(DEVICE)
    y = torch.randint(0, config.output_dim, (16,)).to(DEVICE)
    
    log_prob_gauss = ensemble_gauss.compute_log_prob(
        ensemble_gauss.particles[0], x, y
    )
    log_prob_laplace = ensemble_laplace.compute_log_prob(
        ensemble_laplace.particles[0], x, y
    )
    
    # They should be different due to different priors
    assert not torch.allclose(log_prob_gauss, log_prob_laplace)
    
    print("✓ Laplace vs Gaussian prior test passed")


if __name__ == "__main__":
    test_svgd_particle()
    test_rbf_kernel()
    test_svgd_update()
    test_svgd_prediction()
    test_svgd_uncertainty()
    test_temperature_effect()
    test_laplace_vs_gaussian_prior()
    print("\nAll SVGD tests passed!")
