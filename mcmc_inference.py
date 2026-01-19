"""
MCMC Inference Module for Bayesian Last Layer.

This module implements Stochastic Gradient MCMC (SG-MCMC) for last-layer
Bayesian inference, following the approach in the paper:

Key Components:
1. Stochastic Gradient Langevin Dynamics (SGLD)
2. Gradient-Guided Monte Carlo (GG-MC) following Garriga-Alonso & Fortuin (2021)
3. Cyclical learning rate schedule following Zhang et al. (2019)
4. Temperature scaling for cold/warm posteriors

Temperature Explanation:
-----------------------
The tempered posterior is: p(w|x,y)^(1/T)
- T = 1: Standard Bayesian posterior
- T < 1: "Cold" posterior - sharper, more concentrated
- T > 1: "Warm" posterior - flatter, more spread out

In the loss function, temperature appears as:
L(w) = (1/T) * [log p(y|w,x) + log p(w)] + const
     = (1/T) * NLL + (1/T) * prior_loss

The paper studies whether cold posteriors (T < 1) give better predictions,
and investigates the causes of this "cold posterior effect".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import copy

from config import MCMCConfig, NetworkConfig, DEVICE


@dataclass
class MCMCSample:
    """Container for a single MCMC sample."""
    weight: torch.Tensor
    bias: torch.Tensor
    log_posterior: float


class GaussianPrior:
    """Isotropic Gaussian prior over weights.
    
    p(w) = N(0, sigma^2 * I)
    log p(w) = -0.5 * ||w||^2 / sigma^2 + const
    """
    
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.sigma_sq = sigma ** 2
    
    def log_prob(self, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Compute log prior probability."""
        log_p = -0.5 * (weight.pow(2).sum() + bias.pow(2).sum()) / self.sigma_sq
        return log_p
    
    def grad_log_prob(self, weight: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient of log prior."""
        grad_w = -weight / self.sigma_sq
        grad_b = -bias / self.sigma_sq
        return grad_w, grad_b


class LaplacePrior:
    """Laplace (heavy-tailed) prior over weights.
    
    p(w) = Laplace(0, b) = (1/2b) * exp(-|w|/b)
    log p(w) = -|w|/b + const
    
    The paper shows heavy-tailed priors can reduce the cold posterior effect
    in FCNNs.
    """
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
    
    def log_prob(self, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Compute log prior probability."""
        log_p = -(weight.abs().sum() + bias.abs().sum()) / self.scale
        return log_p
    
    def grad_log_prob(self, weight: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient of log prior (subgradient at 0)."""
        grad_w = -torch.sign(weight) / self.scale
        grad_b = -torch.sign(bias) / self.scale
        return grad_w, grad_b


class SGLD:
    """Stochastic Gradient Langevin Dynamics.
    
    SGLD update rule:
    θ_{t+1} = θ_t + (ε/2) * ∇log p(θ|D) + N(0, ε)
    
    where ε is the learning rate and the noise term adds the required
    stochasticity for sampling from the posterior.
    
    With temperature T, the update becomes:
    θ_{t+1} = θ_t + (ε/2) * (1/T) * ∇log p(θ|D) + N(0, ε)
    
    Note: In practice with minibatches, we scale the likelihood term
    by (N/n) where N is dataset size and n is batch size.
    """
    
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        add_noise: bool = True
    ):
        """
        Args:
            params: Parameters to optimize
            lr: Learning rate (step size ε)
            temperature: Posterior temperature
            weight_decay: L2 regularization (corresponds to Gaussian prior)
            add_noise: Whether to add Langevin noise
        """
        self.params = list(params)
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.add_noise = add_noise
    
    def step(self, loss_grad_fn: Optional[Callable] = None):
        """Perform one SGLD update step.
        
        Args:
            loss_grad_fn: Optional function to compute loss and set gradients.
                         If None, assumes gradients are already computed.
        """
        if loss_grad_fn is not None:
            loss_grad_fn()
        
        with torch.no_grad():
            for param in self.params:
                if param.grad is None:
                    continue
                
                # Scale gradient by temperature
                grad = param.grad / self.temperature
                
                # Add weight decay (prior gradient)
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # Gradient descent step
                param.add_(grad, alpha=-self.lr / 2)
                
                # Add Langevin noise
                if self.add_noise:
                    noise = torch.randn_like(param) * np.sqrt(self.lr)
                    param.add_(noise)
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class CyclicalSGMCMC:
    """Cyclical Stochastic Gradient MCMC.
    
    Implements the cyclical learning rate schedule from Zhang et al. (2019):
    - Learning rate follows a cosine annealing schedule within each cycle
    - Samples are collected from the low-lr phase of each cycle
    - This allows exploring multiple modes of the posterior
    
    The paper uses this with Gradient-Guided MC (similar to SGLD but with
    momentum for faster mixing).
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MCMCConfig,
        prior: GaussianPrior,
        temperature: float = 1.0,
        device: str = "cuda"
    ):
        """
        Args:
            model: The neural network (only last layer will be sampled)
            config: MCMC configuration
            prior: Prior distribution over weights
            temperature: Posterior temperature
            device: Device for computation
        """
        self.model = model
        self.config = config
        self.prior = prior
        self.temperature = temperature
        self.device = device
        
        # Get last layer parameters
        self.last_layer = model.last_layer
        self.params = [self.last_layer.weight, self.last_layer.bias]
        
        # Storage for samples
        self.samples: List[MCMCSample] = []
        
        # Diagnostics
        self.kinetic_temps = []
        self.config_temps = []
    
    def _get_cosine_lr(self, epoch_in_cycle: int) -> float:
        """Compute learning rate using cosine annealing."""
        # Cosine schedule: starts high, decays to 0
        progress = epoch_in_cycle / self.config.epochs_per_cycle
        lr = self.config.learning_rate_init * 0.5 * (1 + np.cos(np.pi * progress))
        return lr
    
    def _should_add_noise(self, epoch_in_cycle: int) -> bool:
        """Determine if Langevin noise should be added.
        
        Following the paper, only add noise in the last portion of each cycle.
        """
        start_noise_epoch = self.config.epochs_per_cycle - self.config.noise_epochs
        return epoch_in_cycle >= start_noise_epoch
    
    def _should_collect_sample(self, epoch_in_cycle: int) -> bool:
        """Determine if current parameters should be collected as sample.
        
        Collect samples from the last few epochs of each cycle.
        """
        start_sample_epoch = self.config.epochs_per_cycle - self.config.samples_per_cycle
        return epoch_in_cycle >= start_sample_epoch
    
    def run_chain(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_data: int
    ) -> List[MCMCSample]:
        """Run the MCMC chain and collect samples.
        
        Args:
            train_loader: DataLoader for training data
            num_data: Total number of training samples (for scaling)
            
        Returns:
            List of posterior samples
        """
        self.model.train()
        self.model.freeze_feature_extractor()  # Only sample last layer
        
        for cycle in range(self.config.num_cycles):
            for epoch_in_cycle in range(self.config.epochs_per_cycle):
                # Get learning rate and noise settings
                lr = self._get_cosine_lr(epoch_in_cycle)
                add_noise = self._should_add_noise(epoch_in_cycle)
                
                # Create SGLD optimizer for this epoch
                sgld = SGLD(
                    self.params,
                    lr=lr,
                    temperature=self.temperature,
                    add_noise=add_noise
                )
                
                # Train for one epoch
                epoch_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    sgld.zero_grad()
                    
                    # Forward pass (only through last layer since features are frozen)
                    with torch.no_grad():
                        features = self.model.get_features(batch_x)
                    
                    logits = self.last_layer(features, use_samples=False)
                    
                    # Compute tempered loss
                    batch_size = batch_x.shape[0]
                    scale = num_data / batch_size  # Scale to full dataset
                    
                    # Negative log-likelihood
                    nll = F.cross_entropy(logits, batch_y, reduction='sum')
                    
                    # Prior (scaled appropriately)
                    log_prior = self.prior.log_prob(
                        self.last_layer.weight, 
                        self.last_layer.bias
                    )
                    
                    # Tempered loss: (1/T) * (scaled_nll - log_prior)
                    loss = (scale * nll - log_prior) / self.temperature
                    
                    loss.backward()
                    sgld.step()
                    
                    epoch_loss += nll.item()
                
                # Collect sample if in collection phase
                if self._should_collect_sample(epoch_in_cycle):
                    sample = MCMCSample(
                        weight=self.last_layer.weight.detach().clone(),
                        bias=self.last_layer.bias.detach().clone(),
                        log_posterior=-epoch_loss / num_data  # Approximate
                    )
                    self.samples.append(sample)
        
        # Remove burn-in samples
        if len(self.samples) > self.config.burn_in_samples:
            self.samples = self.samples[self.config.burn_in_samples:]
        
        # Transfer samples to the model's MCMC layer
        self._transfer_samples_to_model()
        
        return self.samples
    
    def _transfer_samples_to_model(self):
        """Transfer collected samples to the model's BayesianLastLayerMCMC."""
        self.last_layer.clear_samples()
        for sample in self.samples:
            self.last_layer.weight_samples.append(sample.weight)
            self.last_layer.bias_samples.append(sample.bias)
    
    def compute_kinetic_temperature(self, momentum: torch.Tensor) -> float:
        """Compute kinetic temperature diagnostic.
        
        T_kinetic = (1/d) * m^T * M^{-1} * m
        
        Should equal the target temperature if sampling correctly.
        """
        d = momentum.numel()
        # Assuming unit mass matrix
        return (momentum.pow(2).sum() / d).item()
    
    def compute_configurational_temperature(
        self, 
        params: List[torch.Tensor],
        grads: List[torch.Tensor]
    ) -> float:
        """Compute configurational temperature diagnostic.
        
        T_config = (1/d) * θ^T * ∇H(θ)
        
        Should equal the target temperature if sampling correctly.
        """
        total = 0.0
        d = 0
        for p, g in zip(params, grads):
            total += (p * g).sum().item()
            d += p.numel()
        return total / d


class MCMCTrainer:
    """High-level trainer for MCMC inference.
    
    This class handles:
    1. Feature extractor pretraining with SGD
    2. MCMC sampling for the last layer
    3. Multiple chains for diagnostics
    """
    
    def __init__(
        self,
        model: nn.Module,
        mcmc_config: MCMCConfig,
        prior_sigma: float = 1.0,
        device: str = "cuda"
    ):
        self.model = model
        self.mcmc_config = mcmc_config
        self.prior = GaussianPrior(prior_sigma)
        self.device = device
    
    def pretrain_feature_extractor(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        lr: float = 0.01
    ) -> float:
        """Pretrain the feature extractor using standard SGD.
        
        This gives a good initialization for MCMC sampling.
        
        Returns:
            Final validation accuracy
        """
        self.model.train()
        self.model.unfreeze_feature_extractor()
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
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
                logits = self.model(batch_x, use_samples=False)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
        
        # Compute validation accuracy
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x, use_samples=False)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        
        return correct / total
    
    def run_mcmc(
        self,
        train_loader: torch.utils.data.DataLoader,
        temperature: float = 1.0
    ) -> List[MCMCSample]:
        """Run MCMC sampling on the last layer.
        
        Args:
            train_loader: Training data loader
            temperature: Posterior temperature
            
        Returns:
            List of posterior samples
        """
        num_data = len(train_loader.dataset)
        
        sampler = CyclicalSGMCMC(
            self.model,
            self.mcmc_config,
            self.prior,
            temperature=temperature,
            device=self.device
        )
        
        samples = sampler.run_chain(train_loader, num_data)
        return samples


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_gaussian_prior():
    """Test Gaussian prior."""
    prior = GaussianPrior(sigma=1.0)
    
    weight = torch.zeros(10, 5)
    bias = torch.zeros(10)
    
    # Zero weights should have log prob = 0 (up to constant)
    log_p = prior.log_prob(weight, bias)
    assert log_p == 0.0
    
    # Non-zero weights should have negative log prob
    weight = torch.randn(10, 5)
    log_p = prior.log_prob(weight, bias)
    assert log_p < 0
    
    print("✓ Gaussian prior test passed")


def test_laplace_prior():
    """Test Laplace prior."""
    prior = LaplacePrior(scale=1.0)
    
    weight = torch.zeros(10, 5)
    bias = torch.zeros(10)
    
    log_p = prior.log_prob(weight, bias)
    assert log_p == 0.0
    
    weight = torch.randn(10, 5)
    log_p = prior.log_prob(weight, bias)
    assert log_p < 0
    
    print("✓ Laplace prior test passed")


def test_sgld():
    """Test SGLD optimizer."""
    # Create simple parameters
    params = [torch.randn(5, 3, requires_grad=True)]
    
    sgld = SGLD(params, lr=0.01, temperature=1.0, add_noise=True)
    
    # Compute dummy loss and gradients
    loss = params[0].sum()
    loss.backward()
    
    # Store initial value
    initial = params[0].clone()
    
    # Take step
    sgld.step()
    
    # Parameters should have changed
    assert not torch.allclose(params[0], initial)
    
    print("✓ SGLD test passed")


def test_cyclical_lr():
    """Test cyclical learning rate schedule."""
    config = MCMCConfig(
        num_cycles=2,
        epochs_per_cycle=10,
        learning_rate_init=0.01
    )
    
    # Create dummy model and sampler
    from networks import FullNetwork, NetworkConfig
    model = FullNetwork(NetworkConfig(), "mcmc").to(DEVICE)
    prior = GaussianPrior()
    
    sampler = CyclicalSGMCMC(model, config, prior, device=DEVICE)
    
    # Test LR schedule
    lrs = [sampler._get_cosine_lr(e) for e in range(config.epochs_per_cycle)]
    
    # LR should start high and decay
    assert lrs[0] > lrs[-1]
    assert lrs[0] == config.learning_rate_init
    assert lrs[-1] < 0.001  # Should be near 0 at end
    
    print("✓ Cyclical LR test passed")


def test_noise_schedule():
    """Test noise addition schedule."""
    config = MCMCConfig(
        epochs_per_cycle=45,
        noise_epochs=15,
        samples_per_cycle=5
    )
    
    from networks import FullNetwork, NetworkConfig
    model = FullNetwork(NetworkConfig(), "mcmc").to(DEVICE)
    prior = GaussianPrior()
    
    sampler = CyclicalSGMCMC(model, config, prior, device=DEVICE)
    
    # No noise in first 30 epochs
    for e in range(30):
        assert not sampler._should_add_noise(e), f"Epoch {e} should not have noise"
    
    # Noise in last 15 epochs
    for e in range(30, 45):
        assert sampler._should_add_noise(e), f"Epoch {e} should have noise"
    
    print("✓ Noise schedule test passed")


def test_sample_collection():
    """Test sample collection schedule."""
    config = MCMCConfig(
        epochs_per_cycle=45,
        samples_per_cycle=5
    )
    
    from networks import FullNetwork, NetworkConfig
    model = FullNetwork(NetworkConfig(), "mcmc").to(DEVICE)
    prior = GaussianPrior()
    
    sampler = CyclicalSGMCMC(model, config, prior, device=DEVICE)
    
    # No collection in first 40 epochs
    for e in range(40):
        assert not sampler._should_collect_sample(e)
    
    # Collection in last 5 epochs
    for e in range(40, 45):
        assert sampler._should_collect_sample(e)
    
    print("✓ Sample collection test passed")


if __name__ == "__main__":
    test_gaussian_prior()
    test_laplace_prior()
    test_sgld()
    test_cyclical_lr()
    test_noise_schedule()
    test_sample_collection()
    print("\nAll MCMC tests passed!")
