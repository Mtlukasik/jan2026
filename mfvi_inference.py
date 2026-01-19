"""
Mean-Field Variational Inference Module for Bayesian Last Layer.

This module implements MFVI following Blundell et al. (2015) "Weight Uncertainty
in Neural Networks" and the paper's approach in Appendix A.11.

Key Components:
1. ELBO optimization with reparameterization trick
2. KL annealing for stable training
3. Temperature scaling (λ parameter) for cold posteriors

MFVI vs MCMC:
-------------
MFVI:
- Approximates posterior with factorized Gaussian: q(w) = Π_i N(w_i | μ_i, σ_i^2)
- Fast training via gradient descent on ELBO
- May underestimate uncertainty (mode-seeking)
- Temperature (λ) appears in KL term: ELBO = E_q[log p(y|x,w)] - λ*KL(q||p)

MCMC:
- Samples directly from (tempered) posterior
- More accurate but slower
- Temperature (T) appears in posterior: p(w|D)^(1/T)

Note: The relationship between MFVI λ and MCMC T is not straightforward
except at λ = T = 1 (standard Bayesian posterior).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import math

from config import MFVIConfig, NetworkConfig, DEVICE


@dataclass
class MFVITrainingState:
    """State container for MFVI training."""
    epoch: int
    loss: float
    kl_loss: float
    nll_loss: float
    kl_weight: float  # Current KL annealing weight


class MFVITrainer:
    """Trainer for Mean-Field Variational Inference on last layer.
    
    Implements the ELBO:
    L(q; λ) = E_q[log p(y|x,w)] - λ * KL(q(w) || p(w))
    
    where:
    - E_q[log p(y|x,w)] is approximated by sampling w ~ q(w)
    - KL is computed analytically for Gaussian distributions
    - λ controls the "temperature" (λ < 1 gives cold posterior)
    
    Training Details:
    - Uses Adam optimizer
    - KL annealing for first 100 epochs (gradually increases λ)
    - Learning rate decay after 500 epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MFVIConfig,
        device: str = "cuda"
    ):
        """
        Args:
            model: Neural network with BayesianLastLayerMFVI as last_layer
            config: MFVI configuration
            device: Device for computation
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Verify last layer is MFVI
        assert hasattr(model.last_layer, 'kl_divergence'), \
            "Last layer must be BayesianLastLayerMFVI"
        
        # Training history
        self.history: List[MFVITrainingState] = []
    
    def _get_kl_weight(self, epoch: int, temperature: float) -> float:
        """Compute KL weight with annealing and temperature.
        
        KL annealing: gradually increase KL weight from 0 to temperature
        over the first kl_annealing_epochs epochs.
        
        Args:
            epoch: Current epoch
            temperature: Target temperature (λ)
            
        Returns:
            KL weight for this epoch
        """
        if epoch < self.config.kl_annealing_epochs:
            # Linear annealing
            progress = (epoch + 1) / self.config.kl_annealing_epochs
            return temperature * progress
        else:
            return temperature
    
    def _get_learning_rate(self, epoch: int) -> float:
        """Get learning rate with decay schedule."""
        if epoch < 500:
            return self.config.learning_rate_init
        else:
            return self.config.learning_rate_decay
    
    def pretrain_feature_extractor(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        lr: float = 0.01
    ) -> float:
        """Pretrain feature extractor with standard SGD.
        
        Returns:
            Final validation accuracy
        """
        self.model.train()
        self.model.unfreeze_feature_extractor()
        
        # Use deterministic forward pass for pretraining
        optimizer = torch.optim.SGD(
            self.model.feature_extractor.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Also train a temporary deterministic last layer
        temp_last_layer = nn.Linear(
            self.model.config.hidden_dim,
            self.model.config.output_dim
        ).to(self.device)
        
        optimizer_last = torch.optim.SGD(
            temp_last_layer.parameters(),
            lr=lr,
            momentum=0.9
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        for epoch in range(num_epochs):
            self.model.feature_extractor.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                optimizer_last.zero_grad()
                
                features = self.model.get_features(batch_x)
                logits = temp_last_layer(features)
                loss = F.cross_entropy(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                optimizer_last.step()
            
            scheduler.step()
        
        # Initialize MFVI last layer means from pretrained layer
        with torch.no_grad():
            self.model.last_layer.weight_mu.copy_(temp_last_layer.weight)
            self.model.last_layer.bias_mu.copy_(temp_last_layer.bias)
        
        # Compute validation accuracy
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                features = self.model.get_features(batch_x)
                logits = temp_last_layer(features)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        
        return correct / total
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        temperature: float = 1.0,
        num_epochs: Optional[int] = None
    ) -> List[MFVITrainingState]:
        """Train the MFVI last layer.
        
        Args:
            train_loader: Training data loader
            temperature: Temperature (λ) for cold/warm posterior
            num_epochs: Number of epochs (uses config default if None)
            
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.model.train()
        self.model.freeze_feature_extractor()  # Only train last layer
        
        num_data = len(train_loader.dataset)
        
        # Set up optimizer for last layer parameters only
        optimizer = torch.optim.Adam(
            self.model.last_layer.parameters(),
            lr=self.config.learning_rate_init
        )
        
        self.history = []
        
        for epoch in range(num_epochs):
            # Update learning rate if needed
            current_lr = self._get_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Get KL weight with annealing
            kl_weight = self._get_kl_weight(epoch, temperature)
            
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_nll = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_size = batch_x.shape[0]
                
                optimizer.zero_grad()
                
                # Get features (frozen)
                with torch.no_grad():
                    features = self.model.get_features(batch_x)
                
                # Forward pass with sampling
                logits = self.model.last_layer(
                    features, 
                    num_samples=self.config.num_train_samples
                )
                
                # Compute NLL
                if logits.ndim == 3:
                    # Average over samples
                    logits_mean = logits.mean(dim=0)
                else:
                    logits_mean = logits
                
                nll = F.cross_entropy(logits_mean, batch_y)
                
                # Compute KL divergence, scaled by batch/dataset ratio
                kl = self.model.last_layer.kl_divergence()
                kl_scaled = kl * (batch_size / num_data)
                
                # ELBO loss (negative ELBO to minimize)
                # L = NLL + λ * KL
                loss = nll + kl_weight * kl_scaled
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_kl += kl_scaled.item()
                epoch_nll += nll.item()
                num_batches += 1
            
            # Record state
            state = MFVITrainingState(
                epoch=epoch,
                loss=epoch_loss / num_batches,
                kl_loss=epoch_kl / num_batches,
                nll_loss=epoch_nll / num_batches,
                kl_weight=kl_weight
            )
            self.history.append(state)
            
            # Print progress occasionally
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={state.loss:.4f}, NLL={state.nll_loss:.4f}, "
                      f"KL={state.kl_loss:.4f}, λ={kl_weight:.4f}")
        
        return self.history
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Make predictions with uncertainty.
        
        Args:
            x: Input tensor
            num_samples: Number of samples (uses config default if None)
            
        Returns:
            Logits of shape (num_samples, batch_size, num_classes)
        """
        if num_samples is None:
            num_samples = self.config.num_test_samples
        
        self.model.eval()
        
        features = self.model.get_features(x)
        logits = self.model.last_layer(features, num_samples=num_samples)
        
        return logits


class MFVIEnsemble:
    """Ensemble of MFVI models for improved uncertainty.
    
    Running multiple MFVI models with different initializations
    can help capture more of the posterior uncertainty.
    """
    
    def __init__(
        self,
        model_class: type,
        network_config: NetworkConfig,
        mfvi_config: MFVIConfig,
        num_models: int = 5,
        device: str = "cuda"
    ):
        self.model_class = model_class
        self.network_config = network_config
        self.mfvi_config = mfvi_config
        self.num_models = num_models
        self.device = device
        
        self.models: List[nn.Module] = []
        self.trainers: List[MFVITrainer] = []
    
    def initialize_models(self):
        """Create and initialize all ensemble members."""
        for i in range(self.num_models):
            # Create model with different random seed
            torch.manual_seed(42 + i)
            model = self.model_class(
                self.network_config, 
                last_layer_type="mfvi"
            ).to(self.device)
            
            trainer = MFVITrainer(model, self.mfvi_config, self.device)
            
            self.models.append(model)
            self.trainers.append(trainer)
    
    def train_ensemble(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        temperature: float = 1.0,
        pretrain_epochs: int = 100
    ):
        """Train all ensemble members."""
        for i, (model, trainer) in enumerate(zip(self.models, self.trainers)):
            print(f"\n--- Training model {i+1}/{self.num_models} ---")
            
            # Pretrain feature extractor
            val_acc = trainer.pretrain_feature_extractor(
                train_loader, val_loader, pretrain_epochs
            )
            print(f"Pretrain validation accuracy: {val_acc:.4f}")
            
            # Train MFVI last layer
            trainer.train(train_loader, temperature)
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        num_samples_per_model: int = 10
    ) -> torch.Tensor:
        """Make predictions aggregating all ensemble members.
        
        Returns:
            Logits of shape (total_samples, batch_size, num_classes)
        """
        all_logits = []
        
        for model, trainer in zip(self.models, self.trainers):
            logits = trainer.predict(x, num_samples_per_model)
            all_logits.append(logits)
        
        # Concatenate along sample dimension
        return torch.cat(all_logits, dim=0)


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_kl_annealing():
    """Test KL annealing schedule."""
    from networks import FullNetwork
    
    model = FullNetwork(NetworkConfig(), "mfvi").to(DEVICE)
    config = MFVIConfig(kl_annealing_epochs=100)
    trainer = MFVITrainer(model, config, DEVICE)
    
    temperature = 1.0
    
    # At epoch 0, weight should be small
    w0 = trainer._get_kl_weight(0, temperature)
    assert w0 == 0.01, f"Expected 0.01, got {w0}"
    
    # At epoch 50, weight should be 0.5 * temperature
    w50 = trainer._get_kl_weight(49, temperature)
    assert abs(w50 - 0.5) < 0.01, f"Expected ~0.5, got {w50}"
    
    # At epoch 100+, weight should be temperature
    w100 = trainer._get_kl_weight(100, temperature)
    assert w100 == temperature, f"Expected {temperature}, got {w100}"
    
    # With temperature 0.1
    w_cold = trainer._get_kl_weight(100, 0.1)
    assert w_cold == 0.1, f"Expected 0.1, got {w_cold}"
    
    print("✓ KL annealing test passed")


def test_learning_rate_schedule():
    """Test learning rate decay schedule."""
    from networks import FullNetwork
    
    model = FullNetwork(NetworkConfig(), "mfvi").to(DEVICE)
    config = MFVIConfig(
        learning_rate_init=0.01,
        learning_rate_decay=0.001
    )
    trainer = MFVITrainer(model, config, DEVICE)
    
    # Before epoch 500
    lr_early = trainer._get_learning_rate(100)
    assert lr_early == 0.01
    
    # After epoch 500
    lr_late = trainer._get_learning_rate(600)
    assert lr_late == 0.001
    
    print("✓ Learning rate schedule test passed")


def test_mfvi_training_step():
    """Test single MFVI training step."""
    from networks import FullNetwork
    
    model = FullNetwork(NetworkConfig(), "mfvi").to(DEVICE)
    config = MFVIConfig()
    trainer = MFVITrainer(model, config, DEVICE)
    
    # Create dummy data
    x = torch.randn(32, 3072).to(DEVICE)
    y = torch.randint(0, 10, (32,)).to(DEVICE)
    
    # Get initial parameters
    initial_mu = model.last_layer.weight_mu.clone()
    
    # Create simple loader
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Freeze feature extractor and train one step
    model.freeze_feature_extractor()
    
    optimizer = torch.optim.Adam(model.last_layer.parameters(), lr=0.01)
    
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        
        with torch.no_grad():
            features = model.get_features(batch_x)
        
        logits = model.last_layer(features, num_samples=1)
        nll = F.cross_entropy(logits, batch_y)
        kl = model.last_layer.kl_divergence()
        loss = nll + 0.01 * kl
        
        loss.backward()
        optimizer.step()
    
    # Parameters should have changed
    assert not torch.allclose(model.last_layer.weight_mu, initial_mu)
    
    print("✓ MFVI training step test passed")


def test_mfvi_prediction():
    """Test MFVI prediction with samples."""
    from networks import FullNetwork
    
    model = FullNetwork(NetworkConfig(), "mfvi").to(DEVICE)
    config = MFVIConfig(num_test_samples=10)
    trainer = MFVITrainer(model, config, DEVICE)
    
    x = torch.randn(32, 3072).to(DEVICE)
    
    # Multiple samples
    logits = trainer.predict(x, num_samples=10)
    assert logits.shape == (10, 32, 10), f"Got shape {logits.shape}"
    
    # Different samples should give different predictions (due to stochasticity)
    # Check variance across samples
    var = logits.var(dim=0).mean().item()
    assert var > 0, "Samples should have variance"
    
    print("✓ MFVI prediction test passed")


def test_mfvi_kl_divergence():
    """Test KL divergence computation."""
    from networks import BayesianLastLayerMFVI
    
    config = NetworkConfig()
    layer = BayesianLastLayerMFVI(config, prior_log_var=0.0).to(DEVICE)
    
    # At initialization, KL should be positive
    kl = layer.kl_divergence()
    assert kl > 0, "KL should be positive"
    
    # If we set posterior = prior, KL should be near 0
    with torch.no_grad():
        layer.weight_mu.zero_()
        layer.bias_mu.zero_()
        layer.weight_log_var.fill_(0.0)  # variance = 1
        layer.bias_log_var.fill_(0.0)
    
    kl_zero = layer.kl_divergence()
    assert kl_zero < 0.01, f"KL should be near 0 when q=p, got {kl_zero}"
    
    print("✓ MFVI KL divergence test passed")


def test_temperature_effect():
    """Test that temperature affects the KL term correctly."""
    from networks import FullNetwork
    
    model = FullNetwork(NetworkConfig(), "mfvi").to(DEVICE)
    config = MFVIConfig()
    trainer = MFVITrainer(model, config, DEVICE)
    
    # At same epoch (after annealing), different temperatures should give different weights
    temp_1 = trainer._get_kl_weight(200, 1.0)
    temp_01 = trainer._get_kl_weight(200, 0.1)
    temp_3 = trainer._get_kl_weight(200, 3.0)
    
    assert temp_01 < temp_1 < temp_3
    assert temp_01 == 0.1
    assert temp_1 == 1.0
    assert temp_3 == 3.0
    
    print("✓ Temperature effect test passed")


if __name__ == "__main__":
    test_kl_annealing()
    test_learning_rate_schedule()
    test_mfvi_training_step()
    test_mfvi_prediction()
    test_mfvi_kl_divergence()
    test_temperature_effect()
    print("\nAll MFVI tests passed!")
