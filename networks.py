"""
Neural Network Architecture for BNN comparison experiment.

This module implements a 2-layer fully connected network with:
- Batch Normalization
- Dropout
- Configurable activations (ReLU, Tanh, Sigmoid)

Architecture Design Decisions:
-----------------------------
1. No convolutions as per specification - pure FCNN
2. BatchNorm after linear layers (before activation) for training stability
3. Dropout for regularization 
4. The network is split into:
   - Feature extractor (layers before the last linear layer)
   - Last layer (for Bayesian inference)

This separation allows us to:
- Train the feature extractor with standard SGD
- Apply MCMC or MFVI only to the last layer
- This is computationally efficient and follows common practice in BDL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from config import NetworkConfig, DEVICE


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        name: One of "relu", "tanh", "sigmoid"
        
    Returns:
        Activation module
    """
    activations = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]


class FeatureExtractor(nn.Module):
    """Feature extractor: Input -> Hidden representation.
    
    This is the deterministic part of the network that will be
    trained with standard SGD before applying Bayesian inference
    to the last layer.
    
    Architecture:
    Input (3072) -> Linear -> BatchNorm -> Activation -> Dropout -> Output (hidden_dim)
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # First layer: input -> hidden
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Batch normalization
        #self.bn1 = nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity()
        
        # Activation
        self.activation = get_activation(config.activation)
        
        # Dropout
        #self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization for ReLU."""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Features of shape (batch_size, hidden_dim)
        """
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.activation(x)
        #x = self.dropout(x)
        return x


class DeterministicLastLayer(nn.Module):
    """Deterministic last layer for baseline comparison.
    
    Architecture:
    Hidden (hidden_dim) -> Linear -> Output (num_classes)
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Features of shape (batch_size, hidden_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.fc(x)


class BayesianLastLayerMFVI(nn.Module):
    """Mean-Field Variational Inference last layer.
    
    Implements the reparameterization trick for variational inference.
    Each weight and bias has a mean and log-variance parameter.
    
    The variational posterior is:
    q(w) = N(mu_w, sigma_w^2)  (diagonal Gaussian)
    
    We use the local reparameterization trick for efficiency:
    Instead of sampling weights, we sample the pre-activations directly.
    """
    
    def __init__(self, config: NetworkConfig, prior_log_var: float = 0.0):
        super().__init__()
        self.config = config
        self.prior_log_var = prior_log_var  # log(prior_variance)
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(
            torch.empty(config.output_dim, config.hidden_dim)
        )
        self.weight_log_var = nn.Parameter(
            torch.empty(config.output_dim, config.hidden_dim)
        )
        
        # Variational parameters for biases
        self.bias_mu = nn.Parameter(torch.empty(config.output_dim))
        self.bias_log_var = nn.Parameter(torch.empty(config.output_dim))
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize variational parameters.
        
        Initialize means with He initialization.
        Initialize log-variances to small negative values (small initial variance).
        """
        # Initialize means
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias_mu)
        
        # Initialize log-variances (start with small variance)
        nn.init.constant_(self.weight_log_var, -5.0)
        nn.init.constant_(self.bias_log_var, -5.0)
    
    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Forward pass with local reparameterization trick.
        
        Args:
            x: Features of shape (batch_size, hidden_dim)
            num_samples: Number of samples for prediction
            
        Returns:
            Logits of shape (num_samples, batch_size, num_classes) if num_samples > 1
            or (batch_size, num_classes) if num_samples == 1
        """
        batch_size = x.shape[0]
        
        if num_samples == 1:
            return self._single_sample_forward(x)
        else:
            # Multiple samples for prediction
            outputs = []
            for _ in range(num_samples):
                outputs.append(self._single_sample_forward(x))
            return torch.stack(outputs)
    
    def _single_sample_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single sample forward pass using local reparameterization."""
        # Compute mean of pre-activation
        mean = F.linear(x, self.weight_mu, self.bias_mu)
        
        # Compute variance of pre-activation (local reparameterization)
        # Var[Wx] = x^2 * Var[W] (for diagonal covariance)
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)
        
        var = F.linear(x.pow(2), weight_var, bias_var)
        std = torch.sqrt(var + 1e-8)
        
        # Sample from N(mean, var) using reparameterization
        eps = torch.randn_like(mean)
        return mean + std * eps
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence KL(q(w)||p(w)) for regularization.
        
        For Gaussian prior and posterior:
        KL = 0.5 * sum(exp(log_var) / prior_var + (mu - prior_mu)^2 / prior_var 
                       - 1 - log_var + log(prior_var))
        
        With zero-mean prior:
        KL = 0.5 * sum(exp(log_var - prior_log_var) + mu^2 * exp(-prior_log_var) 
                       - 1 - log_var + prior_log_var)
        """
        prior_var = torch.exp(torch.tensor(self.prior_log_var, device=self.weight_mu.device))
        
        # KL for weights
        kl_weights = 0.5 * (
            torch.exp(self.weight_log_var) / prior_var +
            self.weight_mu.pow(2) / prior_var -
            1 -
            self.weight_log_var +
            self.prior_log_var
        ).sum()
        
        # KL for biases
        kl_biases = 0.5 * (
            torch.exp(self.bias_log_var) / prior_var +
            self.bias_mu.pow(2) / prior_var -
            1 -
            self.bias_log_var +
            self.prior_log_var
        ).sum()
        
        return kl_weights + kl_biases


class BayesianLastLayerMCMC(nn.Module):
    """Last layer for MCMC inference.
    
    This is essentially a deterministic layer, but we store
    samples from the posterior obtained via MCMC.
    
    During training: We use SG-MCMC to sample from the posterior.
    During inference: We average predictions over stored samples.
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Current parameters (updated by MCMC)
        self.weight = nn.Parameter(
            torch.empty(config.output_dim, config.hidden_dim)
        )
        self.bias = nn.Parameter(torch.empty(config.output_dim))
        
        # Storage for posterior samples
        self.weight_samples = []
        self.bias_samples = []
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, use_samples: bool = False) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Features of shape (batch_size, hidden_dim)
            use_samples: If True, average over stored samples for prediction
            
        Returns:
            Logits of shape (batch_size, num_classes) or 
            (num_samples, batch_size, num_classes) if returning all sample predictions
        """
        if use_samples and len(self.weight_samples) > 0:
            return self._sample_forward(x)
        else:
            return F.linear(x, self.weight, self.bias)
    
    def _sample_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging over stored MCMC samples."""
        outputs = []
        for w, b in zip(self.weight_samples, self.bias_samples):
            outputs.append(F.linear(x, w, b))
        return torch.stack(outputs)
    
    def store_sample(self):
        """Store current parameters as a posterior sample."""
        self.weight_samples.append(self.weight.detach().clone())
        self.bias_samples.append(self.bias.detach().clone())
    
    def clear_samples(self):
        """Clear stored samples."""
        self.weight_samples = []
        self.bias_samples = []
    
    @property
    def num_samples(self) -> int:
        return len(self.weight_samples)


class FullNetwork(nn.Module):
    """Complete network combining feature extractor and last layer.
    
    This is the main model class that wraps everything together.
    It supports both deterministic, MFVI, and MCMC last layers.
    """
    
    def __init__(self, config: NetworkConfig, last_layer_type: str = "deterministic",
                 prior_log_var: float = 0.0):
        """
        Args:
            config: Network configuration
            last_layer_type: One of "deterministic", "mfvi", "mcmc"
            prior_log_var: Prior log variance for MFVI
        """
        super().__init__()
        self.config = config
        self.last_layer_type = last_layer_type
        
        # Feature extractor (shared for all variants)
        self.feature_extractor = FeatureExtractor(config)
        
        # Last layer (depends on inference type)
        if last_layer_type == "deterministic":
            self.last_layer = DeterministicLastLayer(config)
        elif last_layer_type == "mfvi":
            self.last_layer = BayesianLastLayerMFVI(config, prior_log_var)
        elif last_layer_type == "mcmc":
            self.last_layer = BayesianLastLayerMCMC(config)
        else:
            raise ValueError(f"Unknown last_layer_type: {last_layer_type}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the full network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            **kwargs: Additional arguments for last layer
            
        Returns:
            Logits
        """
        features = self.feature_extractor(x)
        return self.last_layer(features, **kwargs)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without passing through last layer."""
        return self.feature_extractor(x)
    
    def freeze_feature_extractor(self):
        """Freeze feature extractor for last-layer Bayesian inference."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """Unfreeze feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_feature_extractor():
    """Test FeatureExtractor forward pass."""
    config = NetworkConfig()
    extractor = FeatureExtractor(config).to(DEVICE)
    
    # Test forward pass
    x = torch.randn(32, config.input_dim).to(DEVICE)
    features = extractor(x)
    
    assert features.shape == (32, config.hidden_dim), \
        f"Expected (32, {config.hidden_dim}), got {features.shape}"
    
    # Test eval mode (BatchNorm behavior changes)
    extractor.eval()
    features_eval = extractor(x)
    assert features_eval.shape == (32, config.hidden_dim)
    
    print("✓ FeatureExtractor test passed")


def test_deterministic_last_layer():
    """Test DeterministicLastLayer."""
    config = NetworkConfig()
    layer = DeterministicLastLayer(config).to(DEVICE)
    
    x = torch.randn(32, config.hidden_dim).to(DEVICE)
    output = layer(x)
    
    assert output.shape == (32, config.output_dim), \
        f"Expected (32, {config.output_dim}), got {output.shape}"
    
    print("✓ DeterministicLastLayer test passed")


def test_mfvi_last_layer():
    """Test BayesianLastLayerMFVI."""
    config = NetworkConfig()
    layer = BayesianLastLayerMFVI(config).to(DEVICE)
    
    x = torch.randn(32, config.hidden_dim).to(DEVICE)
    
    # Single sample
    output = layer(x, num_samples=1)
    assert output.shape == (32, config.output_dim), \
        f"Expected (32, {config.output_dim}), got {output.shape}"
    
    # Multiple samples
    output = layer(x, num_samples=5)
    assert output.shape == (5, 32, config.output_dim), \
        f"Expected (5, 32, {config.output_dim}), got {output.shape}"
    
    # KL divergence
    kl = layer.kl_divergence()
    assert kl.ndim == 0, "KL should be a scalar"
    assert kl >= 0, "KL should be non-negative"
    
    print("✓ BayesianLastLayerMFVI test passed")


def test_mcmc_last_layer():
    """Test BayesianLastLayerMCMC."""
    config = NetworkConfig()
    layer = BayesianLastLayerMCMC(config).to(DEVICE)
    
    x = torch.randn(32, config.hidden_dim).to(DEVICE)
    
    # Without samples
    output = layer(x)
    assert output.shape == (32, config.output_dim)
    
    # Store some samples
    for _ in range(5):
        layer.weight.data = torch.randn_like(layer.weight)
        layer.bias.data = torch.randn_like(layer.bias)
        layer.store_sample()
    
    assert layer.num_samples == 5
    
    # With samples
    output = layer(x, use_samples=True)
    assert output.shape == (5, 32, config.output_dim), \
        f"Expected (5, 32, {config.output_dim}), got {output.shape}"
    
    # Clear samples
    layer.clear_samples()
    assert layer.num_samples == 0
    
    print("✓ BayesianLastLayerMCMC test passed")


def test_full_network():
    """Test FullNetwork with different last layer types."""
    config = NetworkConfig()
    x = torch.randn(32, config.input_dim).to(DEVICE)
    
    for last_layer_type in ["deterministic", "mfvi", "mcmc"]:
        model = FullNetwork(config, last_layer_type).to(DEVICE)
        
        # Test forward pass
        if last_layer_type == "mfvi":
            output = model(x, num_samples=1)
        elif last_layer_type == "mcmc":
            output = model(x, use_samples=False)
        else:
            output = model(x)
        
        assert output.shape == (32, config.output_dim), \
            f"{last_layer_type}: Expected (32, {config.output_dim}), got {output.shape}"
        
        # Test feature extraction
        features = model.get_features(x)
        assert features.shape == (32, config.hidden_dim)
        
        # Test freeze/unfreeze
        model.freeze_feature_extractor()
        for param in model.feature_extractor.parameters():
            assert not param.requires_grad
        
        model.unfreeze_feature_extractor()
        for param in model.feature_extractor.parameters():
            assert param.requires_grad
    
    print("✓ FullNetwork test passed")


def test_gradient_flow():
    """Test that gradients flow correctly through the network."""
    config = NetworkConfig()
    model = FullNetwork(config, "mfvi").to(DEVICE)
    
    x = torch.randn(32, config.input_dim).to(DEVICE)
    y = torch.randint(0, config.output_dim, (32,)).to(DEVICE)
    
    output = model(x, num_samples=1)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    
    print("✓ Gradient flow test passed")


if __name__ == "__main__":
    test_feature_extractor()
    test_deterministic_last_layer()
    test_mfvi_last_layer()
    test_mcmc_last_layer()
    test_full_network()
    test_gradient_flow()
    print("\nAll network tests passed!")
