"""
ResNet Architecture for CIFAR-10 with Bayesian Last Layer Support.

This module implements ResNet-18 adapted for CIFAR-10 (32x32 images):
- Modified first conv layer (3x3 instead of 7x7, no maxpool)
- Standard ResNet blocks with BatchNorm
- Supports deterministic, MFVI, and SVGD last layers

Architecture:
    Input (3, 32, 32)
    -> Conv3x3(64) + BN + ReLU
    -> Layer1: 2x BasicBlock(64)
    -> Layer2: 2x BasicBlock(128), stride=2
    -> Layer3: 2x BasicBlock(256), stride=2  
    -> Layer4: 2x BasicBlock(512), stride=2
    -> AdaptiveAvgPool -> 512-dim features
    -> Last Layer (deterministic/MFVI/SVGD) -> 10 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type, Union
from config import NetworkConfig, DEVICE


class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convolutions."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetFeatureExtractor(nn.Module):
    """ResNet-18 feature extractor for CIFAR-10.
    
    Modified for 32x32 images:
    - First conv is 3x3 with stride 1 (not 7x7 with stride 2)
    - No max pooling after first conv
    - Output: 512-dimensional feature vector
    """
    
    def __init__(
        self,
        block: Type[BasicBlock] = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],  # ResNet-18 config
        num_classes: int = 10
    ):
        super().__init__()
        
        self.in_channels = 64
        
        # First conv layer - modified for CIFAR-10 (32x32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool for CIFAR-10
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output dimension
        self.output_dim = 512 * block.expansion
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a ResNet layer with multiple blocks."""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Features of shape (batch_size, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class DeterministicLastLayer(nn.Module):
    """Standard linear last layer."""
    
    def __init__(self, input_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class BayesianLastLayerMFVI(nn.Module):
    """Mean-Field Variational Inference last layer with local reparameterization."""
    
    def __init__(self, input_dim: int = 512, num_classes: int = 10, prior_log_var: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prior_log_var = prior_log_var
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(num_classes, input_dim))
        self.weight_log_var = nn.Parameter(torch.empty(num_classes, input_dim))
        
        # Variational parameters for biases
        self.bias_mu = nn.Parameter(torch.empty(num_classes))
        self.bias_log_var = nn.Parameter(torch.empty(num_classes))
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.weight_log_var, -5.0)
        nn.init.constant_(self.bias_log_var, -5.0)
    
    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        if num_samples == 1:
            return self._single_sample_forward(x)
        else:
            outputs = [self._single_sample_forward(x) for _ in range(num_samples)]
            return torch.stack(outputs)
    
    def _single_sample_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Local reparameterization trick."""
        mean = F.linear(x, self.weight_mu, self.bias_mu)
        
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)
        
        var = F.linear(x.pow(2), weight_var, bias_var)
        std = torch.sqrt(var + 1e-8)
        
        eps = torch.randn_like(mean)
        return mean + std * eps
    
    def kl_divergence(self) -> torch.Tensor:
        """KL divergence from posterior to prior."""
        prior_var = torch.exp(torch.tensor(self.prior_log_var, device=self.weight_mu.device))
        
        kl_weights = 0.5 * (
            torch.exp(self.weight_log_var) / prior_var +
            self.weight_mu.pow(2) / prior_var -
            1 - self.weight_log_var + self.prior_log_var
        ).sum()
        
        kl_biases = 0.5 * (
            torch.exp(self.bias_log_var) / prior_var +
            self.bias_mu.pow(2) / prior_var -
            1 - self.bias_log_var + self.prior_log_var
        ).sum()
        
        return kl_weights + kl_biases


class ResNetForBayesianLastLayer(nn.Module):
    """Complete ResNet-18 with support for Bayesian last layer.
    
    This model separates the feature extractor (ResNet backbone) from
    the last layer, allowing us to:
    1. Train the full model with standard SGD
    2. Freeze the feature extractor
    3. Replace/train only the last layer with Bayesian methods
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        last_layer_type: str = "deterministic",
        prior_log_var: float = 0.0
    ):
        super().__init__()
        
        self.last_layer_type = last_layer_type
        
        # Feature extractor (ResNet-18 backbone)
        self.feature_extractor = ResNetFeatureExtractor()
        self.feature_dim = self.feature_extractor.output_dim  # 512
        
        # Last layer
        if last_layer_type == "deterministic":
            self.last_layer = DeterministicLastLayer(self.feature_dim, num_classes)
        elif last_layer_type == "mfvi":
            self.last_layer = BayesianLastLayerMFVI(self.feature_dim, num_classes, prior_log_var)
        else:
            raise ValueError(f"Unknown last_layer_type: {last_layer_type}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.last_layer(features, **kwargs)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.feature_extractor(x)
    
    def freeze_feature_extractor(self):
        """Freeze feature extractor for Bayesian last layer training."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Set to eval mode to freeze BatchNorm statistics
        self.feature_extractor.eval()
    
    def unfreeze_feature_extractor(self):
        """Unfreeze feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.feature_extractor.train()
    
    def train(self, mode: bool = True):
        """Override train to keep feature extractor in eval if frozen."""
        super().train(mode)
        # If feature extractor is frozen, keep it in eval mode
        if not any(p.requires_grad for p in self.feature_extractor.parameters()):
            self.feature_extractor.eval()
        return self


class SVGDParticleResNet(nn.Module):
    """Single SVGD particle using ResNet feature extractor.
    
    For SVGD, each particle has its own last layer but can share
    (or have copies of) the feature extractor.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained_features: Optional[nn.Module] = None
    ):
        super().__init__()
        
        # Feature extractor
        if pretrained_features is not None:
            self.feature_extractor = pretrained_features
        else:
            self.feature_extractor = ResNetFeatureExtractor()
        
        self.feature_dim = 512
        
        # Last layer (trainable via SVGD)
        self.last_layer = nn.Linear(self.feature_dim, num_classes)
        self._init_last_layer()
    
    def _init_last_layer(self):
        nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.last_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.last_layer(features)
    
    def get_last_layer_params(self) -> torch.Tensor:
        """Get flattened last layer parameters for SVGD."""
        return torch.cat([
            self.last_layer.weight.flatten(),
            self.last_layer.bias.flatten()
        ])
    
    def set_last_layer_params(self, params: torch.Tensor):
        """Set last layer parameters from flattened vector."""
        weight_size = self.last_layer.weight.numel()
        weight = params[:weight_size].view_as(self.last_layer.weight)
        bias = params[weight_size:].view_as(self.last_layer.bias)
        
        self.last_layer.weight.data.copy_(weight)
        self.last_layer.bias.data.copy_(bias)


# =============================================================================
# Factory function for easy model creation
# =============================================================================

def create_resnet_model(
    last_layer_type: str = "deterministic",
    num_classes: int = 10,
    prior_log_var: float = 0.0
) -> ResNetForBayesianLastLayer:
    """Create a ResNet-18 model for CIFAR-10.
    
    Args:
        last_layer_type: "deterministic" or "mfvi"
        num_classes: Number of output classes
        prior_log_var: Prior log variance for MFVI
        
    Returns:
        ResNet model with specified last layer
    """
    return ResNetForBayesianLastLayer(
        num_classes=num_classes,
        last_layer_type=last_layer_type,
        prior_log_var=prior_log_var
    )


# =============================================================================
# Unit Tests
# =============================================================================

def test_basic_block():
    """Test BasicBlock."""
    block = BasicBlock(64, 64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == (2, 64, 32, 32), f"Got {out.shape}"
    print("✓ BasicBlock test passed")


def test_feature_extractor():
    """Test ResNet feature extractor."""
    fe = ResNetFeatureExtractor()
    x = torch.randn(2, 3, 32, 32)
    features = fe(x)
    assert features.shape == (2, 512), f"Got {features.shape}"
    print("✓ ResNetFeatureExtractor test passed")


def test_full_model():
    """Test full ResNet model."""
    model = ResNetForBayesianLastLayer(num_classes=10, last_layer_type="deterministic")
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), f"Got {out.shape}"
    print("✓ ResNetForBayesianLastLayer (deterministic) test passed")


def test_mfvi_model():
    """Test MFVI model."""
    model = ResNetForBayesianLastLayer(num_classes=10, last_layer_type="mfvi")
    x = torch.randn(2, 3, 32, 32)
    
    # Single sample
    out = model(x, num_samples=1)
    assert out.shape == (2, 10), f"Got {out.shape}"
    
    # Multiple samples
    out = model(x, num_samples=5)
    assert out.shape == (5, 2, 10), f"Got {out.shape}"
    
    # KL divergence
    kl = model.last_layer.kl_divergence()
    assert kl.item() >= 0
    
    print("✓ ResNetForBayesianLastLayer (MFVI) test passed")


def test_freeze_unfreeze():
    """Test freezing feature extractor."""
    model = ResNetForBayesianLastLayer(num_classes=10, last_layer_type="deterministic")
    
    # Check initially trainable
    assert all(p.requires_grad for p in model.feature_extractor.parameters())
    
    # Freeze
    model.freeze_feature_extractor()
    assert not any(p.requires_grad for p in model.feature_extractor.parameters())
    
    # Last layer should still be trainable
    assert all(p.requires_grad for p in model.last_layer.parameters())
    
    # Unfreeze
    model.unfreeze_feature_extractor()
    assert all(p.requires_grad for p in model.feature_extractor.parameters())
    
    print("✓ Freeze/unfreeze test passed")


def test_svgd_particle():
    """Test SVGD particle."""
    particle = SVGDParticleResNet(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = particle(x)
    assert out.shape == (2, 10), f"Got {out.shape}"
    
    # Test parameter get/set
    params = particle.get_last_layer_params()
    expected_size = 512 * 10 + 10  # weight + bias
    assert params.shape == (expected_size,), f"Got {params.shape}"
    
    new_params = torch.randn_like(params)
    particle.set_last_layer_params(new_params)
    retrieved = particle.get_last_layer_params()
    assert torch.allclose(new_params, retrieved)
    
    print("✓ SVGDParticleResNet test passed")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_parameter_count():
    """Check parameter count is reasonable for ResNet-18."""
    model = ResNetForBayesianLastLayer(num_classes=10)
    total = count_parameters(model)
    # ResNet-18 for CIFAR-10 should have ~11M parameters
    print(f"Total parameters: {total:,}")
    assert 10_000_000 < total < 15_000_000, f"Unexpected param count: {total}"
    print("✓ Parameter count test passed")


if __name__ == "__main__":
    test_basic_block()
    test_feature_extractor()
    test_full_model()
    test_mfvi_model()
    test_freeze_unfreeze()
    test_svgd_particle()
    test_parameter_count()
    print("\n✓ All ResNet tests passed!")
