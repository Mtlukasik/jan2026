"""
Configuration for SVGD vs MFVI Last-Layer Bayesian Inference Comparison.

This module contains all hyperparameters and configuration settings
for comparing SVGD vs MFVI last-layer Bayesian inference on CIFAR-10.

Based on: "Bayesian Neural Network Priors Revisited" (Fortuin et al., ICLR 2022)
"""

import torch
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Device configuration with better detection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Print device info
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Random seed for reproducibility
SEED = 42

# Auto-detect optimal num_workers
def get_num_workers():
    try:
        cpu_count = os.cpu_count() or 2
        # Use fewer workers to avoid issues
        return min(2, cpu_count)
    except:
        return 0


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    batch_size: int = 128
    num_workers: int = field(default_factory=get_num_workers)
    # CIFAR-10 statistics
    cifar10_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    cifar10_std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    # Input dimension after flattening (32x32x3)
    input_dim: int = 3072
    num_classes: int = 10


@dataclass
class NetworkConfig:
    """Configuration for the 2-layer fully connected network."""
    input_dim: int = 3072  # 32*32*3 for CIFAR-10
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 10   # CIFAR-10 classes
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"


@dataclass
class MFVIConfig:
    """Configuration for Mean-Field Variational Inference on last layer."""
    num_epochs: int = 500
    batch_size: int = 128
    learning_rate_init: float = 0.01
    learning_rate_decay: float = 0.001
    kl_annealing_epochs: int = 100
    num_train_samples: int = 1
    num_test_samples: int = 10
    prior_log_var: float = 0.0


@dataclass
class SVGDConfig:
    """Configuration for Stein Variational Gradient Descent inference."""
    n_particles: int = 20           # Number of particles in ensemble
    svgd_lr: float = 1e-3           # Learning rate for SVGD updates
    feature_lr: float = 1e-4        # Learning rate for feature extractor (Phase 2/Step 3)
    prior_std: float = 1.0          # Prior standard deviation
    bandwidth_scale: float = 1.0    # Kernel bandwidth scaling factor
    num_epochs: int = 200           # Total training epochs (Phase 1/Step 2)
    phase2_epochs: int = 50         # Joint training epochs (Phase 2/Step 3), 0 to skip
    use_laplace_prior: bool = True  # Use Laplace prior (vs Gaussian)
    weight_decay: float = 5e-4      # Weight decay for feature optimizer (Phase 2)
    grad_clip: float = 1.0          # Gradient clipping norm
    init_std: float = 0.01          # Std for particle initialization perturbation


@dataclass
class TemperatureConfig:
    """Temperature settings for cold/warm posterior experiments."""
    temperatures: List[float] = field(default_factory=lambda: [
        0.001, 0.01, 0.03, 0.1, 0.3, 1.0
    ])


@dataclass
class OODConfig:
    """Configuration for Out-of-Distribution detection (SVHN as OOD)."""
    ood_dataset: str = "SVHN"
    svhn_mean: Tuple[float, float, float] = (0.4377, 0.4438, 0.4728)
    svhn_std: Tuple[float, float, float] = (0.1980, 0.2010, 0.1970)


@dataclass
class CalibrationConfig:
    """Configuration for calibration measurement (ECE)."""
    num_bins: int = 15
    rotation_angles: List[float] = field(default_factory=lambda: [
        0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180
    ])


@dataclass
class ExperimentConfig:
    """Main configuration combining all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mfvi: MFVIConfig = field(default_factory=MFVIConfig)
    svgd: SVGDConfig = field(default_factory=SVGDConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    
    # Experiment settings
    num_replicates: int = 3  # Number of independent runs for statistics
    save_dir: str = "./results"
    seed: int = SEED


# Default configuration instance
default_config = ExperimentConfig()
