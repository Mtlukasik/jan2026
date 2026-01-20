"""
Configuration and constants for BNN comparison experiment.

This module contains all hyperparameters and configuration settings
for comparing MCMC vs MFVI last-layer Bayesian inference on CIFAR-10.

Based on: "Bayesian Neural Network Priors Revisited" (Fortuin et al., ICLR 2022)
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed for reproducibility
SEED = 42


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    batch_size: int = 128
    num_workers: int = 4
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
    activation: str = "relu"  # Options: "relu", "tanh", "sigmoid"


@dataclass 
class MCMCConfig:
    """Configuration for MCMC (SG-MCMC) inference on last layer.
    
    Based on the paper's approach using Gradient-Guided Monte Carlo (GG-MC)
    with cyclical learning rate schedule.
    """
    num_cycles: int = 60  # Number of SG-MCMC cycles
    epochs_per_cycle: int = 45
    samples_per_cycle: int = 5  # Samples taken from last 5 epochs
    burn_in_samples: int = 50  # Samples to discard
    learning_rate_init: float = 0.01
    # Only add Langevin noise in last 15 epochs of each cycle
    noise_epochs: int = 15
    # Number of chains for diagnostics
    num_chains: int = 5
    

@dataclass
class MFVIConfig:
    """Configuration for Mean-Field Variational Inference on last layer.
    
    Based on Blundell et al. (2015) and the paper's appendix A.11.
    """
    num_epochs: int = 200
    batch_size: int = 500
    learning_rate_init: float = 0.01
    learning_rate_decay: float = 0.001  # After 500 epochs
    # KL annealing for first 100 epochs
    kl_annealing_epochs: int = 100
    # Number of samples for training/testing
    num_train_samples: int = 1
    num_test_samples: int = 10
    # Prior variance
    prior_log_var: float = 0.0  # log(1.0)


@dataclass
class SVGDConfig:
    """Configuration for Stein Variational Gradient Descent inference.
    
    SVGD maintains an ensemble of particles that approximate the posterior.
    Based on Liu & Wang (2016) "Stein Variational Gradient Descent".
    """
    n_particles: int =  100  # Number of particles in ensemble
    svgd_lr: float = 1e-3  # Learning rate for SVGD updates
    feature_lr: float = 1e-3  # Learning rate for feature extractor
    prior_std: float = 1.0  # Prior standard deviation
    bandwidth_scale: float = 1.0  # Kernel bandwidth scaling factor
    num_epochs: int = 200  # Total training epochs
    use_laplace_prior: bool = True  # Use Laplace prior (vs Gaussian)
    weight_decay: float = 5e-4  # Weight decay for feature optimizer
    grad_clip: float = 1.0  # Gradient clipping norm


@dataclass
class TemperatureConfig:
    """Temperature settings for cold/warm posterior experiments.
    
    The paper tests temperatures: 0.001, 0.01, 0.03, 0.1, 0.3, 1.0
    and some experiments include T > 1 (warm posteriors).
    """
    temperatures: List[float] = field(default_factory=lambda: [
        0.001, 0.01, 0.03, 0.1, 0.3, 1.0
    ])
    # Extended temperatures including warm posteriors
    extended_temperatures: List[float] = field(default_factory=lambda: [
        0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
    ])


@dataclass
class OODConfig:
    """Configuration for Out-of-Distribution detection experiments.
    
    The paper uses different OOD datasets based on the in-distribution:
    - For MNIST: FashionMNIST as OOD
    - For FashionMNIST: MNIST as OOD  
    - For CIFAR-10: SVHN as OOD (common choice in literature)
    
    We'll use SVHN as OOD for CIFAR-10 in-distribution.
    """
    # OOD dataset for CIFAR-10
    ood_dataset: str = "SVHN"
    # SVHN statistics (for normalization consistency)
    svhn_mean: Tuple[float, float, float] = (0.4377, 0.4438, 0.4728)
    svhn_std: Tuple[float, float, float] = (0.1980, 0.2010, 0.1970)


@dataclass
class CalibrationConfig:
    """Configuration for calibration measurement.
    
    The paper uses Expected Calibration Error (ECE) with rotated
    inputs for calibration under distribution shift.
    """
    num_bins: int = 15  # For ECE computation
    # Rotation angles for calibration under shift (degrees)
    rotation_angles: List[float] = field(default_factory=lambda: [
        0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180
    ])


@dataclass
class ExperimentConfig:
    """Main configuration combining all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
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
