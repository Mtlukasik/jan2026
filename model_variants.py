"""
Model Variants Configuration

This file defines all model variants to be trained in the experiment.
Each variant specifies:
- name: unique identifier used in folder names
- method: 'svgd' or 'mfvi'
- prior_type: 'laplace' or 'gaussian' (for SVGD)
- prior_std: standard deviation of prior
- n_particles: number of particles (for SVGD)
- Other method-specific hyperparameters

Usage:
    from model_variants import MODEL_VARIANTS, get_variant, get_variants_by_method
    
    # Get all variants
    for variant in MODEL_VARIANTS:
        print(variant['name'])
    
    # Get specific variant
    variant = get_variant('svgd_laplace_std1')
    
    # Get all SVGD variants
    svgd_variants = get_variants_by_method('svgd')
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ModelVariant:
    """Configuration for a single model variant."""
    
    # Identification
    name: str                           # Unique name (used in folder names)
    method: str                         # 'svgd' or 'mfvi'
    description: str = ""               # Human-readable description
    
    # Prior configuration
    prior_type: str = "laplace"         # 'laplace' or 'gaussian' (SVGD only)
    prior_std: float = 1.0              # Prior standard deviation
    
    # SVGD-specific
    n_particles: int = 50               # Number of particles
    svgd_lr: float = 1e-3               # SVGD learning rate
    kernel_bandwidth_scale: float = 1.0 # RBF kernel bandwidth multiplier
    
    # MFVI-specific
    mfvi_lr: float = 0.01               # MFVI learning rate
    kl_annealing_epochs: int = 100      # KL annealing schedule
    
    # Training
    step2_epochs: int = 150             # Epochs for Step 2 (frozen features)
    step3_epochs: int = 50              # Epochs for Step 3 (joint training)
    
    # Feature learning (Step 3)
    feature_lr: float = 1e-4            # Learning rate for feature extractor
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_run_name(self, temperature: float, replicate: int) -> str:
        """Generate run folder name."""
        return f"last_layer_{self.name}_T{temperature}_replicate_{replicate}"
    
    def get_joint_run_name(self, temperature: float, replicate: int) -> str:
        """Generate joint training run folder name."""
        return f"joint_{self.name}_T{temperature}_replicate_{replicate}"


# =============================================================================
# Define Model Variants
# =============================================================================

MODEL_VARIANTS: List[ModelVariant] = [
    
    # =========================================================================
    # SVGD with Laplace Prior - varying prior_std
    # =========================================================================
    ModelVariant(
        name="svgd_laplace_std0.1",
        method="svgd",
        description="SVGD with tight Laplace prior (std=0.1)",
        prior_type="laplace",
        prior_std=0.1,
        n_particles=50,
    ),
    ModelVariant(
        name="svgd_laplace_std0.5",
        method="svgd",
        description="SVGD with medium-tight Laplace prior (std=0.5)",
        prior_type="laplace",
        prior_std=0.5,
        n_particles=50,
    ),
    ModelVariant(
        name="svgd_laplace_std1",
        method="svgd",
        description="SVGD with standard Laplace prior (std=1.0)",
        prior_type="laplace",
        prior_std=1.0,
        n_particles=50,
    ),
    ModelVariant(
        name="svgd_laplace_std2",
        method="svgd",
        description="SVGD with wide Laplace prior (std=2.0)",
        prior_type="laplace",
        prior_std=2.0,
        n_particles=50,
    ),
    
    # =========================================================================
    # SVGD with Gaussian Prior - varying prior_std
    # =========================================================================
    ModelVariant(
        name="svgd_gauss_std0.1",
        method="svgd",
        description="SVGD with tight Gaussian prior (std=0.1)",
        prior_type="gaussian",
        prior_std=0.1,
        n_particles=50,
    ),
    ModelVariant(
        name="svgd_gauss_std0.5",
        method="svgd",
        description="SVGD with medium-tight Gaussian prior (std=0.5)",
        prior_type="gaussian",
        prior_std=0.5,
        n_particles=50,
    ),
    ModelVariant(
        name="svgd_gauss_std1",
        method="svgd",
        description="SVGD with standard Gaussian prior (std=1.0)",
        prior_type="gaussian",
        prior_std=1.0,
        n_particles=50,
    ),
    ModelVariant(
        name="svgd_gauss_std2",
        method="svgd",
        description="SVGD with wide Gaussian prior (std=2.0)",
        prior_type="gaussian",
        prior_std=2.0,
        n_particles=50,
    ),

    # =========================================================================
    # MFVI variants - varying prior_std
    # =========================================================================
    ModelVariant(
        name="mfvi_std0.1",
        method="mfvi",
        description="MFVI with tight prior (std=0.1)",
        prior_std=0.1,
    ),
    ModelVariant(
        name="mfvi_std0.5",
        method="mfvi",
        description="MFVI with medium-tight prior (std=0.5)",
        prior_std=0.5,
    ),
    ModelVariant(
        name="mfvi_std1",
        method="mfvi",
        description="MFVI with standard prior (std=1.0)",
        prior_std=1.0,
    ),
    ModelVariant(
        name="mfvi_std2",
        method="mfvi",
        description="MFVI with wide prior (std=2.0)",
        prior_std=2.0,
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_variant(name: str) -> Optional[ModelVariant]:
    """Get a variant by name."""
    for v in MODEL_VARIANTS:
        if v.name == name:
            return v
    return None


def get_variants_by_method(method: str) -> List[ModelVariant]:
    """Get all variants for a specific method."""
    return [v for v in MODEL_VARIANTS if v.method == method]


def get_variants_by_prior(prior_type: str) -> List[ModelVariant]:
    """Get all variants with a specific prior type."""
    return [v for v in MODEL_VARIANTS if v.prior_type == prior_type]


def get_variant_names() -> List[str]:
    """Get all variant names."""
    return [v.name for v in MODEL_VARIANTS]


def print_variants():
    """Print all variants in a formatted table."""
    print("\n" + "=" * 100)
    print("MODEL VARIANTS")
    print("=" * 100)
    print(f"{'Name':<25} {'Method':<8} {'Prior':<10} {'Std':<6} {'Particles':<10} Description")
    print("-" * 100)
    
    for v in MODEL_VARIANTS:
        particles = str(v.n_particles) if v.method == 'svgd' else '-'
        prior = v.prior_type if v.method == 'svgd' else 'gaussian'
        print(f"{v.name:<25} {v.method:<8} {prior:<10} {v.prior_std:<6} {particles:<10} {v.description}")
    
    print("=" * 100)
    print(f"Total variants: {len(MODEL_VARIANTS)}")
    print(f"  SVGD: {len(get_variants_by_method('svgd'))}")
    print(f"  MFVI: {len(get_variants_by_method('mfvi'))}")


# =============================================================================
# Custom Configuration (modify this for your experiments)
# =============================================================================

# Temperatures to test
TEMPERATURES = [0.001, 0.01,  0.1,  1.0]

# Number of replicates per configuration
NUM_REPLICATES = 3

# Which variants to actually run (set to None to run all)
# Example: ACTIVE_VARIANTS = ['svgd_laplace_std1', 'svgd_gauss_std1', 'mfvi_std1']
ACTIVE_VARIANTS = None  # Run all variants


def get_active_variants() -> List[ModelVariant]:
    """Get the list of variants to actually run."""
    if ACTIVE_VARIANTS is None:
        return MODEL_VARIANTS
    return [v for v in MODEL_VARIANTS if v.name in ACTIVE_VARIANTS]


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print_variants()
    
    print("\n\nExample run names for svgd_laplace_std1:")
    v = get_variant("svgd_laplace_std1")
    if v:
        for temp in [0.01, 0.1, 1.0]:
            for rep in range(1, 4):
                print(f"  {v.get_run_name(temp, rep)}")
