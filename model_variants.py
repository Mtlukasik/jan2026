"""
Model Variants Configuration

Defines all model variants to be trained. Each variant specifies parameters
that override the defaults in config.py.

Usage in pipeline_resnet.py:
    from model_variants import MODEL_VARIANTS
    
    for variant in MODEL_VARIANTS:
        # variant['name'], variant['method'], variant['prior_type'], etc.
"""

from typing import List, Dict, Any

# =============================================================================
# Model Variants Definition
# =============================================================================

MODEL_VARIANTS: List[Dict[str, Any]] = [
    
    # =========================================================================
    # SVGD with Laplace Prior - varying prior_std
    # =========================================================================
    {
        "name": "svgd_laplace_std1",
        "method": "svgd",
        "prior_type": "laplace",
        "prior_std": 1.0,
        "n_particles": 50,
    },
    {
        "name": "svgd_laplace_std0.5",
        "method": "svgd",
        "prior_type": "laplace",
        "prior_std": 0.5,
        "n_particles": 50,
    },
    {
        "name": "svgd_laplace_std2",
        "method": "svgd",
        "prior_type": "laplace",
        "prior_std": 2.0,
        "n_particles": 50,
    },
    
    # =========================================================================
    # SVGD with Gaussian Prior - varying prior_std
    # =========================================================================
    {
        "name": "svgd_gauss_std1",
        "method": "svgd",
        "prior_type": "gaussian",
        "prior_std": 1.0,
        "n_particles": 50,
    },
    {
        "name": "svgd_gauss_std0.5",
        "method": "svgd",
        "prior_type": "gaussian",
        "prior_std": 0.5,
        "n_particles": 50,
    },
    {
        "name": "svgd_gauss_std2",
        "method": "svgd",
        "prior_type": "gaussian",
        "prior_std": 2.0,
        "n_particles": 50,
    },

    
    # =========================================================================
    # MFVI variants - varying prior_std
    # =========================================================================
    {
        "name": "mfvi_std1",
        "method": "mfvi",
        "prior_std": 1.0,
    },
    {
        "name": "mfvi_std0.5",
        "method": "mfvi",
        "prior_std": 0.5,
    },
    {
        "name": "mfvi_std2",
        "method": "mfvi",
        "prior_std": 2.0,
    },
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_variant(name: str) -> Dict[str, Any]:
    """Get variant by name."""
    for v in MODEL_VARIANTS:
        if v["name"] == name:
            return v
    return None


def get_variant_names() -> List[str]:
    """Get all variant names."""
    return [v["name"] for v in MODEL_VARIANTS]


def get_svgd_variants() -> List[Dict[str, Any]]:
    """Get all SVGD variants."""
    return [v for v in MODEL_VARIANTS if v["method"] == "svgd"]


def get_mfvi_variants() -> List[Dict[str, Any]]:
    """Get all MFVI variants."""
    return [v for v in MODEL_VARIANTS if v["method"] == "mfvi"]


def print_variants():
    """Print all variants."""
    print("\n" + "=" * 80)
    print("MODEL VARIANTS")
    print("=" * 80)
    print(f"{'Name':<25} {'Method':<8} {'Prior':<10} {'Std':<6} {'Particles':<10}")
    print("-" * 80)
    
    for v in MODEL_VARIANTS:
        prior = v.get('prior_type', '-')
        particles = str(v.get('n_particles', '-'))
        print(f"{v['name']:<25} {v['method']:<8} {prior:<10} {v.get('prior_std', 1.0):<6} {particles:<10}")
    
    print("=" * 80)


# =============================================================================
# Configuration
# =============================================================================

# Temperatures to test (can be overridden)
TEMPERATURES = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]

# Number of replicates
NUM_REPLICATES = 3

# Active variants (None = all, or list of names)
ACTIVE_VARIANTS = None  # e.g., ["svgd_laplace_std1", "mfvi_std1"]


def get_active_variants() -> List[Dict[str, Any]]:
    """Get variants to actually run."""
    if ACTIVE_VARIANTS is None:
        return MODEL_VARIANTS
    return [v for v in MODEL_VARIANTS if v["name"] in ACTIVE_VARIANTS]


if __name__ == "__main__":
    print_variants()

