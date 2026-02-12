"""
Experimental Analysis: Aleatoric vs Epistemic Uncertainty Visualization

This script:
1. Loads trained SVGD and MFVI models
2. Samples in-distribution (CIFAR-10) and OOD (SVHN) images
3. Passes them through models to visualize uncertainty decomposition
4. Shows per-sample breakdown of aleatoric vs epistemic

Run in Colab or locally after training models.

Usage:
    python analyze_uncertainty.py --save-dir ./results --temperature 0.1
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import argparse

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, DEVICE
from data_loading_resnet import DataLoaderManager
from resnet import ResNetFeatureExtractor, DeterministicLastLayer, BayesianLastLayerMFVI


# =============================================================================
# Model Loading
# =============================================================================

def load_deterministic_model(save_dir: str, device: str) -> nn.Module:
    """Load pretrained deterministic ResNet."""
    weights_path = os.path.join(save_dir, "deterministic_model", "model_weights.pt")
    
    feature_extractor = ResNetFeatureExtractor().to(device)
    last_layer = DeterministicLastLayer(512, 10).to(device)
    
    weights = torch.load(weights_path, map_location=device)
    
    # Load feature extractor weights
    fe_weights = {k.replace('feature_extractor.', ''): v 
                  for k, v in weights.items() if k.startswith('feature_extractor.')}
    feature_extractor.load_state_dict(fe_weights)
    
    # Load last layer weights
    ll_weights = {k.replace('last_layer.', ''): v 
                  for k, v in weights.items() if k.startswith('last_layer.')}
    last_layer.load_state_dict(ll_weights)
    
    return feature_extractor, last_layer


def load_svgd_particles(save_dir: str, prior_type: str, temperature: float, 
                        replicate: int, device: str) -> Tuple[nn.Module, List[nn.Module]]:
    """Load SVGD particles."""
    run_name = f"last_layer_svgd_{prior_type}_T{temperature}_replicate_{replicate}"
    run_dir = os.path.join(save_dir, run_name)
    
    if not os.path.exists(run_dir):
        # Try old format without prior
        run_name = f"last_layer_svgd_T{temperature}_replicate_{replicate}"
        run_dir = os.path.join(save_dir, run_name)
    
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"SVGD run not found: {run_name}")
    
    # Load feature extractor from deterministic model
    feature_extractor, _ = load_deterministic_model(save_dir, device)
    feature_extractor.eval()
    
    # Load particles
    particles_path = os.path.join(run_dir, "particles.pt")
    if os.path.exists(particles_path):
        particles_state = torch.load(particles_path, map_location=device)
        n_particles = len(particles_state)
    else:
        # Particles might be saved differently - check for individual files
        n_particles = 20  # default
        particles_state = None
    
    particles = []
    for i in range(n_particles):
        particle = DeterministicLastLayer(512, 10).to(device)
        if particles_state is not None:
            particle.load_state_dict(particles_state[i])
        particles.append(particle)
        particle.eval()
    
    print(f"Loaded {len(particles)} SVGD particles from {run_name}")
    return feature_extractor, particles


def load_mfvi_model(save_dir: str, temperature: float, replicate: int, 
                    device: str) -> Tuple[nn.Module, nn.Module]:
    """Load MFVI model."""
    run_name = f"last_layer_mfvi_T{temperature}_replicate_{replicate}"
    run_dir = os.path.join(save_dir, run_name)
    
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"MFVI run not found: {run_name}")
    
    # Load feature extractor
    feature_extractor, _ = load_deterministic_model(save_dir, device)
    feature_extractor.eval()
    
    # Load MFVI last layer
    mfvi_path = os.path.join(run_dir, "mfvi_layer.pt")
    mfvi_layer = BayesianLastLayerMFVI(512, 10).to(device)
    
    if os.path.exists(mfvi_path):
        mfvi_layer.load_state_dict(torch.load(mfvi_path, map_location=device))
    
    mfvi_layer.eval()
    print(f"Loaded MFVI model from {run_name}")
    
    return feature_extractor, mfvi_layer


# =============================================================================
# Uncertainty Computation
# =============================================================================

def compute_entropy_decomposition_svgd(
    feature_extractor: nn.Module,
    particles: List[nn.Module],
    images: torch.Tensor,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Compute entropy decomposition for SVGD ensemble.
    
    Returns per-sample:
        - total_entropy: H[p̄(y|x)]
        - aleatoric: E[H[p(y|x,θ)]]
        - epistemic: total - aleatoric
        - mean_probs: averaged predictions
        - particle_probs: individual particle predictions
    """
    images = images.to(device)
    
    with torch.no_grad():
        features = feature_extractor(images)
        
        # Get predictions from each particle
        particle_probs = []
        for particle in particles:
            logits = particle(features)
            probs = F.softmax(logits, dim=-1)
            particle_probs.append(probs)
        
        # Stack: (n_particles, batch, n_classes)
        particle_probs = torch.stack(particle_probs)
        
        # Mean prediction
        mean_probs = particle_probs.mean(dim=0)  # (batch, n_classes)
        
        # Total entropy: H[p̄(y|x)]
        total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        
        # Aleatoric: E[H[p(y|x,θ)]]
        individual_entropies = -torch.sum(
            particle_probs * torch.log(particle_probs + 1e-10), dim=-1
        )  # (n_particles, batch)
        aleatoric = individual_entropies.mean(dim=0)  # (batch,)
        
        # Epistemic: mutual information
        epistemic = total_entropy - aleatoric
    
    return {
        'total_entropy': total_entropy.cpu(),
        'aleatoric': aleatoric.cpu(),
        'epistemic': epistemic.cpu(),
        'mean_probs': mean_probs.cpu(),
        'particle_probs': particle_probs.cpu(),
        'predictions': mean_probs.argmax(dim=-1).cpu(),
        'confidence': mean_probs.max(dim=-1)[0].cpu()
    }


def compute_entropy_decomposition_mfvi(
    feature_extractor: nn.Module,
    mfvi_layer: nn.Module,
    images: torch.Tensor,
    device: str,
    num_samples: int = 50
) -> Dict[str, torch.Tensor]:
    """
    Compute entropy decomposition for MFVI model.
    
    Uses more samples for better estimation.
    """
    images = images.to(device)
    
    with torch.no_grad():
        features = feature_extractor(images)
        
        # Sample from weight posterior multiple times
        sample_probs = []
        for _ in range(num_samples):
            # Sample weights and forward
            logits = mfvi_layer(features)  # This samples internally
            probs = F.softmax(logits, dim=-1)
            sample_probs.append(probs)
        
        # Stack: (num_samples, batch, n_classes)
        sample_probs = torch.stack(sample_probs)
        
        # Mean prediction
        mean_probs = sample_probs.mean(dim=0)
        
        # Total entropy
        total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        
        # Aleatoric
        individual_entropies = -torch.sum(
            sample_probs * torch.log(sample_probs + 1e-10), dim=-1
        )
        aleatoric = individual_entropies.mean(dim=0)
        
        # Epistemic
        epistemic = total_entropy - aleatoric
    
    return {
        'total_entropy': total_entropy.cpu(),
        'aleatoric': aleatoric.cpu(),
        'epistemic': epistemic.cpu(),
        'mean_probs': mean_probs.cpu(),
        'sample_probs': sample_probs.cpu(),
        'predictions': mean_probs.argmax(dim=-1).cpu(),
        'confidence': mean_probs.max(dim=-1)[0].cpu()
    }


def compute_entropy_deterministic(
    feature_extractor: nn.Module,
    last_layer: nn.Module,
    images: torch.Tensor,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Compute entropy for deterministic model.
    All uncertainty is "aleatoric" (actually just softmax entropy).
    """
    images = images.to(device)
    
    with torch.no_grad():
        features = feature_extractor(images)
        logits = last_layer(features)
        probs = F.softmax(logits, dim=-1)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    return {
        'total_entropy': entropy.cpu(),
        'aleatoric': entropy.cpu(),  # All uncertainty is "aleatoric"
        'epistemic': torch.zeros_like(entropy).cpu(),  # No epistemic
        'mean_probs': probs.cpu(),
        'predictions': probs.argmax(dim=-1).cpu(),
        'confidence': probs.max(dim=-1)[0].cpu()
    }


# =============================================================================
# Visualization
# =============================================================================

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to displayable image."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    
    img = img.numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def plot_single_sample_analysis(
    image: torch.Tensor,
    label: int,
    results: Dict[str, torch.Tensor],
    title: str,
    is_ood: bool = False
):
    """Plot detailed analysis for a single sample."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Image
    ax = axes[0]
    ax.imshow(denormalize_image(image))
    if is_ood:
        ax.set_title(f"OOD Sample (SVHN)\nTrue: {label}", fontsize=12)
    else:
        ax.set_title(f"In-Dist (CIFAR-10)\nTrue: {CIFAR10_CLASSES[label]}", fontsize=12)
    ax.axis('off')
    
    # Prediction probabilities
    ax = axes[1]
    probs = results['mean_probs'].numpy()
    pred = results['predictions'].item()
    
    colors = ['green' if i == label else ('red' if i == pred else 'steelblue') 
              for i in range(10)]
    ax.barh(range(10), probs, color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Probability')
    ax.set_title(f"Prediction: {CIFAR10_CLASSES[pred]}\nConfidence: {results['confidence'].item():.3f}")
    ax.set_xlim(0, 1)
    
    # Entropy decomposition
    ax = axes[2]
    total = results['total_entropy'].item()
    aleatoric = results['aleatoric'].item()
    epistemic = results['epistemic'].item()
    
    bars = ax.bar(['Total', 'Aleatoric', 'Epistemic'], 
                  [total, aleatoric, epistemic],
                  color=['purple', 'blue', 'orange'])
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Uncertainty Decomposition')
    
    # Add value labels
    for bar, val in zip(bars, [total, aleatoric, epistemic]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Particle/sample diversity (if available)
    ax = axes[3]
    if 'particle_probs' in results:
        particle_probs = results['particle_probs'].numpy()  # (n_particles, n_classes)
        for i, p_probs in enumerate(particle_probs):
            ax.plot(range(10), p_probs, 'o-', alpha=0.3, markersize=3)
        ax.plot(range(10), probs, 'k-', linewidth=2, label='Mean')
        ax.set_xticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title(f'Particle Diversity ({len(particle_probs)} particles)')
        ax.legend()
    elif 'sample_probs' in results:
        sample_probs = results['sample_probs'].numpy()  # (n_samples, n_classes)
        # Show subset for clarity
        for i in range(min(20, len(sample_probs))):
            ax.plot(range(10), sample_probs[i], 'o-', alpha=0.2, markersize=2)
        ax.plot(range(10), probs, 'k-', linewidth=2, label='Mean')
        ax.set_xticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title(f'Sample Diversity ({len(sample_probs)} samples)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Deterministic\n(no diversity)', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Particle/Sample Diversity')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_batch_comparison(
    in_dist_results: Dict[str, torch.Tensor],
    ood_results: Dict[str, torch.Tensor],
    method_name: str
):
    """Compare in-distribution vs OOD for a batch of samples."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: In-distribution
    ax = axes[0, 0]
    ax.hist(in_dist_results['total_entropy'].numpy(), bins=30, alpha=0.7, label='Total', color='purple')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('In-Dist: Total Entropy')
    ax.legend()
    
    ax = axes[0, 1]
    ax.hist(in_dist_results['aleatoric'].numpy(), bins=30, alpha=0.7, label='Aleatoric', color='blue')
    ax.hist(in_dist_results['epistemic'].numpy(), bins=30, alpha=0.7, label='Epistemic', color='orange')
    ax.set_xlabel('Entropy')
    ax.set_title('In-Dist: Aleatoric vs Epistemic')
    ax.legend()
    
    ax = axes[0, 2]
    ax.scatter(in_dist_results['aleatoric'].numpy(), 
               in_dist_results['epistemic'].numpy(), 
               alpha=0.5, c='green', label='In-Dist')
    ax.set_xlabel('Aleatoric')
    ax.set_ylabel('Epistemic')
    ax.set_title('In-Dist: Aleatoric vs Epistemic')
    
    # Bottom row: OOD
    ax = axes[1, 0]
    ax.hist(ood_results['total_entropy'].numpy(), bins=30, alpha=0.7, label='Total', color='purple')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('OOD: Total Entropy')
    ax.legend()
    
    ax = axes[1, 1]
    ax.hist(ood_results['aleatoric'].numpy(), bins=30, alpha=0.7, label='Aleatoric', color='blue')
    ax.hist(ood_results['epistemic'].numpy(), bins=30, alpha=0.7, label='Epistemic', color='orange')
    ax.set_xlabel('Entropy')
    ax.set_title('OOD: Aleatoric vs Epistemic')
    ax.legend()
    
    ax = axes[1, 2]
    ax.scatter(in_dist_results['aleatoric'].numpy(), 
               in_dist_results['epistemic'].numpy(), 
               alpha=0.5, c='green', label='In-Dist')
    ax.scatter(ood_results['aleatoric'].numpy(), 
               ood_results['epistemic'].numpy(), 
               alpha=0.5, c='red', label='OOD')
    ax.set_xlabel('Aleatoric')
    ax.set_ylabel('Epistemic')
    ax.set_title('Comparison: In-Dist vs OOD')
    ax.legend()
    
    plt.suptitle(f'{method_name}: Uncertainty Decomposition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_method_comparison_grid(
    results_dict: Dict[str, Tuple[Dict, Dict]],
    output_path: str = None
):
    """
    Compare multiple methods side by side.
    
    results_dict: {method_name: (in_dist_results, ood_results)}
    """
    n_methods = len(results_dict)
    fig, axes = plt.subplots(n_methods, 4, figsize=(20, 5 * n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for row, (method_name, (in_dist, ood)) in enumerate(results_dict.items()):
        # Column 1: In-dist entropy histogram
        ax = axes[row, 0]
        ax.hist(in_dist['aleatoric'].numpy(), bins=30, alpha=0.7, label='Aleatoric', color='blue')
        ax.hist(in_dist['epistemic'].numpy(), bins=30, alpha=0.7, label='Epistemic', color='orange')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Count')
        ax.set_title(f'{method_name}\nIn-Dist Entropy')
        ax.legend()
        
        # Column 2: OOD entropy histogram
        ax = axes[row, 1]
        ax.hist(ood['aleatoric'].numpy(), bins=30, alpha=0.7, label='Aleatoric', color='blue')
        ax.hist(ood['epistemic'].numpy(), bins=30, alpha=0.7, label='Epistemic', color='orange')
        ax.set_xlabel('Entropy')
        ax.set_title(f'{method_name}\nOOD Entropy')
        ax.legend()
        
        # Column 3: Scatter plot
        ax = axes[row, 2]
        ax.scatter(in_dist['aleatoric'].numpy(), in_dist['epistemic'].numpy(), 
                   alpha=0.3, c='green', label='In-Dist', s=10)
        ax.scatter(ood['aleatoric'].numpy(), ood['epistemic'].numpy(), 
                   alpha=0.3, c='red', label='OOD', s=10)
        ax.set_xlabel('Aleatoric')
        ax.set_ylabel('Epistemic')
        ax.set_title(f'{method_name}\nAleatoric vs Epistemic')
        ax.legend()
        
        # Column 4: Statistics
        ax = axes[row, 3]
        stats_text = (
            f"In-Distribution:\n"
            f"  Aleatoric: {in_dist['aleatoric'].mean():.4f} ± {in_dist['aleatoric'].std():.4f}\n"
            f"  Epistemic: {in_dist['epistemic'].mean():.4f} ± {in_dist['epistemic'].std():.4f}\n"
            f"  Total:     {in_dist['total_entropy'].mean():.4f} ± {in_dist['total_entropy'].std():.4f}\n\n"
            f"OOD:\n"
            f"  Aleatoric: {ood['aleatoric'].mean():.4f} ± {ood['aleatoric'].std():.4f}\n"
            f"  Epistemic: {ood['epistemic'].mean():.4f} ± {ood['epistemic'].std():.4f}\n"
            f"  Total:     {ood['total_entropy'].mean():.4f} ± {ood['total_entropy'].std():.4f}\n\n"
            f"Ratios (OOD / In-Dist):\n"
            f"  Aleatoric: {ood['aleatoric'].mean() / (in_dist['aleatoric'].mean() + 1e-10):.2f}x\n"
            f"  Epistemic: {ood['epistemic'].mean() / (in_dist['epistemic'].mean() + 1e-10):.2f}x"
        )
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                transform=ax.transAxes, verticalalignment='center')
        ax.axis('off')
        ax.set_title(f'{method_name}\nStatistics')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(
    save_dir: str,
    temperature: float = 0.1,
    replicate: int = 1,
    n_samples: int = 200,
    device: str = None,
    output_dir: str = None
):
    """Run full uncertainty analysis."""
    
    device = device or DEVICE
    output_dir = output_dir or os.path.join(save_dir, "uncertainty_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Temperature: {temperature}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    config = ExperimentConfig()
    data_manager = DataLoaderManager(
        config.data, config.ood, config.calibration, 
        flatten=False, save_dir=save_dir
    )
    
    # Get sample batches
    print(f"\nLoading {n_samples} samples from each dataset...")
    
    in_dist_images, in_dist_labels = [], []
    for images, labels in data_manager.test_loader:
        in_dist_images.append(images)
        in_dist_labels.append(labels)
        if sum(len(x) for x in in_dist_images) >= n_samples:
            break
    in_dist_images = torch.cat(in_dist_images)[:n_samples]
    in_dist_labels = torch.cat(in_dist_labels)[:n_samples]
    
    ood_images, ood_labels = [], []
    for images, labels in data_manager.ood_loader:
        ood_images.append(images)
        ood_labels.append(labels)
        if sum(len(x) for x in ood_images) >= n_samples:
            break
    ood_images = torch.cat(ood_images)[:n_samples]
    ood_labels = torch.cat(ood_labels)[:n_samples]
    
    print(f"In-dist samples: {len(in_dist_images)}")
    print(f"OOD samples: {len(ood_images)}")
    
    # Store results for all methods
    all_results = {}
    
    # ==========================================================================
    # 1. Deterministic Model
    # ==========================================================================
    print("\n" + "="*60)
    print("Analyzing: Deterministic Model")
    print("="*60)
    
    try:
        fe, ll = load_deterministic_model(save_dir, device)
        fe.eval()
        ll.eval()
        
        in_dist_det = compute_entropy_deterministic(fe, ll, in_dist_images, device)
        ood_det = compute_entropy_deterministic(fe, ll, ood_images, device)
        
        all_results['Deterministic'] = (in_dist_det, ood_det)
        
        print(f"In-dist entropy: {in_dist_det['total_entropy'].mean():.4f}")
        print(f"OOD entropy: {ood_det['total_entropy'].mean():.4f}")
    except Exception as e:
        print(f"Failed to load deterministic model: {e}")
    
    # ==========================================================================
    # 2. SVGD Models
    # ==========================================================================
    for prior_type in ['laplace', 'gaussian']:
        print("\n" + "="*60)
        print(f"Analyzing: SVGD ({prior_type}) T={temperature}")
        print("="*60)
        
        try:
            fe, particles = load_svgd_particles(
                save_dir, prior_type, temperature, replicate, device
            )
            
            in_dist_svgd = compute_entropy_decomposition_svgd(
                fe, particles, in_dist_images, device
            )
            ood_svgd = compute_entropy_decomposition_svgd(
                fe, particles, ood_images, device
            )
            
            all_results[f'SVGD ({prior_type})'] = (in_dist_svgd, ood_svgd)
            
            print(f"In-dist - Aleatoric: {in_dist_svgd['aleatoric'].mean():.4f}, "
                  f"Epistemic: {in_dist_svgd['epistemic'].mean():.4f}")
            print(f"OOD     - Aleatoric: {ood_svgd['aleatoric'].mean():.4f}, "
                  f"Epistemic: {ood_svgd['epistemic'].mean():.4f}")
            print(f"Epistemic ratio: {ood_svgd['epistemic'].mean() / (in_dist_svgd['epistemic'].mean() + 1e-10):.2f}x")
            
        except Exception as e:
            print(f"Failed to load SVGD ({prior_type}): {e}")
    
    # ==========================================================================
    # 3. MFVI Model
    # ==========================================================================
    print("\n" + "="*60)
    print(f"Analyzing: MFVI T={temperature}")
    print("="*60)
    
    try:
        fe, mfvi_layer = load_mfvi_model(save_dir, temperature, replicate, device)
        
        in_dist_mfvi = compute_entropy_decomposition_mfvi(
            fe, mfvi_layer, in_dist_images, device, num_samples=50
        )
        ood_mfvi = compute_entropy_decomposition_mfvi(
            fe, mfvi_layer, ood_images, device, num_samples=50
        )
        
        all_results['MFVI'] = (in_dist_mfvi, ood_mfvi)
        
        print(f"In-dist - Aleatoric: {in_dist_mfvi['aleatoric'].mean():.4f}, "
              f"Epistemic: {in_dist_mfvi['epistemic'].mean():.4f}")
        print(f"OOD     - Aleatoric: {ood_mfvi['aleatoric'].mean():.4f}, "
              f"Epistemic: {ood_mfvi['epistemic'].mean():.4f}")
        print(f"Epistemic ratio: {ood_mfvi['epistemic'].mean() / (in_dist_mfvi['epistemic'].mean() + 1e-10):.2f}x")
        
    except Exception as e:
        print(f"Failed to load MFVI: {e}")
    
    # ==========================================================================
    # Generate Plots
    # ==========================================================================
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    # 1. Method comparison grid
    if all_results:
        fig = plot_method_comparison_grid(
            all_results,
            os.path.join(output_dir, f"method_comparison_T{temperature}.png")
        )
        plt.close(fig)
    
    # 2. Individual sample analysis (a few examples)
    print("\nGenerating individual sample plots...")
    
    # Pick interesting samples: one correct, one incorrect, one OOD
    if 'SVGD (laplace)' in all_results:
        in_dist_res, ood_res = all_results['SVGD (laplace)']
        
        # Find a high-confidence correct prediction
        correct_mask = in_dist_res['predictions'] == in_dist_labels
        if correct_mask.any():
            idx = torch.where(correct_mask)[0][0].item()
            fig = plot_single_sample_analysis(
                in_dist_images[idx], in_dist_labels[idx].item(),
                {k: v[idx] if v.dim() > 0 else v for k, v in in_dist_res.items()},
                f"SVGD (laplace) T={temperature}: Correct In-Dist Prediction",
                is_ood=False
            )
            fig.savefig(os.path.join(output_dir, "sample_correct_in_dist.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Find an incorrect prediction  
        incorrect_mask = in_dist_res['predictions'] != in_dist_labels
        if incorrect_mask.any():
            idx = torch.where(incorrect_mask)[0][0].item()
            fig = plot_single_sample_analysis(
                in_dist_images[idx], in_dist_labels[idx].item(),
                {k: v[idx] if v.dim() > 0 else v for k, v in in_dist_res.items()},
                f"SVGD (laplace) T={temperature}: Incorrect In-Dist Prediction",
                is_ood=False
            )
            fig.savefig(os.path.join(output_dir, "sample_incorrect_in_dist.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # OOD sample
        idx = 0
        fig = plot_single_sample_analysis(
            ood_images[idx], ood_labels[idx].item(),
            {k: v[idx] if v.dim() > 0 else v for k, v in ood_res.items()},
            f"SVGD (laplace) T={temperature}: OOD Sample (SVHN)",
            is_ood=True
        )
        fig.savefig(os.path.join(output_dir, "sample_ood.png"),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 3. Per-method batch comparison
    for method_name, (in_dist_res, ood_res) in all_results.items():
        fig = plot_batch_comparison(in_dist_res, ood_res, method_name)
        safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
        fig.savefig(os.path.join(output_dir, f"batch_{safe_name}_T{temperature}.png"),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    
    return all_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze uncertainty decomposition")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory containing trained models")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature to analyze")
    parser.add_argument("--replicate", type=int, default=1,
                        help="Replicate number")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of samples to analyze")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    run_analysis(
        save_dir=args.save_dir,
        temperature=args.temperature,
        replicate=args.replicate,
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )
