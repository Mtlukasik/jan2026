"""
Evaluation Metrics for BNN comparison experiment.

This module implements the four key metrics from the paper:
1. Error (Test Error / Classification Error Rate)
2. Likelihood (Negative Log-Likelihood)
3. Calibration (Expected Calibration Error - ECE)
4. OOD Detection (AUROC using predictive uncertainty)

Metric Explanations and Design Decisions:
-----------------------------------------

1. ERROR:
   - Simply the classification error rate: (# misclassified) / (# total)
   - Lower is better
   - For Bayesian models, we average predictions over samples before taking argmax

2. LIKELIHOOD (Negative Log-Likelihood):
   - Measures predictive quality considering uncertainty
   - NLL = -log p(y|x, D) where p(y|x,D) is the predictive distribution
   - For BNNs: p(y|x,D) ≈ (1/S) Σ p(y|x, w_s) where w_s are posterior samples
   - Lower is better
   - This is a proper scoring rule (Gneiting & Raftery, 2007)

3. CALIBRATION (Expected Calibration Error):
   - Measures if predicted probabilities match actual frequencies
   - ECE = Σ (|B_m|/N) |acc(B_m) - conf(B_m)|
   - Where B_m are bins of predictions grouped by confidence
   - Lower is better
   - We use 15 bins as in the paper
   - Evaluated on rotated images to test calibration under distribution shift

4. OOD DETECTION (AUROC):
   - Measures ability to distinguish in-distribution from OOD data
   - We use predictive entropy as the uncertainty measure:
     H[p(y|x)] = -Σ p(y_c|x) log p(y_c|x)
   - Higher entropy → more uncertain → likely OOD
   - AUROC computed treating OOD detection as binary classification
   - Higher is better (but paper plots with reversed y-axis so lower appears better)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class MetricResults:
    """Container for all metric results."""
    error: float
    nll: float  # Negative log-likelihood
    ece: float  # Expected Calibration Error
    ood_auroc: float  # AUROC for OOD detection
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "error": self.error,
            "nll": self.nll,
            "ece": self.ece,
            "ood_auroc": self.ood_auroc
        }


def compute_error(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification error rate.
    
    Args:
        logits: Predicted logits of shape (batch_size, num_classes) or
                (num_samples, batch_size, num_classes) for Bayesian models
        targets: True labels of shape (batch_size,)
        
    Returns:
        Error rate (0 to 1)
    """
    # If we have samples, average the probabilities first
    if logits.ndim == 3:
        # Convert to probabilities, average, then get predictions
        probs = F.softmax(logits, dim=-1)  # (num_samples, batch_size, num_classes)
        mean_probs = probs.mean(dim=0)  # (batch_size, num_classes)
        predictions = mean_probs.argmax(dim=-1)
    else:
        predictions = logits.argmax(dim=-1)
    
    error_rate = (predictions != targets).float().mean().item()
    return error_rate


def compute_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute negative log-likelihood.
    
    For Bayesian models, we compute:
    NLL = -log( (1/S) Σ_s p(y|x, w_s) )
    
    This properly accounts for model uncertainty through averaging
    in probability space (not logit space).
    
    Args:
        logits: Shape (batch_size, num_classes) or (num_samples, batch_size, num_classes)
        targets: Shape (batch_size,)
        
    Returns:
        Average NLL per sample
    """
    if logits.ndim == 3:
        # Bayesian model with samples
        num_samples, batch_size, num_classes = logits.shape
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (S, B, C)
        
        # Use log-sum-exp trick for numerical stability
        # log( (1/S) Σ_s p(y|x, w_s) ) = log(Σ_s exp(log p(y|x, w_s))) - log(S)
        
        # Select the log probabilities of true classes
        targets_expanded = targets.unsqueeze(0).expand(num_samples, -1)  # (S, B)
        log_probs_target = torch.gather(
            log_probs, dim=-1, 
            index=targets_expanded.unsqueeze(-1)
        ).squeeze(-1)  # (S, B)
        
        # Log-sum-exp across samples
        log_mean_prob = torch.logsumexp(log_probs_target, dim=0) - np.log(num_samples)
        
        nll = -log_mean_prob.mean().item()
    else:
        # Standard model
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, targets).item()
    
    return nll


def compute_ece(logits: torch.Tensor, targets: torch.Tensor, 
                num_bins: int = 15) -> float:
    """Compute Expected Calibration Error.
    
    ECE measures the discrepancy between predicted confidence and actual accuracy.
    We bin predictions by confidence and compute weighted average of
    |accuracy - confidence| per bin.
    
    Args:
        logits: Shape (batch_size, num_classes) or (num_samples, batch_size, num_classes)
        targets: Shape (batch_size,)
        num_bins: Number of confidence bins
        
    Returns:
        ECE value (0 to 1, lower is better)
    """
    # Get probabilities
    if logits.ndim == 3:
        probs = F.softmax(logits, dim=-1).mean(dim=0)  # (B, C)
    else:
        probs = F.softmax(logits, dim=-1)
    
    # Get confidence (max probability) and predictions
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()
    
    # Bin boundaries
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    total_samples = len(targets)
    
    for i in range(num_bins):
        # Find samples in this bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().sum() / total_samples
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * torch.abs(avg_accuracy - avg_confidence)
    
    return ece.item()


def compute_predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute predictive entropy as uncertainty measure.
    
    H[p(y|x)] = -Σ_c p(y=c|x) log p(y=c|x)
    
    For Bayesian models, we first average the probabilities over samples,
    then compute entropy of the averaged distribution.
    
    Args:
        logits: Shape (batch_size, num_classes) or (num_samples, batch_size, num_classes)
        
    Returns:
        Entropy for each sample, shape (batch_size,)
    """
    if logits.ndim == 3:
        probs = F.softmax(logits, dim=-1).mean(dim=0)  # (B, C)
    else:
        probs = F.softmax(logits, dim=-1)
    
    # Add small epsilon for numerical stability
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return entropy


def compute_mutual_information(logits: torch.Tensor) -> torch.Tensor:
    """Compute mutual information as epistemic uncertainty measure.
    
    MI[y, w | x, D] = H[E_w[p(y|x,w)]] - E_w[H[p(y|x,w)]]
    
    This captures the uncertainty due to model parameters (epistemic uncertainty).
    Only applicable to Bayesian models with samples.
    
    Args:
        logits: Shape (num_samples, batch_size, num_classes)
        
    Returns:
        Mutual information for each sample, shape (batch_size,)
    """
    if logits.ndim != 3:
        raise ValueError("Mutual information requires samples, got 2D tensor")
    
    # Probabilities per sample
    probs = F.softmax(logits, dim=-1)  # (S, B, C)
    
    # Mean probability
    mean_probs = probs.mean(dim=0)  # (B, C)
    
    # Entropy of mean (total uncertainty)
    H_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
    
    # Mean of entropies (aleatoric uncertainty)
    H_individual = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (S, B)
    mean_H = H_individual.mean(dim=0)  # (B,)
    
    # Mutual information (epistemic uncertainty)
    mi = H_mean - mean_H
    
    return mi


def compute_ood_auroc(in_dist_logits: torch.Tensor, 
                       ood_logits: torch.Tensor,
                       uncertainty_type: str = "entropy") -> float:
    """Compute AUROC for OOD detection.
    
    We use uncertainty (entropy or mutual information) as the score.
    Higher uncertainty → more likely OOD → positive class.
    
    Args:
        in_dist_logits: Logits for in-distribution data
        ood_logits: Logits for OOD data
        uncertainty_type: "entropy" or "mutual_information"
        
    Returns:
        AUROC score (0.5 = random, 1.0 = perfect)
    """
    # Compute uncertainties
    if uncertainty_type == "entropy":
        in_dist_uncertainty = compute_predictive_entropy(in_dist_logits)
        ood_uncertainty = compute_predictive_entropy(ood_logits)
    elif uncertainty_type == "mutual_information":
        in_dist_uncertainty = compute_mutual_information(in_dist_logits)
        ood_uncertainty = compute_mutual_information(ood_logits)
    else:
        raise ValueError(f"Unknown uncertainty_type: {uncertainty_type}")
    
    # Convert to numpy
    in_dist_scores = in_dist_uncertainty.cpu().numpy()
    ood_scores = ood_uncertainty.cpu().numpy()
    
    # Create labels: 0 for in-distribution, 1 for OOD
    labels = np.concatenate([
        np.zeros(len(in_dist_scores)),
        np.ones(len(ood_scores))
    ])
    scores = np.concatenate([in_dist_scores, ood_scores])
    
    # Compute AUROC
    auroc = roc_auc_score(labels, scores)
    
    return auroc


class MetricsComputer:
    """Unified interface for computing all metrics.
    
    This class handles the computation of all four metrics
    and provides utilities for batch evaluation.
    """
    
    def __init__(self, num_bins: int = 15, device: str = "cuda"):
        self.num_bins = num_bins
        self.device = device
    
    @torch.no_grad()
    def compute_all_metrics(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        ood_loader: torch.utils.data.DataLoader,
        num_samples: int = 1,
        is_mcmc: bool = False
    ) -> MetricResults:
        """Compute all four metrics.
        
        Args:
            model: The neural network model
            test_loader: DataLoader for in-distribution test data
            ood_loader: DataLoader for OOD data
            num_samples: Number of samples for Bayesian prediction
            is_mcmc: If True, use MCMC sampling; if False, use MFVI
            
        Returns:
            MetricResults containing all metrics
        """
        model.eval()
        
        # Collect predictions for in-distribution data
        all_logits_in = []
        all_targets = []
        
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            if is_mcmc:
                logits = model(x, use_samples=True)
            elif num_samples > 1:
                logits = model(x, num_samples=num_samples)
            else:
                logits = model(x)
            
            all_logits_in.append(logits)
            all_targets.append(y)
        
        # Concatenate
        if all_logits_in[0].ndim == 3:
            # (num_samples, batch, classes) -> need to concat along batch
            all_logits_in = torch.cat(all_logits_in, dim=1)
        else:
            all_logits_in = torch.cat(all_logits_in, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute in-distribution metrics
        error = compute_error(all_logits_in, all_targets)
        nll = compute_nll(all_logits_in, all_targets)
        ece = compute_ece(all_logits_in, all_targets, self.num_bins)
        
        # Collect predictions for OOD data
        all_logits_ood = []
        
        for x, _ in ood_loader:
            x = x.to(self.device)
            
            if is_mcmc:
                logits = model(x, use_samples=True)
            elif num_samples > 1:
                logits = model(x, num_samples=num_samples)
            else:
                logits = model(x)
            
            all_logits_ood.append(logits)
        
        if all_logits_ood[0].ndim == 3:
            all_logits_ood = torch.cat(all_logits_ood, dim=1)
        else:
            all_logits_ood = torch.cat(all_logits_ood, dim=0)
        
        # Compute OOD AUROC
        ood_auroc = compute_ood_auroc(all_logits_in, all_logits_ood, "entropy")
        
        return MetricResults(
            error=error,
            nll=nll,
            ece=ece,
            ood_auroc=ood_auroc
        )
    
    @torch.no_grad()
    def compute_calibration_metrics(
        self,
        model: torch.nn.Module,
        rotated_loaders: Dict[float, torch.utils.data.DataLoader],
        num_samples: int = 1,
        is_mcmc: bool = False
    ) -> Dict[float, float]:
        """Compute ECE for various rotation angles.
        
        Args:
            model: The neural network model
            rotated_loaders: Dict mapping rotation angle to DataLoader
            num_samples: Number of samples for Bayesian prediction
            is_mcmc: If True, use MCMC sampling
            
        Returns:
            Dict mapping rotation angle to ECE
        """
        model.eval()
        results = {}
        
        for angle, loader in rotated_loaders.items():
            all_logits = []
            all_targets = []
            
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                if is_mcmc:
                    logits = model(x, use_samples=True)
                elif num_samples > 1:
                    logits = model(x, num_samples=num_samples)
                else:
                    logits = model(x)
                
                all_logits.append(logits)
                all_targets.append(y)
            
            if all_logits[0].ndim == 3:
                all_logits = torch.cat(all_logits, dim=1)
            else:
                all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            results[angle] = compute_ece(all_logits, all_targets, self.num_bins)
        
        return results


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_compute_error():
    """Test error computation."""
    # Deterministic case
    logits = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]  # This will be wrong if target is 0
    ])
    targets = torch.tensor([0, 1, 2, 0])
    
    error = compute_error(logits, targets)
    assert abs(error - 0.25) < 1e-6, f"Expected 0.25, got {error}"
    
    # Bayesian case (3 samples)
    logits_3d = torch.stack([logits, logits, logits])
    error_3d = compute_error(logits_3d, targets)
    assert abs(error_3d - 0.25) < 1e-6
    
    print("✓ compute_error test passed")


def test_compute_nll():
    """Test NLL computation."""
    # Perfect predictions should have low NLL
    logits_perfect = torch.tensor([
        [10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0]
    ])
    targets = torch.tensor([0, 1])
    
    nll_perfect = compute_nll(logits_perfect, targets)
    assert nll_perfect < 0.01, f"Perfect predictions should have low NLL, got {nll_perfect}"
    
    # Random predictions should have higher NLL
    logits_random = torch.zeros(2, 3)
    nll_random = compute_nll(logits_random, targets)
    assert nll_random > nll_perfect
    
    # Bayesian case
    logits_3d = torch.stack([logits_perfect, logits_perfect])
    nll_3d = compute_nll(logits_3d, targets)
    assert abs(nll_3d - nll_perfect) < 0.01
    
    print("✓ compute_nll test passed")


def test_compute_ece():
    """Test ECE computation."""
    # Perfect calibration: confidence matches accuracy
    # All predictions correct with confidence 0.8
    logits = torch.log(torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ]))
    targets = torch.tensor([0, 1, 2])
    
    ece = compute_ece(logits, targets, num_bins=10)
    # ECE should be |1.0 - 0.8| * 1.0 = 0.2 (all correct, conf=0.8)
    assert abs(ece - 0.2) < 0.05, f"Expected ECE near 0.2, got {ece}"
    
    print("✓ compute_ece test passed")


def test_compute_predictive_entropy():
    """Test predictive entropy computation."""
    # Confident prediction (low entropy)
    logits_confident = torch.tensor([[10.0, -10.0, -10.0]])
    entropy_confident = compute_predictive_entropy(logits_confident)
    
    # Uniform prediction (high entropy)
    logits_uniform = torch.tensor([[0.0, 0.0, 0.0]])
    entropy_uniform = compute_predictive_entropy(logits_uniform)
    
    assert entropy_confident < entropy_uniform, \
        "Confident predictions should have lower entropy"
    
    # Check entropy bounds
    assert entropy_confident >= 0
    assert entropy_uniform <= np.log(3) + 0.01  # Max entropy for 3 classes
    
    print("✓ compute_predictive_entropy test passed")


def test_compute_mutual_information():
    """Test mutual information computation."""
    # Samples with same predictions -> MI = 0
    logits_same = torch.tensor([
        [[10.0, -10.0], [-10.0, 10.0]],
        [[10.0, -10.0], [-10.0, 10.0]],
    ])
    mi_same = compute_mutual_information(logits_same)
    assert (mi_same < 0.01).all(), "Same samples should have near-zero MI"
    
    # Samples with different predictions -> higher MI
    logits_diff = torch.tensor([
        [[10.0, -10.0], [-10.0, 10.0]],
        [[-10.0, 10.0], [10.0, -10.0]],
    ])
    mi_diff = compute_mutual_information(logits_diff)
    assert (mi_diff > mi_same).all(), "Different samples should have higher MI"
    
    print("✓ compute_mutual_information test passed")


def test_compute_ood_auroc():
    """Test OOD AUROC computation."""
    # In-distribution: confident predictions
    in_dist_logits = torch.tensor([
        [10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, -10.0, 10.0],
    ])
    
    # OOD: uncertain predictions
    ood_logits = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, -0.1],
        [0.0, 0.1, -0.1],
    ])
    
    auroc = compute_ood_auroc(in_dist_logits, ood_logits, "entropy")
    assert auroc > 0.8, f"Should be able to distinguish, got AUROC={auroc}"
    
    print("✓ compute_ood_auroc test passed")


def test_metric_results():
    """Test MetricResults dataclass."""
    results = MetricResults(
        error=0.1,
        nll=0.5,
        ece=0.05,
        ood_auroc=0.95
    )
    
    d = results.to_dict()
    assert d["error"] == 0.1
    assert d["nll"] == 0.5
    assert d["ece"] == 0.05
    assert d["ood_auroc"] == 0.95
    
    print("✓ MetricResults test passed")


if __name__ == "__main__":
    test_compute_error()
    test_compute_nll()
    test_compute_ece()
    test_compute_predictive_entropy()
    test_compute_mutual_information()
    test_compute_ood_auroc()
    test_metric_results()
    print("\nAll metrics tests passed!")
