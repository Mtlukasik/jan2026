# SVGD vs MFVI Last Layer Bayesian Inference Comparison

This project implements a comparison of **Stein Variational** and **MFVI (Mean-Field Variational Inference)** for last-layer Bayesian inference on CIFAR-10, following the methodology in:

> **"Bayesian Neural Network Priors Revisited"**  
> Vincent Fortuin*, Adrià Garriga-Alonso*, et al.  
> ICLR 2022

## Project Structure

```
bnn_comparison/
├── config.py              # Configuration classes and hyperparameters
├── data_loading.py        # Data loading utilities (CIFAR-10, SVHN)
├── networks.py            # Neural network architectures
├── metrics.py             # Evaluation metrics (Error, NLL, ECE, AUROC)
├── mcmc_inference.py      # MCMC/SG-MCMC implementation
├── mfvi_inference.py      # Mean-Field VI implementation
├── experiment_runner.py   # Main experiment orchestration
├── visualization.py       # Plotting utilities
├── main.py                # Entry point script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
python main.py --mode test

# Generate demo plots
python main.py --mode demo

# Run quick experiment (for testing)
python main.py --mode quick

# Run full experiment
python main.py --mode full --replicates 5

# Print experiment explanation
python main.py --mode explain
```

## Experiment Design

### Network Architecture
- **Input**: Flattened CIFAR-10 images (32×32×3 = 3072 dimensions)
- **Layer 1**: Linear(3072 → 512) + BatchNorm + ReLU + Dropout(0.2)
- **Layer 2**: Linear(512 → 10) ← *Bayesian inference applied here*

We use a simple 2-layer FCNN (no convolutions) to isolate the effect of Bayesian inference methods.

### Methods Compared

#### MCMC (Stochastic Gradient MCMC)
- Uses Stochastic Gradient Langevin Dynamics (SGLD)
- Cyclical learning rate schedule for better exploration
- Temperature applied to posterior: `p(w|D)^(1/T)`

#### MFVI (Mean-Field Variational Inference)
- Approximates posterior with factorized Gaussian: `q(w) = Π N(w_i | μ_i, σ_i²)`
- Optimizes ELBO with reparameterization trick
- Temperature (λ) scales the KL term: `E_q[log p(y|x,w)] - λ * KL(q||p)`

### Temperature / Cold Posterior Effect
- **T = 1**: Standard Bayesian posterior
- **T < 1**: "Cold" posterior (more concentrated, often better)
- **T > 1**: "Warm" posterior (more spread, usually worse)

Temperatures tested: `[0.001, 0.01, 0.03, 0.1, 0.3, 1.0]`

## Metrics

### 1. Error (Classification Error Rate)
```
Error = (# wrong predictions) / (# total predictions)
```
- Lower is better
- For Bayesian models: average probabilities over samples, then take argmax

### 2. NLL (Negative Log-Likelihood)
```
NLL = -E[log p(y|x, D)]
```
- For Bayesian models: `p(y|x,D) ≈ (1/S) Σ_s p(y|x, w_s)`
- Lower is better
- Proper scoring rule that accounts for uncertainty

### 3. ECE (Expected Calibration Error)
```
ECE = Σ_m (|B_m|/N) * |acc(B_m) - conf(B_m)|
```
- Measures if predicted confidence matches actual accuracy
- Lower is better (0 = perfectly calibrated)
- Computed with 15 bins

### 4. OOD AUROC (Out-of-Distribution Detection)
```
Score = H[p(y|x)] = -Σ_c p(y=c|x) log p(y=c|x)  (predictive entropy)
```
- Higher entropy → more uncertain → likely OOD
- AUROC for distinguishing CIFAR-10 (in-dist) vs SVHN (OOD)
- Higher is better

## Data

### In-Distribution: CIFAR-10
- 50,000 training images, 10,000 test images
- 32×32×3 color images
- 10 object classes

### Out-of-Distribution: SVHN
- Street View House Numbers
- Same dimensions (32×32×3)
- Semantically different domain (digits vs objects)
- Standard OOD benchmark for CIFAR-10

## Expected Results

Based on the paper's findings:

1. **MFVI vs MCMC**: MFVI generally performs worse, especially at T=1
2. **Cold Posterior Effect**: Lower temperatures often improve Error and NLL
3. **Calibration**: May not follow the same pattern as accuracy metrics
4. **OOD Detection**: Behavior varies; not always improved by cold posteriors

## Output

Results are saved to `./results/`:
- `raw_results.json`: All experiment data
- `figures/comparison.png`: Main comparison figure

The comparison figure follows the paper's style:
- 4 panels: Error, NLL, ECE, OOD AUROC
- X-axis: Temperature (log scale)
- Lines with shaded error regions (standard error)
- OOD AUROC y-axis inverted (so lower = better visually for all)

## References

```bibtex
@inproceedings{fortuin2022bayesian,
  title={Bayesian Neural Network Priors Revisited},
  author={Fortuin, Vincent and Garriga-Alonso, Adri{\`a} and Ober, Sebastian W and Wenzel, Florian and R{\"a}tsch, Gunnar and Turner, Richard E and van der Wilk, Mark and Aitchison, Laurence},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## License

MIT License
