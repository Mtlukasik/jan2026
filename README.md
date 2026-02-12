# SVGD vs MFVI Deep learning Last Layer Bayesian Inference Comparison

This project implements a comparison of **Stein Variational** and **MFVI (Mean-Field Variational Inference)** for last-layer Bayesian inference on CIFAR-10, following the methodology in:

> **"Bayesian Neural Network Priors Revisited"**  
> Vincent Fortuin*, Adrià Garriga-Alonso*, et al.  
> ICLR 2022

pipeline_resnet.py is a CLI “orchestrator” for a 3-stage experiment. Each stage is implemented as a trainer class:

Step 1 → DeterministicResNetTrainer

Step 2 → BayesianLastLayerTrainer

Step 3 → JointTrainer

Plus status reporting → print_status(...)

It uses these external modules (must exist in your project):

config.py → defines ExperimentConfig, DEVICE, SEED

data_loading_resnet.py → defines DataLoaderManager

resnet.py → defines ResNetForBayesianLastLayer, ResNetFeatureExtractor

optionally model_variants.py → variant registry/helpers

1) Directory layout and “done markers”

Everything is written under --save-dir (default ./results). Completion is tracked by a sentinel file named COMPLETED inside each run folder.

Step 1 output folder
{save_dir}/deterministic_model/
  model_weights.pt
  training_history.json
  metrics.json
  training_curves.png
  COMPLETED

Step 2 output folders (many runs)

Each run is a separate directory.

Default naming patterns:

SVGD: last_layer_svgd_{prior}_T{temp}_replicate_{k}

MFVI: last_layer_mfvi_T{temp}_replicate_{k}

Each run folder contains:

{save_dir}/last_layer_.../
  results.json
  hyperparameters.json
  training_curves.png
  COMPLETED
  particles.pt            (SVGD only)
  mfvi_layer.pt           (MFVI only)

Step 3 output folders (many runs)

Default naming:

joint_{method}_T{temp}_replicate_{k}

Each run folder contains:

{save_dir}/joint_.../
  results.json
  hyperparameters.json
  training_curves.png
  COMPLETED
  particles.pt            (SVGD only)
  mfvi_layer.pt           (MFVI only)
  feature_extractor.pt    (both methods)


Important operational detail: reruns are prevented by checking for COMPLETED. Use --force to remove that marker (only) and rerun.

2) Inputs, configuration, reproducibility
Global config object

At runtime, config = ExperimentConfig() is created once in main(). This is the single source of truth for:

dataset/batch size split settings (config.data, etc.)

temperatures list: config.temperature.temperatures

number of replicates: config.num_replicates

per-method hyperparameters (config.svgd, config.mfvi)

Seeds

The code sets seeds in main():

torch.manual_seed(SEED)

np.random.seed(SEED)

if CUDA: torch.cuda.manual_seed_all(SEED)

Then Step 2/3 add replicate-specific offsets:

Step 2: SEED + replicate

Step 3: SEED + replicate + 1000

This means runs are deterministic given identical environment + deterministic ops and stable data splits.

Data split persistence

DataLoaderManager(..., save_dir=save_dir) is called in steps 1–3. The intent is:

Step 1 creates/saves split metadata under save_dir (implementation in DataLoaderManager)

Steps 2 and 3 reuse the same split

So don’t change --save-dir between steps if you want exact same split.

3) Step 1: deterministic model training (what the code does)
Entry point
python pipeline_resnet.py --step 1 --save-dir ./results

Behavior summary

Creates {save_dir}/deterministic_model/

Builds augmented train loader (RandomCrop + flip) via DataLoaderManager

Instantiates ResNetForBayesianLastLayer(... last_layer_type="deterministic")

Trains for --epochs (default 200) with:

SGD(momentum=0.9, weight_decay=5e-4)

CosineAnnealingLR(T_max=num_epochs)

Checkpointing / best model selection

Every val_frequency=5 epochs it runs _evaluate() on val set.

Tracks best_val_acc and saves model_weights.pt when validation accuracy improves.

At the end, reloads best weights and computes final metrics on test set and OOD set.

Skip logic

If {save_dir}/deterministic_model/COMPLETED exists:

Step 1 prints [SKIP] ... and loads metrics.json instead of training.

4) Step 2: last-layer-only runs (what gets executed)
Entry point (run everything pending)
python pipeline_resnet.py --step 2 --save-dir ./results

Preconditions

Requires {save_dir}/deterministic_model/COMPLETED

Loads {save_dir}/deterministic_model/model_weights.pt into memory (self.pretrained_weights)

Run enumeration

train_all() computes “pending” runs by scanning {save_dir} folders for last_layer_* with COMPLETED.

It builds a run grid:

methods: default ["svgd","mfvi"]

temperatures: config.temperature.temperatures

replicates: config.num_replicates

SVGD only: also iterates config.svgd.prior_types

MFVI ignores prior_type dimension

So total count =

SVGD: len(prior_types) * len(temps) * reps

MFVI: len(temps) * reps

What each run does at a high level

Both methods follow the same structural pattern:

Make a run directory.

Seed RNGs based on replicate.

Load feature extractor weights from Step 1.

Freeze or unfreeze features depending on step (Step 2 freezes).

Train the last layer parameters according to the method.

Periodically evaluate on val loader.

At end compute metrics on test + ood loaders.

Save:

artifacts (particles.pt or mfvi_layer.pt)

results.json, hyperparameters.json, training_curves.png

COMPLETED

Running a single Step 2 run by name

You can run just one model using --name:

Examples:

python pipeline_resnet.py --step 2 --save-dir ./results \
  --name last_layer_svgd_laplace_T0.1_replicate_1

python pipeline_resnet.py --step 2 --save-dir ./results \
  --name last_layer_mfvi_T0.1_replicate_2


Note: The code supports multiple naming conventions via parse_run_name(), including “variant-like” names (e.g. last_layer_svgd_laplace_std1_T0.1_replicate_1).

--force

If the folder has COMPLETED, the run is blocked unless:

you pass --force, which deletes only the COMPLETED file and reruns

5) Step 3: joint training runs (optional) and the gotcha
Entry point (run everything pending)
python pipeline_resnet.py --step 3 --save-dir ./results

Preconditions

Requires Step 1 COMPLETED

Requires the corresponding Step 2 run COMPLETED (per method/temp/replicate)

How Step 3 decides what to run

It scans for:

Step 2 availability: last_layer_{method}_T{temp}_replicate_{k}/COMPLETED

Step 3 completion: joint_{method}_T{temp}_replicate_{k}/COMPLETED

It only trains Step 3 configurations where Step 2 exists and Step 3 doesn’t.

Running a single Step 3 run by name
python pipeline_resnet.py --step 3 --save-dir ./results \
  --name joint_svgd_T0.1_replicate_1

Operational gotcha: Step 3 SVGD expects a Step 2 folder name that may not exist

In your Step 2 SVGD naming, you introduced:

last_layer_svgd_{prior}_T{temp}_replicate_{k} (note {prior})

But Step 3 SVGD hardcodes:

step2_name = f"last_layer_svgd_T{temperature}_replicate_{replicate + 1}"


(no prior segment)

So if you actually trained Step 2 SVGD under the new scheme (with _laplace_/_gaussian_ in the name), Step 3 SVGD will fail to find the Step 2 directory.

Meaning (software operation):

Step 3 MFVI likely works as-is.

Step 3 SVGD likely breaks unless you also have old-format SVGD folders without prior in the name, or you modify Step 3 to include prior/variant in the directory lookup.

If you want Step 3 to work with the new Step 2 SVGD naming, you need to:

pass/encode prior info into the Step 3 run name and lookup
or

store a pointer in Step 2 results.json and read it
or

decide a canonical Step 2 SVGD folder naming and use it consistently in both steps.

6) Status command (what it checks)
python pipeline_resnet.py --status --save-dir ./results


It prints:

Step 1 done/not done

Step 2: counts completed among expected grid but currently counts only old naming:

name = f"last_layer_{method}_T{t}_replicate_{r+1}"


This ignores SVGD prior dimension and ignores the new SVGD naming template with {prior}.

So with your new naming, --status may undercount SVGD Step 2 completion.

7) How to operate it reliably (recommended workflow)
A) First time run
# step 1
python pipeline_resnet.py --step 1 --save-dir ./results

# step 2 (all)
python pipeline_resnet.py --step 2 --save-dir ./results

# optional: step 3
python pipeline_resnet.py --step 3 --save-dir ./results

B) Run one configuration only
python pipeline_resnet.py --step 2 --save-dir ./results \
  --name last_layer_mfvi_T0.1_replicate_1

C) Rerun a failed/modified run
python pipeline_resnet.py --step 2 --save-dir ./results \
  --name last_layer_mfvi_T0.1_replicate_1 --force

D) Verify outputs

Existence of COMPLETED is your “done” flag.

results.json is the summary artifact per run.

hyperparameters.json records config snapshot used for the run.

8) What to look at when debugging
Common failure points

Step ordering: Step 2 requires Step 1; Step 3 requires matching Step 2.

Naming mismatch (major): Step 3 SVGD and --status logic assume old naming.

Device mismatch: weights loaded with map_location=self.device but check DEVICE in config.py.

Run discovery: completed runs are discovered by scanning folders and checking COMPLETED. If you delete only weights but leave COMPLETED, the run will be considered done.

Data split mismatch: using a different --save-dir effectively creates a new split universe.

Quick “is it done?” checks

ls ./results/deterministic_model/COMPLETED

find ./results -maxdepth 2 -name COMPLETED

9) Practical checklist before you run big sweeps

Ensure Step 2 naming and Step 3 lookup are consistent (especially SVGD + priors/variants).

Ensure config.temperature.temperatures, config.num_replicates, and config.svgd.prior_types match the experiment you want.

Decide whether you use --variants mode. If yes:

python pipeline_resnet.py --list-variants

python pipeline_resnet.py --step 2 --variants ...

Use one stable --save-dir for the whole experiment.

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
- Proper scoring rule that accounts for epistemic uncertainty

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
