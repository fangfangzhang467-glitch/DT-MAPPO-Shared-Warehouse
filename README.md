# DT-MAPPO Shared-Warehouse Reproducibility Repository

Reproducibility implementation for the study **“Multi-Agent Deep Reinforcement Learning–Based Storage Location Optimization in Digital Twin–Enabled Shared Warehouses.”**

This repository implements the computational pipeline needed to reproduce the digital-twin warehouse environment, proposed MAPPO policy, comparison baselines, ablation experiments, robustness tests, and statistical analyses described in the manuscript.

## Why this repository exists

The study reports a custom computational pipeline rather than a purely analytical model. The repository therefore exposes the underlying environment and learning code in a form that can be independently executed, inspected, and archived in a DOI-assigning repository when the manuscript revision is finalized.

No DOI is claimed here because the manuscript is still under review. No license file is included.

## Repository structure

```text
DT-MAPPO-Shared-Warehouse/
├── README.md
├── REPRODUCIBILITY.md
├── CODE_AVAILABILITY.md
├── requirements.txt
├── config.yaml
├── generate_dataset.py
├── train.py
├── evaluate.py
├── reproduce_all.py
├── validate_repository.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── warehouse_twin.py
│   ├── demand_model.py
│   ├── congestion_model.py
│   ├── marl_env.py
│   ├── networks.py
│   ├── mappo.py
│   ├── qmix.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── statistics.py
│   └── utils.py
└── data/
    ├── README.md
    ├── warehouse_layout.csv
    ├── tenant_profiles.csv
    ├── product_compatibility.csv
    ├── scenario_manifest.yaml
    ├── reference_events.csv.gz
    └── checksums.sha256
```

Only two main folders are used: `src/` and `data/`.

## Model represented by the code

The warehouse digital twin uses the manuscript scale:

- Floor area: 18,000 m²
- Storage zones: 12
- Aisles: 48
- Physical storage locations: 9,600
- Tenants: 6
- Rack levels: 4–6
- Planning horizon represented by configuration: 180 operational days
- Storage-decision interval: 300 s
- Digital-twin timestamp resolution: 1 s
- Demand: tenant-specific non-stationary Poisson process
- Congestion: dynamic aisle-level state with persistence and spillover

The executable MARL environment represents the six tenants as decentralized agents. Each agent selects from dynamically generated compatible location candidates. The digital twin always executes the selected action against the original physical slot ID, so the candidate interface does not replace the 9,600-location warehouse model.

## Proposed DT-MAPPO implementation

The proposed controller follows centralized training and decentralized execution:

- separate decentralized stochastic actors;
- centralized critic using the complete digital-twin state;
- masked categorical actions;
- PPO clipped surrogate objective;
- generalized advantage estimation;
- entropy regularization;
- normalized advantages;
- value clipping;
- gradient clipping;
- deterministic random-seed control.

### Important algorithmic clarification

The manuscript text contains replay-buffer, target-network, and q-network terminology in the MAPPO subsection. Those mechanisms are inconsistent with an on-policy PPO/MAPPO implementation. The repository therefore uses a coherent on-policy MAPPO implementation and reserves replay memory and target networks for the QMIX baseline.

## Implemented comparison methods

| Method | Implementation |
|---|---|
| Static class-based slotting | Feasible location selected by static physical proximity |
| Rolling-horizon heuristic | Dynamic score using distance, zone utilization, demand, and live congestion |
| Single-agent PPO | Centralized global-state controller using the same warehouse environment |
| QMIX | Decentralized Q-networks + monotonic mixer + replay + target networks |
| DT-MAPPO | Proposed centralized-training/decentralized-execution policy |

All methods are evaluated through the same digital-twin transition and metric code.

## Reward and measured outcomes

The reward includes retrieval efficiency, congestion, tenant fairness, invalid-action penalties, and throughput benefit. Exact coefficients are defined in `config.yaml`.

Evaluation records:

- average retrieval time;
- travel distance;
- throughput;
- mean congestion;
- congestion variance;
- congestion stability;
- tenant fairness;
- cumulative reward;
- composite performance index.

## Reproducibility defaults

Several exact numerical values are not stated in the manuscript, including reward weights, burst probabilities, PPO clipping settings, GAE coefficient, training-episode count, perturbation magnitudes, and observation-delay/noise levels. These are not silently inferred in the source files. They are declared in `config.yaml` and labeled as reproducibility defaults.

This makes every assumption inspectable and replaceable during revision.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate deterministic reference data

```bash
python generate_dataset.py
```

The command generates the 9,600-location layout, six-tenant profile table, compatibility table, scenario manifest, deterministic reference event stream, and SHA-256 checksums.

## Validate before training

```bash
python validate_repository.py
```

Validation checks repository completeness, data checksums, state/action dimensions, deterministic reset behavior, action-mask validity, fairness bounds, and congestion bounds.

## Train DT-MAPPO

```bash
python train.py --method mappo --seed 11
```

The default five seeds are:

```text
11, 23, 37, 51, 79
```

To run another seed:

```bash
python train.py --method mappo --seed 23
```

## Train learned baselines

```bash
python train.py --method single_ppo --seed 11
python train.py --method qmix --seed 11
```

## Evaluate heuristic baselines

No training is required:

```bash
python evaluate.py --method static --seed 11 --scenario normal
python evaluate.py --method rolling --seed 11 --scenario normal
```

## Evaluate the proposed model

```bash
python evaluate.py --method mappo --seed 11 --scenario normal
```

Robustness conditions:

```bash
python evaluate.py --method mappo --seed 11 --scenario demand_shock
python evaluate.py --method mappo --seed 11 --scenario severe_demand_shock
python evaluate.py --method mappo --seed 11 --scenario layout_perturbation
python evaluate.py --method mappo --seed 11 --scenario delayed_twin
```

## One-command reproduction

Full configured experiment:

```bash
python reproduce_all.py
```

Fast integrity check:

```bash
python reproduce_all.py --quick
```

The quick mode is not intended for manuscript performance reporting. It exists only to confirm that data generation, training, evaluation, checkpoint loading, ablation execution, and statistics complete end-to-end.

## Ablation experiments

The repository evaluates:

1. full DT-MAPPO;
2. MAPPO without dynamic twin feedback;
3. centralized PPO instead of MARL;
4. MAPPO without the fairness reward term.

These tests replace qualitative labels such as “Best” or “Worse” with numerical outputs generated by identical evaluation code.

## Robustness experiments

The configuration includes:

- nominal demand;
- 1.75× demand shock;
- 2.25× severe demand shock;
- 15% blocked-aisle layout perturbation;
- delayed and noisy twin observation.

These magnitudes are repository defaults because exact disturbance magnitudes were not provided in the manuscript.

## Statistical evaluation

The pipeline aggregates independent seeds and reports:

- mean;
- standard deviation;
- 95% confidence interval;
- paired t-test when paired differences satisfy the configured normality criterion;
- Wilcoxon signed-rank alternative otherwise;
- paired standardized effect size;
- Holm-adjusted p-values for multiple comparisons.

The comparison is paired by seed and evaluation episode so competing policies face matched experimental conditions.

## Generated outputs

Runtime outputs remain in the existing `data/` folder to preserve the two-folder repository structure. Important generated files include:

```text
data/training_mappo_seed11.csv
data/training_single_ppo_seed11.csv
data/training_qmix_seed11.csv
data/model_mappo_seed11.pt
data/model_single_ppo_seed11.pt
data/model_qmix_seed11.pt
data/results_main_comparison.csv
data/results_main_summary.csv
data/results_ablation.csv
data/results_robustness.csv
data/results_statistical_tests.csv
```

Generated checkpoints and result files are intentionally excluded from Git tracking by `.gitignore`. The immutable release archived for peer review may include selected final result tables/checkpoints if journal policy requires them.

## Manuscript-result mapping

| Manuscript claim | Reproducibility source |
|---|---|
| Shared warehouse digital twin | `src/warehouse_twin.py` |
| Non-stationary tenant demand | `src/demand_model.py` |
| Aisle congestion dynamics | `src/congestion_model.py` |
| Agent observation/action interface | `src/marl_env.py` |
| MAPPO policy and critic | `src/mappo.py`, `src/networks.py` |
| QMIX baseline | `src/qmix.py` |
| Static/rolling heuristics | `src/baselines.py` |
| Retrieval/congestion/fairness metrics | `src/metrics.py` |
| Confidence intervals and significance | `src/statistics.py` |
| Dataset characteristics | `generate_dataset.py`, `data/*.csv` |
| Ablation table | `data/results_ablation.csv` after reproduction |
| Robustness statements | `data/results_robustness.csv` after reproduction |
| Main baseline comparison | `data/results_main_comparison.csv` after reproduction |

## Scientific-use note

The scripts intentionally do not hard-code performance numbers from the manuscript. Reported values should come from completed runs. If a manuscript placeholder or qualitative result is replaced during revision, the corresponding number should be copied from a version-controlled or archived result file produced by this repository.

## DOI deposition workflow

When the revision is ready for archival:

1. create the final GitHub release/commit used for reported experiments;
2. archive that exact release in a recognized DOI-assigning repository such as Zenodo;
3. obtain the DOI from the archive;
4. add the DOI to the manuscript's Methods or Code Availability section;
5. do not alter the archived code after using its DOI as the reproducibility citation.
