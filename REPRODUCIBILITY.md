# Reproducibility Protocol

## 1. Scope

The repository operationalizes the computational claims of the associated shared-warehouse digital-twin study. It is designed to reproduce the environment, training logic, baselines, ablations, robustness tests, and statistical analysis from configuration-controlled experiments.

## 2. Manuscript-specified settings

The implementation directly encodes the reported warehouse scale: 18,000 m² floor area, 12 storage zones, 48 aisles, 9,600 storage locations, six tenants, four-to-six rack levels, 180 operational days, five-minute decision epochs, one-second twin-state timestamps, non-stationary Poisson demand, and dynamic aisle-level congestion.

## 3. Explicit reproducibility defaults

The manuscript does not provide numerical values for every reward coefficient, demand-cycle amplitude, burst probability, congestion coefficient, PPO clipping parameter, GAE coefficient, number of training episodes, demand-shock magnitude, layout-perturbation magnitude, or twin-delay/noise magnitude. These values are therefore declared explicitly in `config.yaml` as **repository defaults**. They are not represented as hidden or originally reported manuscript values.

## 4. Algorithmic consistency

The proposed method is implemented as on-policy MAPPO with decentralized masked actors and a centralized state-value critic. PPO clipping, generalized advantage estimation, entropy regularization, value loss, advantage normalization, gradient clipping, and deterministic seed control are included.

Replay memory and target networks are **not** used by MAPPO. They are implemented only for the QMIX baseline, where such off-policy machinery is algorithmically appropriate. This resolves the manuscript's mixed description of MAPPO with replay-buffer/target-network terminology.

## 5. Decision model

Six tenant agents are used as the concrete decision entities. Each agent receives a tenant-local observation derived from zone utilization, zone-level congestion summaries, own occupancy share, demand intensity, and fairness state. The centralized critic receives the complete digital-twin state.

The physical warehouse contains 9,600 storage locations. To keep learning tractable without discarding location-level feasibility, the environment dynamically filters compatible available locations and exposes up to 32 ranked candidate locations plus a no-op action. The physical slot ID selected by each categorical action remains explicit and is executed by the twin.

## 6. Digital-twin timing

Operational events carry second-resolution timestamps, matching the stated twin synchronization resolution. Allocation actions are issued every 300 seconds, matching the stated decision interval. The simulator is event-driven between decision epochs rather than performing unnecessary neural inference every second.

## 7. Baselines

1. Static class/proximity slotting.
2. Rolling-horizon dynamic heuristic using utilization and live congestion.
3. Centralized single-agent PPO with global-state access for all allocation heads.
4. QMIX with decentralized Q-networks, monotonic mixing, replay memory, target networks, and epsilon-greedy exploration.
5. Proposed DT-MAPPO.

## 8. Ablations

- Full DT-MAPPO.
- Without dynamic twin feedback: congestion state is not dynamically updated.
- Without MARL: centralized PPO replaces the distributed tenant agents.
- Without fairness reward: the fairness term is removed from the objective.

## 9. Robustness scenarios

Configuration-controlled scenarios include normal operation, demand shock, severe demand shock, layout perturbation by blocked aisles, and delayed/noisy digital-twin observations. Exact magnitudes are recorded in `config.yaml`.

## 10. Statistical protocol

Five fixed seeds are defined by default. Evaluation is performed using matched method/scenario/seed/episode tuples. The repository computes mean, standard deviation, 95% t-based confidence intervals, paired significance tests, a paired standardized effect size, and Holm correction for multiple comparisons. Paired t-tests are selected when the paired difference passes the configured normality criterion; otherwise Wilcoxon signed-rank tests are used.

## 11. Commands

Generate reference data:

```bash
python generate_dataset.py
```

Validate the repository:

```bash
python validate_repository.py
```

Train the proposed method:

```bash
python train.py --method mappo --seed 11
```

Train baselines:

```bash
python train.py --method single_ppo --seed 11
python train.py --method qmix --seed 11
```

Evaluate:

```bash
python evaluate.py --method mappo --seed 11 --scenario normal
```

Full five-seed reproduction:

```bash
python reproduce_all.py
```

Low-cost integrity/smoke reproduction:

```bash
python reproduce_all.py --quick
```

## 12. Output provenance

Training logs, checkpoints, and result CSVs are written into the existing `data/` folder to preserve the requested two-folder repository structure. Generated numerical outputs should be used to replace unresolved manuscript placeholders; numbers should not be copied into the paper before successful execution.
