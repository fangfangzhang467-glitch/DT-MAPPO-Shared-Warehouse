from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from evaluate import evaluate
from src.statistics import aggregate_seed_results, holm_adjust, paired_test
from src.utils import load_config, project_root


def run(cmd):
    print("$", " ".join(map(str,cmd)))
    subprocess.run(cmd,check=True,cwd=project_root())


def main():
    p=argparse.ArgumentParser(description="One-command reproduction pipeline for the manuscript experiments.")
    p.add_argument("--quick",action="store_true",help="Smoke reproduction: 1 seed, 2 train episodes, 2 evaluation episodes.")
    p.add_argument("--skip-training",action="store_true")
    args=p.parse_args(); cfg=load_config(); root=project_root()
    seeds=[cfg["project"]["seed_set"][0]] if args.quick else list(cfg["project"]["seed_set"])
    train_eps=2 if args.quick else int(cfg["training"]["episodes"])
    eval_eps=2 if args.quick else int(cfg["training"]["evaluation_episodes"])

    run([sys.executable,"generate_dataset.py"])
    learned=["single_ppo","qmix","mappo"]
    if not args.skip_training:
        for seed in seeds:
            for method in learned:
                run([sys.executable,"train.py","--method",method,"--seed",str(seed),"--episodes",str(train_eps)])

    frames=[]
    for seed in seeds:
        for method in ["static","rolling","single_ppo","qmix","mappo"]:
            frames.append(evaluate(method,seed,"normal",eval_eps))
    main_df=pd.concat(frames,ignore_index=True)
    main_df.to_csv(root/"data"/"results_main_comparison.csv",index=False)
    aggregate_seed_results(main_df).to_csv(root/"data"/"results_main_summary.csv",index=False)

    # Robustness: proposed model under operational disturbances.
    robustness=[]
    for seed in seeds:
        for scenario in ["normal","demand_shock","severe_demand_shock","layout_perturbation","delayed_twin"]:
            robustness.append(evaluate("mappo",seed,scenario,eval_eps))
    rob_df=pd.concat(robustness,ignore_index=True)
    rob_df.to_csv(root/"data"/"results_robustness.csv",index=False)

    # Ablations. without_marl is represented by the separately trained centralized PPO baseline.
    ablation=[]
    for seed in seeds:
        ablation.append(evaluate("mappo",seed,"normal",eval_eps,"full_model"))
        ablation.append(evaluate("mappo",seed,"normal",eval_eps,"without_dynamic_twin"))
        ablation.append(evaluate("mappo",seed,"normal",eval_eps,"without_fairness_reward"))
        ablation.append(evaluate("single_ppo",seed,"normal",eval_eps,"without_marl"))
    abl_df=pd.concat(ablation,ignore_index=True)
    abl_df.to_csv(root/"data"/"results_ablation.csv",index=False)

    # Paired statistical comparisons against DT-MAPPO using identical seeds/episodes.
    stats_rows=[]
    metrics=["average_retrieval_time","congestion_variance","congestion_stability","fairness","throughput","composite_index"]
    proposed=main_df[main_df.method=="mappo"].sort_values(["seed","episode"])
    for baseline in ["static","rolling","single_ppo","qmix"]:
        base=main_df[main_df.method==baseline].sort_values(["seed","episode"])
        for metric in metrics:
            t=paired_test(proposed[metric].to_numpy(),base[metric].to_numpy(),alpha=float(cfg["statistics"]["alpha"]))
            stats_rows.append({"baseline":baseline,"metric":metric,**t})
    stats_df=pd.DataFrame(stats_rows)
    stats_df["p_holm"]=holm_adjust(stats_df["p_value"].to_numpy())
    stats_df.to_csv(root/"data"/"results_statistical_tests.csv",index=False)

    print("\nReproduction outputs written to data/results_*.csv")

if __name__=="__main__": main()
