from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import centralized_observation
from src.mappo import MAPPO, RolloutBatch, collect_rollout
from src.marl_env import SharedWarehouseMAEnv
from src.metrics import summarize_episode
from src.qmix import QMIX, Transition
from src.utils import load_config, project_root, resolve_device, set_global_seed


def train_mappo(cfg, seed: int, episodes: int, method: str, scenario: str, ablation: str):
    env = SharedWarehouseMAEnv(cfg, seed, scenario=scenario, ablation=ablation)
    device = resolve_device(cfg["project"].get("device", "auto"))
    central = method == "single_ppo"
    obs_dim = env.global_state_dim if central else env.local_obs_dim
    model_cfg = cfg["single_agent_ppo"] if central else cfg["mappo"]
    # Fill common MAPPO fields for the centralized baseline.
    if central:
        base = cfg["mappo"].copy(); base.update(model_cfg); model_cfg = base
    model = MAPPO(obs_dim, env.global_state_dim, env.action_dim, env.num_agents, model_cfg, device)

    rollout_steps = int(model_cfg["rollout_steps"])
    total_steps = episodes * int(cfg["training"]["max_steps_per_episode"])
    updates = max(1, math.ceil(total_steps / rollout_steps))
    logs = []

    for update in range(updates):
        if not central:
            batch, infos, *_ = collect_rollout(env, model, rollout_steps)
        else:
            state, _, masks = env.reset()
            states=[]; obss=[]; maskss=[]; actions=[]; logps=[]; rewards=[]; dones=[]; values=[]; infos=[]
            for _ in range(rollout_steps):
                cobs = centralized_observation(state, env.num_agents)
                a, lp = model.act(cobs, masks)
                v = model.value(state)
                ns, _, nm, r, d, info = env.step(a)
                states.append(state); obss.append(cobs); maskss.append(masks); actions.append(a); logps.append(lp)
                rewards.append(float(r[0])); dones.append(bool(d[0])); values.append(v); infos.append(info)
                state, masks = ns, nm
                if bool(d[0]):
                    state, _, masks = env.reset()
            batch = RolloutBatch(
                states=np.asarray(states,np.float32), obs=np.asarray(obss,np.float32), masks=np.asarray(maskss,bool),
                actions=np.asarray(actions,np.int64), log_probs=np.asarray(logps,np.float32), rewards=np.asarray(rewards,np.float32),
                dones=np.asarray(dones,np.float32), values=np.asarray(values,np.float32), next_value=model.value(state))
        losses = model.update(batch)
        summary = summarize_episode(infos)
        logs.append({"update": update + 1, **summary, **losses})
        if (update + 1) % max(1, updates // 10) == 0:
            print(f"[{method}] update {update+1}/{updates} reward={summary.get('cumulative_reward',0):.3f}")

    ckpt = project_root() / "data" / f"model_{method}_seed{seed}.pt"
    model.save(str(ckpt))
    pd.DataFrame(logs).to_csv(project_root() / "data" / f"training_{method}_seed{seed}.csv", index=False)
    return ckpt


def train_qmix(cfg, seed: int, episodes: int, scenario: str):
    env = SharedWarehouseMAEnv(cfg, seed, scenario=scenario, ablation="full_model")
    device = resolve_device(cfg["project"].get("device", "auto"))
    model = QMIX(env.local_obs_dim, env.global_state_dim, env.action_dim, env.num_agents, cfg["qmix"], device)
    logs=[]; global_step=0
    max_steps = int(cfg["training"]["max_steps_per_episode"])
    for ep in range(episodes):
        state, obs, masks = env.reset(); infos=[]; losses=[]
        for _ in range(max_steps):
            actions = model.act(obs, masks, step=global_step)
            ns, no, nm, r, d, info = env.step(actions)
            model.store(Transition(state, obs, masks, actions, float(r[0]), ns, no, nm, bool(d[0])))
            out = model.train_step()
            if np.isfinite(out["loss"]): losses.append(out["loss"])
            infos.append(info); state, obs, masks = ns, no, nm; global_step += 1
            if bool(d[0]): break
        summary=summarize_episode(infos)
        logs.append({"episode": ep+1, **summary, "loss": float(np.mean(losses)) if losses else np.nan})
        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"[qmix] episode {ep+1}/{episodes} reward={summary.get('cumulative_reward',0):.3f}")
    ckpt = project_root()/"data"/f"model_qmix_seed{seed}.pt"; model.save(str(ckpt))
    pd.DataFrame(logs).to_csv(project_root()/"data"/f"training_qmix_seed{seed}.csv", index=False)
    return ckpt


def main():
    p=argparse.ArgumentParser(description="Train reproducibility baselines or DT-MAPPO.")
    p.add_argument("--method", choices=["mappo","single_ppo","qmix"], default="mappo")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--scenario", choices=["normal","demand_shock","severe_demand_shock","layout_perturbation","delayed_twin"], default="normal")
    p.add_argument("--ablation", choices=["full_model","without_dynamic_twin","without_marl","without_fairness_reward"], default="full_model")
    args=p.parse_args(); cfg=load_config(); set_global_seed(args.seed)
    episodes=args.episodes or int(cfg["training"]["episodes"])
    if args.method=="qmix": train_qmix(cfg,args.seed,episodes,args.scenario)
    else: train_mappo(cfg,args.seed,episodes,args.method,args.scenario,args.ablation)

if __name__=="__main__": main()
