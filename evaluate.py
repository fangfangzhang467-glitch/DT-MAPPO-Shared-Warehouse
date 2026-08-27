from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.baselines import RollingHorizonHeuristic, StaticClassBasedSlotting, centralized_observation
from src.mappo import MAPPO
from src.marl_env import SharedWarehouseMAEnv
from src.metrics import composite_index, summarize_episode
from src.qmix import QMIX
from src.utils import load_config, project_root, resolve_device, set_global_seed


def evaluate(method: str, seed: int, scenario: str, episodes: int, ablation: str = "full_model", checkpoint: str | None = None):
    cfg=load_config(); set_global_seed(seed)
    env=SharedWarehouseMAEnv(cfg,seed,scenario=scenario,ablation=ablation)
    device=resolve_device(cfg["project"].get("device","auto"))
    policy=None
    if method=="static": policy=StaticClassBasedSlotting()
    elif method=="rolling": policy=RollingHorizonHeuristic()
    elif method=="mappo":
        policy=MAPPO(env.local_obs_dim,env.global_state_dim,env.action_dim,env.num_agents,cfg["mappo"],device)
        path=Path(checkpoint) if checkpoint else project_root()/"data"/f"model_mappo_seed{seed}.pt"; policy.load(str(path))
    elif method=="single_ppo":
        mc=cfg["mappo"].copy(); mc.update(cfg["single_agent_ppo"])
        policy=MAPPO(env.global_state_dim,env.global_state_dim,env.action_dim,env.num_agents,mc,device)
        path=Path(checkpoint) if checkpoint else project_root()/"data"/f"model_single_ppo_seed{seed}.pt"; policy.load(str(path))
    elif method=="qmix":
        policy=QMIX(env.local_obs_dim,env.global_state_dim,env.action_dim,env.num_agents,cfg["qmix"],device)
        path=Path(checkpoint) if checkpoint else project_root()/"data"/f"model_qmix_seed{seed}.pt"; policy.load(str(path))
    else: raise ValueError(method)

    rows=[]
    for ep in range(episodes):
        state,obs,masks=env.reset(); infos=[]
        for _ in range(int(cfg["training"]["max_steps_per_episode"])):
            if method in {"static","rolling"}: actions=policy.select(env,masks)
            elif method=="mappo": actions,_=policy.act(obs,masks,deterministic=True)
            elif method=="single_ppo": actions,_=policy.act(centralized_observation(state,env.num_agents),masks,deterministic=True)
            else: actions=policy.act(obs,masks,step=0,deterministic=True)
            state,obs,masks,r,d,info=env.step(actions); infos.append(info)
            if bool(d[0]): break
        s=summarize_episode(infos); s["composite_index"]=composite_index(s)
        rows.append({"method":method,"seed":seed,"scenario":scenario,"ablation":ablation,"episode":ep+1,**s})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description="Evaluate a warehouse allocation method on paired digital-twin scenarios.")
    p.add_argument("--method",choices=["static","rolling","single_ppo","qmix","mappo"],required=True)
    p.add_argument("--seed",type=int,default=11); p.add_argument("--scenario",default="normal",choices=["normal","demand_shock","severe_demand_shock","layout_perturbation","delayed_twin"])
    p.add_argument("--episodes",type=int,default=None); p.add_argument("--ablation",default="full_model",choices=["full_model","without_dynamic_twin","without_marl","without_fairness_reward"])
    p.add_argument("--checkpoint",default=None); p.add_argument("--output",default=None)
    args=p.parse_args(); cfg=load_config(); n=args.episodes or int(cfg["training"]["evaluation_episodes"])
    df=evaluate(args.method,args.seed,args.scenario,n,args.ablation,args.checkpoint)
    out=Path(args.output) if args.output else project_root()/"data"/f"results_{args.method}_{args.scenario}_seed{args.seed}.csv"
    df.to_csv(out,index=False); print(df.mean(numeric_only=True).to_string()); print(f"Saved: {out}")

if __name__=="__main__": main()
