from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.marl_env import SharedWarehouseMAEnv
from src.utils import load_config, project_root, sha256_file


def ensure_data():
    if not (project_root()/"data"/"warehouse_layout.csv").exists():
        subprocess.run([sys.executable,"generate_dataset.py"],cwd=project_root(),check=True)


def main():
    ensure_data(); cfg=load_config(); root=project_root()
    required=["README.md","REPRODUCIBILITY.md","CODE_AVAILABILITY.md","requirements.txt","config.yaml","generate_dataset.py","train.py","evaluate.py","reproduce_all.py"]
    for name in required:
        assert (root/name).exists(),f"Missing {name}"
    assert not (root/"LICENSE").exists(),"A LICENSE file was not requested for this under-review repository."

    env=SharedWarehouseMAEnv(cfg,seed=11)
    s,o,m=env.reset()
    assert s.shape==(env.global_state_dim,)
    assert o.shape==(env.num_agents,env.local_obs_dim)
    assert m.shape==(env.num_agents,env.action_dim)
    assert np.all(m[:,-1]),"No-op must always be available."
    before=int(env.twin.occupancy.sum())
    actions=np.full(env.num_agents,env.max_candidates,dtype=int)
    ns,no,nm,r,d,info=env.step(actions)
    assert np.all(np.isfinite(ns)) and np.all(np.isfinite(no))
    assert 0.0 <= info["fairness"] <= 1.000001
    assert np.all((info["congestion"]>=0)&(info["congestion"]<=1.0+1e-6))
    assert int(env.twin.occupancy.sum()) >= 0

    # Deterministic reset for fixed seed/scenario.
    env2=SharedWarehouseMAEnv(cfg,seed=11)
    s2,o2,m2=env2.reset()
    env3=SharedWarehouseMAEnv(cfg,seed=11)
    s3,o3,m3=env3.reset()
    assert np.allclose(s2,s3) and np.allclose(o2,o3) and np.array_equal(m2,m3)

    # Verify recorded checksums.
    checksum_path=root/"data"/"checksums.sha256"
    if checksum_path.exists():
        for line in checksum_path.read_text().splitlines():
            digest,name=line.split(maxsplit=1); name=name.strip()
            assert sha256_file(root/"data"/name)==digest,f"Checksum mismatch: {name}"
    print("Repository validation PASSED")

if __name__=="__main__": main()
