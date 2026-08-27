from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.demand_model import NonStationaryDemandModel
from src.utils import load_config, project_root, sha256_file


def generate_layout(cfg: dict, seed: int) -> pd.DataFrame:
    w = cfg["warehouse"]
    n = int(w["storage_locations"])
    zones = int(w["zones"])
    aisles = int(w["aisles"])
    rng = np.random.default_rng(seed)

    slot_id = np.arange(n)
    aisle = slot_id % aisles
    zone = aisle // max(aisles // zones, 1)
    zone = np.minimum(zone, zones - 1)
    position = slot_id // aisles
    x = (aisle % 8) * 4.0 + 2.0
    y = (position % 200) * 1.5 + 1.0
    levels = rng.integers(int(w["rack_levels_min"]), int(w["rack_levels_max"]) + 1, n)
    capacity = rng.integers(52, 70, n)
    category = rng.choice(["general", "fast", "fragile", "bulk"], n, p=[0.45, 0.25, 0.18, 0.12])
    return pd.DataFrame({
        "slot_id": slot_id, "zone": zone, "aisle": aisle, "x": x, "y": y,
        "level": levels, "capacity": capacity, "category": category,
    })


def generate_tenant_profiles() -> pd.DataFrame:
    return pd.DataFrame([
        [0, "Tenant A", "Low", 18, 14, "General goods"],
        [1, "Tenant B", "Medium", 12, 19, "Temperature-insensitive"],
        [2, "Tenant C", "High", 7, 26, "Fast-moving consumer goods"],
        [3, "Tenant D", "Medium", 10, 17, "Fragile items"],
        [4, "Tenant E", "High", 6, 16, "Promotional goods"],
        [5, "Tenant F", "Low", 21, 8, "Bulk inventory"],
    ], columns=["tenant_id", "tenant_name", "demand_volatility", "avg_turnover_days", "inventory_share_pct", "product_profile"])


def generate_compatibility() -> pd.DataFrame:
    allowed = {
        0: {"general", "fast"},
        1: {"general", "fast", "bulk"},
        2: {"fast", "general"},
        3: {"fragile", "general"},
        4: {"fast", "general"},
        5: {"bulk", "general"},
    }
    rows = []
    for t in range(6):
        for cat in ["general", "fast", "fragile", "bulk"]:
            rows.append([t, cat, cat in allowed[t]])
    return pd.DataFrame(rows, columns=["tenant_id", "category", "allowed"])


def generate_reference_events(cfg: dict, seed: int, n_steps: int = 2016) -> pd.DataFrame:
    model = NonStationaryDemandModel(cfg["demand"], int(cfg["warehouse"]["decision_interval_seconds"]), seed)
    rows = []
    for step in range(n_steps):
        snap = model.sample(step)
        for tenant in range(len(snap.inbound_units)):
            rows.append({
                "step": step,
                "time_seconds": step * int(cfg["warehouse"]["decision_interval_seconds"]),
                "tenant_id": tenant,
                "inbound_units": int(snap.inbound_units[tenant]),
                "outbound_orders": int(snap.outbound_orders[tenant]),
                "demand_intensity_per_hour": float(snap.demand_intensity[tenant]),
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Generate reproducible shared-warehouse reference data.")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    cfg = load_config()
    data = project_root() / "data"
    data.mkdir(exist_ok=True)

    generate_layout(cfg, args.seed).to_csv(data / "warehouse_layout.csv", index=False)
    generate_tenant_profiles().to_csv(data / "tenant_profiles.csv", index=False)
    generate_compatibility().to_csv(data / "product_compatibility.csv", index=False)

    with open(data / "scenario_manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": cfg["experiments"], "ablations": cfg["ablation"]}, f, sort_keys=False)

    events = generate_reference_events(cfg, args.seed + 1)
    events.to_csv(data / "reference_events.csv.gz", index=False, compression="gzip")

    files = ["warehouse_layout.csv", "tenant_profiles.csv", "product_compatibility.csv", "scenario_manifest.yaml", "reference_events.csv.gz"]
    with open(data / "checksums.sha256", "w", encoding="utf-8") as f:
        for name in files:
            f.write(f"{sha256_file(data / name)}  {name}\n")
    print(f"Generated {len(files)} reference data files in {data}")


if __name__ == "__main__":
    main()
