from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(data: Any, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(record: Dict[str, Any], path: str | os.PathLike) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def safe_mean(values) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if a.size else 0.0
