from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from config import RESULTS_DIR


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_json(file_name: str, payload: Any) -> Path:
    output_dir = ensure_results_dir()
    out_path = output_dir / file_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def write_text(file_name: str, content: str) -> Path:
    output_dir = ensure_results_dir()
    out_path = output_dir / file_name
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_pickle(file_name: str, obj: Any) -> Path:
    output_dir = ensure_results_dir()
    out_path = output_dir / file_name
    with out_path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path
