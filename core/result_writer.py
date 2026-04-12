import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_run_outputs(
    out_dir: str,
    dataset_name: str,
    final_info: Dict[str, Any],
    train_log_info: List[Dict[str, Any]],
    val_log_info: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any] | None = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset_summary = dataset_summary or {}
    final_info = _jsonable(final_info)
    train_log_info = _jsonable(train_log_info)
    val_log_info = _jsonable(val_log_info)
    dataset_summary = _jsonable(dataset_summary)

    all_results = {
        f"{dataset_name}_0_train_log_info": train_log_info,
        f"{dataset_name}_0_val_log_info": val_log_info,
        f"{dataset_name}_0_final_info": final_info,
    }

    final_info_wrapped = {
        dataset_name: {
            "means": final_info,
            "stderrs": {
                key: 0.0
                for key, value in final_info.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            "final_info_dict": final_info,
        }
    }

    (out_path / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_path / "final_info.json").write_text(
        json.dumps(final_info_wrapped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_path / "all_results.npy").open("wb") as handle:
        np.save(handle, all_results, allow_pickle=True)