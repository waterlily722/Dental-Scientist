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


def extract_final_info_payload(payload: Dict[str, Any], dataset_name: str | None = None) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}

    resolved_dataset_name = dataset_name or next(iter(payload.keys()), "")
    if not resolved_dataset_name:
        return {}

    dataset_payload = payload.get(resolved_dataset_name, {})
    if not isinstance(dataset_payload, dict):
        return {}

    if isinstance(dataset_payload.get("result"), dict):
        return dataset_payload["result"]
    if isinstance(dataset_payload.get("final_info_dict"), dict):
        return dataset_payload["final_info_dict"]
    if isinstance(dataset_payload.get("means"), dict):
        return dataset_payload["means"]
    return dataset_payload


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
            "result_type": "single_run",
            "result": final_info,
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
