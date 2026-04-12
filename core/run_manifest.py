import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def create_run_manifest(
    *,
    idea: Dict[str, Any],
    experiment: str,
    folder_name: str,
    base_dir: str,
    cli_args: Dict[str, Any],
    baseline_snapshot: Dict[str, Any],
    memory_hits: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "experiment": experiment,
        "idea": deepcopy(idea),
        "folder_name": folder_name,
        "base_dir": base_dir,
        "cli_args": _jsonable(deepcopy(cli_args)),
        "baseline_snapshot": _jsonable(deepcopy(baseline_snapshot)),
        "memory_hits": _jsonable(deepcopy(memory_hits or [])),
        "status": "initialized",
        "stages": {},
        "artifacts": {},
        "summary": {},
    }


def update_stage(
    manifest: Dict[str, Any],
    stage_name: str,
    *,
    status: str,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    stage = dict(manifest.get("stages", {}).get(stage_name, {}))
    stage["status"] = status
    stage["updated_at"] = utc_now_iso()
    if "started_at" not in stage:
        stage["started_at"] = stage["updated_at"]
    if details:
        stage.update(_jsonable(details))
    manifest.setdefault("stages", {})[stage_name] = stage
    manifest["status"] = status
    manifest["updated_at"] = utc_now_iso()
    return manifest


def finalize_manifest(
    manifest: Dict[str, Any],
    *,
    success: bool,
    summary: Dict[str, Any] | None = None,
    artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    manifest["status"] = "completed" if success else "failed"
    manifest["updated_at"] = utc_now_iso()
    if summary:
        manifest["summary"] = _jsonable(summary)
    if artifacts:
        merged = dict(manifest.get("artifacts", {}))
        merged.update(_jsonable(artifacts))
        manifest["artifacts"] = merged
    return manifest


def write_run_manifest(folder_name: str, manifest: Dict[str, Any]) -> Path:
    path = Path(folder_name) / "run_manifest.json"
    path.write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
