import ast
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_import_roots(source: str) -> List[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.append(node.module.split(".")[0])
    return roots


def _result(level: str, rule: str, message: str) -> Dict[str, str]:
    return {"level": level, "rule": rule, "message": message}


def _finalize_results(results: List[Dict[str, str]]) -> Dict[str, Any]:
    status = "pass"
    if any(item["level"] == "fail" for item in results):
        status = "fail"
    elif any(item["level"] == "warning" for item in results):
        status = "warning"
    return {"status": status, "checks": results}


def pre_experiment_validate(
    *,
    folder_name: str,
    baseline_snapshot: Dict[str, Any],
    task_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    folder = Path(folder_name)
    experiment_path = folder / "experiment.py"
    source = _safe_read_text(experiment_path)
    results: List[Dict[str, str]] = []

    task_context = task_context or {}
    if "--split_file" not in source:
        results.append(
            _result(
                "warning",
                "split_override_missing",
                "experiment.py does not visibly expose --split_file; verify the benchmark split protocol remains configurable and unchanged.",
            )
        )

    if "http://" in source or "https://" in source:
        results.append(_result("fail", "external_data_url", "experiment.py references an external URL, which may indicate external data usage."))

    suspicious_tokens = ["wget ", "curl ", "gdown", "kaggle", "huggingface_hub", "requests.get(", "urllib.request"]
    if any(token in source for token in suspicious_tokens):
        results.append(_result("fail", "external_data_fetch", "experiment.py appears to fetch external resources or datasets."))

    primary_metric_name = str(baseline_snapshot.get("primary_metric_name", "")).strip()
    if primary_metric_name and primary_metric_name not in source:
        results.append(
            _result(
                "warning",
                "primary_metric_visibility",
                f"Baseline primary metric '{primary_metric_name}' is not visible in experiment.py; verify the main metric was not removed.",
            )
        )

    if "--out_dir" not in source:
        results.append(_result("fail", "out_dir_cli", "experiment.py no longer exposes the --out_dir CLI argument."))
    if "parser.add_argument(\"--out_dir\"" not in source and "parser.add_argument('--out_dir'" not in source:
        results.append(_result("fail", "out_dir_parser", "experiment.py no longer defines --out_dir via argparse."))

    allowed_prefixes = {
        "argparse",
        "os",
        "sys",
        "json",
        "math",
        "time",
        "random",
        "pathlib",
        "dataclasses",
        "typing",
        "collections",
        "copy",
        "numpy",
        "np",
        "torch",
        "PIL",
        "core",
    }
    template_known = {"torchvision", "cv2", "sklearn", "scipy", "pandas", "matplotlib"}
    for root in sorted(set(_extract_import_roots(source))):
        if root in allowed_prefixes or root in template_known:
            continue
        results.append(
            _result(
                "warning",
                "new_dependency",
                f"experiment.py imports '{root}', which may be a new dependency outside the template baseline.",
            )
        )

    return _finalize_results(results)


def post_experiment_validate(
    *,
    folder_name: str,
    baseline_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    folder = Path(folder_name)
    results: List[Dict[str, str]] = []
    run_dirs = sorted(path for path in folder.glob("run_*") if path.is_dir() and path.name != "run_0")
    if not run_dirs:
        results.append(_result("warning", "missing_runs", "No run_* outputs were found for this idea."))
        return _finalize_results(results)

    primary_metric_name = str(baseline_snapshot.get("primary_metric_name", "")).strip()
    for run_dir in run_dirs:
        final_info_path = run_dir / "final_info.json"
        payload = _load_json(final_info_path)
        if not payload:
            results.append(_result("fail", "missing_final_info", f"{run_dir.name} is missing final_info.json."))
            continue
        dataset_name = next(iter(payload.keys()), "")
        means = payload.get(dataset_name, {}).get("means", {}) if dataset_name else {}
        if not isinstance(means, dict):
            results.append(_result("fail", "malformed_final_info", f"{run_dir.name} final_info.json has an unexpected structure."))
            continue

        if primary_metric_name:
            scorecard = means.get("scorecard", {}) if isinstance(means.get("scorecard", {}), dict) else {}
            best_val_metrics = means.get("best_val_metrics", {}) if isinstance(means.get("best_val_metrics", {}), dict) else {}
            best_test_metrics = means.get("best_test_metrics", {}) if isinstance(means.get("best_test_metrics", {}), dict) else {}
            has_metric = (
                scorecard.get("primary_metric_name") == primary_metric_name
                or primary_metric_name in best_val_metrics
                or primary_metric_name in best_test_metrics
            )
            if not has_metric:
                results.append(
                    _result(
                        "fail",
                        "missing_primary_metric",
                        f"{run_dir.name} does not preserve the baseline primary metric '{primary_metric_name}'.",
                    )
                )

    return _finalize_results(results)
