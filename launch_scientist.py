import argparse
import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from ai_scientist.generate_ideas import check_idea_novelty, generate_ideas
from ai_scientist.llm import AVAILABLE_LLMS, create_client
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import load_paper, perform_improvement, perform_review
from ai_scientist.perform_writeup import generate_latex, perform_writeup
from core.dental_context import build_dental_task_context, write_dental_task_context
from core.memory import append_memory_record, format_memory_for_prompt, retrieve_relevant_memories
from core.registry import list_task_names
from core.run_manifest import create_run_manifest, finalize_manifest, update_stage, write_run_manifest
from core.validators import post_experiment_validate, pre_experiment_validate

NUM_REFLECTIONS = 3
DENTAL_TEMPLATES = {"dental_cls_v1", "dental_det_v1", "dental_seg_v1", "dental_keypoint_v1"}


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="dental_cls_v1",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-max-2026-01-23",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--review-model",
        type=str,
        default="qwen3-max-2026-01-23",
        choices=AVAILABLE_LLMS,
        help="Model to use for paper review.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="semanticscholar",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="",
        help="Optional benchmark task name override for templates that support task-driven execution.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Optional data_root override forwarded to templates that support it.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="",
        help="Optional split_file override forwarded to templates that support it.",
    )
    parser.add_argument(
        "--memory_dir",
        type=str,
        default="",
        help="Optional directory or .jsonl path used for lightweight lab memory records.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Optional num_seeds override forwarded to templates that support it.",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="Skip the writeup stage.",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="Skip the review stage.",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def check_latex_dependencies():
    required_dependencies = ["pdflatex", "chktex"]
    missing_deps = []
    for dep in required_dependencies:
        if shutil.which(dep) is None:
            missing_deps.append(dep)
    if missing_deps:
        print("Error: Required LaTeX dependencies not found:", file=sys.stderr)
        return False
    return True


def resolve_aider_model(model_name: str) -> str:
    if model_name == "deepseek-coder-v2-0724":
        return "deepseek/deepseek-coder"
    if model_name == "deepseek-reasoner":
        return "deepseek/deepseek-reasoner"
    if model_name == "llama3.1-405b":
        return "openrouter/meta-llama/llama-3.1-405b-instruct"
    if "qwen" in model_name:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault(
            "OPENAI_API_BASE",
            os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        return f"openai/{model_name}"
    return model_name


def resolve_default_task_name(experiment: str, task_name: str) -> str:
    if task_name:
        return task_name
    available = set(list_task_names())
    preferred_by_template = {
        "dental_cls_v1": "dental_caries_classificationv3",
        "dental_det_v1": "Dental_Radiography",
        "dental_seg_v1": "AlphaDent",
        "dental_keypoint_v1": "Dentistry_Computer_Vision_Dataset",
    }
    preferred = preferred_by_template.get(experiment, "")
    if preferred and preferred in available:
        return preferred
    return task_name


def ensure_required_baseline(base_dir: str, experiment: str) -> None:
    baseline_path = osp.join(base_dir, "run_0", "final_info.json")
    if experiment == "dental_cls_v1" and not osp.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline required for {experiment} but missing: {baseline_path}. "
            "Please run `python experiment.py --out_dir run_0` in the template first."
        )


def load_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_baseline_snapshot(base_dir: str) -> Dict[str, Any]:
    baseline_path = osp.join(base_dir, "run_0", "final_info.json")
    baseline_wrapped = load_json_file(baseline_path)
    if not baseline_wrapped:
        return {
            "available": False,
            "path": baseline_path,
            "dataset_name": "",
            "primary_metric_name": "",
            "best_val_primary": None,
            "best_test_primary": None,
            "best_epoch": None,
            "means": {},
        }

    dataset_name = next(iter(baseline_wrapped.keys()))
    means = baseline_wrapped.get(dataset_name, {}).get("means", {})
    scorecard = means.get("scorecard", {}) if isinstance(means, dict) else {}
    return {
        "available": True,
        "path": baseline_path,
        "dataset_name": dataset_name,
        "primary_metric_name": scorecard.get("primary_metric_name", means.get("primary_metric_name", "")),
        "best_val_primary": scorecard.get("best_val_primary"),
        "best_test_primary": scorecard.get("best_test_primary"),
        "best_epoch": scorecard.get("best_epoch", means.get("best_epoch")),
        "means": means if isinstance(means, dict) else {},
    }


def flatten_baseline_results(baseline_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    means = baseline_snapshot.get("means", {})
    if not isinstance(means, dict):
        return {}
    flattened = {}
    for key, value in means.items():
        if isinstance(value, dict) and "means" in value:
            flattened[key] = value["means"]
        else:
            flattened[key] = value
    return flattened


def infer_change_axis(idea: Dict[str, Any]) -> str:
    text = " ".join(str(idea.get(key, "")) for key in ["Name", "Title", "Experiment"]).lower()
    axis_keywords = {
        "preprocessing": ["preprocess", "equalize", "normalize", "contrast"],
        "augmentation_policy": ["augment", "flip", "crop", "jitter", "noise"],
        "backbone_selection": ["backbone", "encoder", "efficientnet", "resnet", "unet", "heatmap"],
        "loss_design": ["loss", "focal", "dice", "weight", "calibration"],
        "threshold_selection": ["threshold", "calibration", "temperature"],
        "optimization": ["optimizer", "learning rate", "scheduler", "weight decay"],
    }
    for axis, keywords in axis_keywords.items():
        if any(keyword in text for keyword in keywords):
            return axis
    return "general_template_modification"


def resolve_memory_dir(args, results_dir: str) -> str:
    if args.memory_dir:
        return args.memory_dir
    return osp.join(results_dir, "memory")


def build_run_config(args, task_name: str) -> Dict[str, Any]:
    return {
        "task_name": task_name,
        "data_root": args.data_root,
        "split_file": args.split_file,
        "num_seeds": args.num_seeds,
    }


def prepare_template_context(args, base_dir: str, task_name: str) -> Dict[str, Any]:
    if args.experiment not in DENTAL_TEMPLATES:
        return {}
    context_path = write_dental_task_context(
        repo_root=Path(".").resolve(),
        base_dir=Path(base_dir).resolve(),
        task_name=task_name,
        data_root_override=args.data_root,
        split_file_override=args.split_file,
    )
    print(f"Prepared dental task context at: {context_path}")
    return build_dental_task_context(
        repo_root=Path(".").resolve(),
        base_dir=Path(base_dir).resolve(),
        task_name=task_name,
        data_root_override=args.data_root,
        split_file_override=args.split_file,
    )


def prepare_run_folder(
    base_dir: str,
    results_dir: str,
    idea: Dict[str, Any],
    cli_args: Dict[str, Any],
    baseline_snapshot: Dict[str, Any],
    memory_hits: list[Dict[str, Any]] | None = None,
) -> Tuple[str, str, Dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    if osp.exists(folder_name):
        raise FileExistsError(f"Folder {folder_name} already exists.")
    shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)
    notes = osp.join(folder_name, "notes.txt")
    baseline_results = flatten_baseline_results(baseline_snapshot)
    with open(notes, "w", encoding="utf-8") as handle:
        handle.write(f"# Title: {idea['Title']}\n")
        handle.write(f"# Experiment description: {idea['Experiment']}\n")
        handle.write("## Run 0: Baseline\n")
        handle.write(f"Results: {baseline_results}\n")
        handle.write("Description: Baseline results.\n")

    manifest = create_run_manifest(
        idea=idea,
        experiment=osp.basename(base_dir),
        folder_name=folder_name,
        base_dir=base_dir,
        cli_args=cli_args,
        baseline_snapshot=baseline_snapshot,
        memory_hits=memory_hits,
    )
    write_run_manifest(folder_name, manifest)
    return idea_name, folder_name, manifest


def create_experiment_coder(folder_name: str, idea_name: str, model: str, include_writeup: bool = False):
    exp_file = osp.join(folder_name, "experiment.py")
    notes = osp.join(folder_name, "notes.txt")
    fnames = [exp_file, osp.join(folder_name, "plot.py"), notes]
    if include_writeup:
        fnames = [exp_file, osp.join(folder_name, "latex", "template.tex"), notes]
    io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
    return Coder.create(
        main_model=Model(resolve_aider_model(model)),
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )


def run_experiment_stage(
    *,
    idea: Dict[str, Any],
    folder_name: str,
    model: str,
    baseline_snapshot: Dict[str, Any],
    run_config: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    baseline_results = flatten_baseline_results(baseline_snapshot)
    idea_name = osp.basename(folder_name)
    pre_validation = pre_experiment_validate(
        folder_name=folder_name,
        baseline_snapshot=baseline_snapshot,
        task_context=load_json_file(osp.join(folder_name, "task_context.json")),
    )
    update_stage(manifest, "pre_experiment_validate", status=pre_validation["status"], details=pre_validation)
    write_run_manifest(folder_name, manifest)
    if pre_validation["status"] == "fail":
        return False, manifest, {"pre_validation": pre_validation}

    coder = create_experiment_coder(folder_name, idea_name, model)
    success = perform_experiments(
        idea,
        folder_name,
        coder,
        baseline_results,
        run_config=run_config,
    )
    experiment_details = {"success": success, "run_config": run_config}
    update_stage(
        manifest,
        "experiment",
        status="completed" if success else "failed",
        details=experiment_details,
    )
    write_run_manifest(folder_name, manifest)
    if not success:
        return False, manifest, {"pre_validation": pre_validation, "experiment": experiment_details}

    post_validation = post_experiment_validate(
        folder_name=folder_name,
        baseline_snapshot=baseline_snapshot,
    )
    update_stage(manifest, "post_experiment_validate", status=post_validation["status"], details=post_validation)
    write_run_manifest(folder_name, manifest)
    if post_validation["status"] == "fail":
        return False, manifest, {"pre_validation": pre_validation, "experiment": experiment_details, "post_validation": post_validation}
    return True, manifest, {"pre_validation": pre_validation, "experiment": experiment_details, "post_validation": post_validation}


def run_writeup_stage(
    *,
    idea: Dict[str, Any],
    folder_name: str,
    model: str,
    client,
    client_model,
    writeup: str,
    engine: str,
    skip_writeup: bool,
    manifest: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    if skip_writeup:
        details = {"skipped": True}
        update_stage(manifest, "writeup", status="skipped", details=details)
        write_run_manifest(folder_name, manifest)
        return True, manifest, details

    if writeup != "latex":
        details = {"error": f"Writeup format {writeup} not supported."}
        update_stage(manifest, "writeup", status="failed", details=details)
        write_run_manifest(folder_name, manifest)
        return False, manifest, details

    coder = create_experiment_coder(folder_name, osp.basename(folder_name), model, include_writeup=True)
    try:
        perform_writeup(idea, folder_name, coder, client, client_model, engine=engine)
    except Exception as exc:
        details = {"error": str(exc)}
        update_stage(manifest, "writeup", status="failed", details=details)
        write_run_manifest(folder_name, manifest)
        return False, manifest, details

    details = {"skipped": False, "pdf_path": osp.join(folder_name, f"{idea['Name']}.pdf")}
    update_stage(manifest, "writeup", status="completed", details=details)
    write_run_manifest(folder_name, manifest)
    return True, manifest, details


def run_review_stage(
    *,
    idea: Dict[str, Any],
    folder_name: str,
    writeup: str,
    review_model: str,
    improvement: bool,
    skip_review: bool,
    skip_writeup: bool,
    manifest: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    if skip_review or skip_writeup:
        details = {"skipped": True, "reason": "skip_review flag set" if skip_review else "writeup was skipped"}
        update_stage(manifest, "review", status="skipped", details=details)
        write_run_manifest(folder_name, manifest)
        return True, manifest, details

    if writeup != "latex":
        details = {"error": f"Writeup format {writeup} not supported for review."}
        update_stage(manifest, "review", status="failed", details=details)
        write_run_manifest(folder_name, manifest)
        return False, manifest, details

    try:
        review_client, review_client_model = create_client(review_model)
        paper_path = f"{folder_name}/{idea['Name']}.pdf"
        paper_text = load_paper(paper_path)
        review = perform_review(
            paper_text,
            model=review_client_model,
            client=review_client,
            num_reflections=3,
            num_fs_examples=1,
            num_reviews_ensemble=3,
            temperature=0.1,
        )
        with open(osp.join(folder_name, "review.txt"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(review, indent=4, ensure_ascii=False))
    except Exception as exc:
        details = {"error": str(exc)}
        update_stage(manifest, "review", status="failed", details=details)
        write_run_manifest(folder_name, manifest)
        return False, manifest, details

    details = {"skipped": False, "improvement": improvement}
    if improvement:
        try:
            coder = create_experiment_coder(folder_name, osp.basename(folder_name), review_model, include_writeup=True)
            perform_improvement(review, coder)
            generate_latex(coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf")
            paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
            review_client, review_client_model = create_client(review_model)
            review_improved = perform_review(
                paper_text,
                model=review_client_model,
                client=review_client,
                num_reflections=3,
                num_fs_examples=1,
                num_reviews_ensemble=3,
                temperature=0.1,
            )
            with open(osp.join(folder_name, "review_improved.txt"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(review_improved, ensure_ascii=False))
            details["improved_pdf_path"] = f"{folder_name}/{idea['Name']}_improved.pdf"
        except Exception as exc:
            details = {"error": str(exc), "improvement": True}
            update_stage(manifest, "review", status="failed", details=details)
            write_run_manifest(folder_name, manifest)
            return False, manifest, details

    update_stage(manifest, "review", status="completed", details=details)
    write_run_manifest(folder_name, manifest)
    return True, manifest, details


def summarize_experiment_outcome(folder_name: str, baseline_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    primary_metric_name = str(baseline_snapshot.get("primary_metric_name", "")).strip()
    run_dirs = sorted(path for path in Path(folder_name).glob("run_*") if path.is_dir() and path.name != "run_0")
    completed_runs = [path.name for path in run_dirs if (path / "final_info.json").exists()]
    summary: Dict[str, Any] = {
        "completed_runs": completed_runs,
        "num_completed_runs": len(completed_runs),
        "primary_metric_name": primary_metric_name,
    }
    if not completed_runs:
        return summary

    last_run = run_dirs[-1]
    payload = load_json_file(str(last_run / "final_info.json"))
    dataset_name = next(iter(payload.keys()), "")
    means = payload.get(dataset_name, {}).get("means", {}) if dataset_name else {}
    if isinstance(means, dict):
        scorecard = means.get("scorecard", {}) if isinstance(means.get("scorecard", {}), dict) else {}
        summary["last_run"] = last_run.name
        summary["last_run_best_val_primary"] = scorecard.get("best_val_primary")
        summary["last_run_best_test_primary"] = scorecard.get("best_test_primary")
    return summary


def append_memory_after_experiment(
    *,
    memory_dir: str,
    idea: Dict[str, Any],
    folder_name: str,
    experiment_success: bool,
    baseline_snapshot: Dict[str, Any],
    post_validation: Dict[str, Any] | None,
) -> str:
    summary = summarize_experiment_outcome(folder_name, baseline_snapshot)
    failure_reason = ""
    if not experiment_success:
        failure_reason = "experiment_stage_failed"
    elif post_validation and post_validation.get("status") == "fail":
        failure_reason = "post_validation_failed"

    lesson = "Completed experiment stage without protocol violations."
    if failure_reason:
        lesson = "Keep the change narrower and preserve benchmark/template assumptions."
    elif summary.get("primary_metric_name"):
        lesson = f"Track {summary['primary_metric_name']} consistently and compare against the baseline scorecard."

    append_memory_record(
        memory_dir,
        {
            "idea_name": idea.get("Name", ""),
            "change_axis": infer_change_axis(idea),
            "success": bool(experiment_success and (post_validation or {}).get("status", "pass") != "fail"),
            "failure_reason": failure_reason,
            "lesson": lesson,
        },
    )
    return lesson


def finalize_run_manifest(
    *,
    folder_name: str,
    manifest: Dict[str, Any],
    success: bool,
    baseline_snapshot: Dict[str, Any],
    stage_results: Dict[str, Any],
) -> Dict[str, Any]:
    summary = summarize_experiment_outcome(folder_name, baseline_snapshot)
    summary["stage_results"] = stage_results
    artifacts = {
        "run_manifest": osp.join(folder_name, "run_manifest.json"),
        "log_file": osp.join(folder_name, "log.txt"),
        "review_file": osp.join(folder_name, "review.txt"),
    }
    finalize_manifest(manifest, success=success, summary=summary, artifacts=artifacts)
    write_run_manifest(folder_name, manifest)
    return manifest


def worker(
    queue,
    base_dir,
    results_dir,
    model,
    writeup,
    improvement,
    review_model,
    engine,
    gpu_id,
    run_config,
    experiment_name,
    memory_dir,
    skip_writeup,
    skip_review,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    client, client_model = create_client(model)
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir=base_dir,
            results_dir=results_dir,
            idea=idea,
            experiment_name=experiment_name,
            model=model,
            client=client,
            client_model=client_model,
            writeup=writeup,
            improvement=improvement,
            review_model=review_model,
            engine=engine,
            run_config=run_config,
            memory_dir=memory_dir,
            skip_writeup=skip_writeup,
            skip_review=skip_review,
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(
    *,
    base_dir,
    results_dir,
    idea,
    experiment_name,
    model,
    client,
    client_model,
    writeup,
    improvement,
    review_model,
    engine,
    run_config,
    memory_dir,
    skip_writeup,
    skip_review,
    log_file=False,
):
    baseline_snapshot = load_baseline_snapshot(base_dir)
    memory_hits = retrieve_relevant_memories(
        memory_dir,
        query_text=" ".join([experiment_name, run_config.get("task_name", ""), idea.get("Title", ""), idea.get("Experiment", "")]),
        limit=5,
    )
    cli_args = {
        "task_name": run_config.get("task_name", ""),
        "data_root": run_config.get("data_root", ""),
        "split_file": run_config.get("split_file", ""),
        "num_seeds": run_config.get("num_seeds", None),
        "skip_writeup": skip_writeup,
        "skip_review": skip_review,
    }
    idea_name, folder_name, manifest = prepare_run_folder(
        base_dir,
        results_dir,
        idea,
        cli_args,
        baseline_snapshot,
        memory_hits=memory_hits,
    )

    original_stdout = None
    original_stderr = None
    log = None
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a", encoding="utf-8")
        sys.stdout = log
        sys.stderr = log

    stage_results: Dict[str, Any] = {}
    overall_success = False
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")

        print_time()
        print("*Starting Experiments*")
        experiment_success, manifest, experiment_stage = run_experiment_stage(
            idea=idea,
            folder_name=folder_name,
            model=model,
            baseline_snapshot=baseline_snapshot,
            run_config=run_config,
            manifest=manifest,
        )
        stage_results["experiment_stage"] = experiment_stage
        memory_lesson = append_memory_after_experiment(
            memory_dir=memory_dir,
            idea=idea,
            folder_name=folder_name,
            experiment_success=experiment_success,
            baseline_snapshot=baseline_snapshot,
            post_validation=experiment_stage.get("post_validation"),
        )
        stage_results["memory"] = {"lesson": memory_lesson}
        update_stage(manifest, "memory", status="completed", details=stage_results["memory"])
        write_run_manifest(folder_name, manifest)
        if not experiment_success:
            finalize_run_manifest(
                folder_name=folder_name,
                manifest=manifest,
                success=False,
                baseline_snapshot=baseline_snapshot,
                stage_results=stage_results,
            )
            return False

        print_time()
        print("*Starting Writeup*")
        writeup_success, manifest, writeup_stage = run_writeup_stage(
            idea=idea,
            folder_name=folder_name,
            model=model,
            client=client,
            client_model=client_model,
            writeup=writeup,
            engine=engine,
            skip_writeup=skip_writeup,
            manifest=manifest,
        )
        stage_results["writeup_stage"] = writeup_stage
        if not writeup_success:
            finalize_run_manifest(
                folder_name=folder_name,
                manifest=manifest,
                success=False,
                baseline_snapshot=baseline_snapshot,
                stage_results=stage_results,
            )
            return False

        print_time()
        print("*Starting Review*")
        review_success, manifest, review_stage = run_review_stage(
            idea=idea,
            folder_name=folder_name,
            writeup=writeup,
            review_model=review_model,
            improvement=improvement,
            skip_review=skip_review,
            skip_writeup=skip_writeup,
            manifest=manifest,
        )
        stage_results["review_stage"] = review_stage
        if not review_success:
            finalize_run_manifest(
                folder_name=folder_name,
                manifest=manifest,
                success=False,
                baseline_snapshot=baseline_snapshot,
                stage_results=stage_results,
            )
            return False

        overall_success = True
        finalize_run_manifest(
            folder_name=folder_name,
            manifest=manifest,
            success=True,
            baseline_snapshot=baseline_snapshot,
            stage_results=stage_results,
        )
        return True
    except Exception as exc:
        stage_results["exception"] = {"error": str(exc), "traceback": traceback.format_exc()}
        update_stage(manifest, "runtime", status="failed", details=stage_results["exception"])
        finalize_run_manifest(
            folder_name=folder_name,
            manifest=manifest,
            success=False,
            baseline_snapshot=baseline_snapshot,
            stage_results=stage_results,
        )
        print(f"Failed to evaluate idea {idea_name}: {exc}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file and log is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


if __name__ == "__main__":
    args = parse_arguments()

    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")

    if args.writeup == "latex" and not args.skip_writeup and not check_latex_dependencies():
        sys.exit(1)

    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    task_name = resolve_default_task_name(args.experiment, args.task_name)
    ensure_required_baseline(base_dir, args.experiment)
    prepare_template_context(args, base_dir, task_name)

    memory_dir = resolve_memory_dir(args, results_dir)
    memory_hits = retrieve_relevant_memories(
        memory_dir,
        query_text=" ".join([args.experiment, task_name]),
        limit=5,
    )
    memory_context = format_memory_for_prompt(memory_hits)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
        memory_context=memory_context,
    )
    if not args.skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
            engine=args.engine,
        )

    with open(osp.join(base_dir, "ideas.json"), "w", encoding="utf-8") as handle:
        json.dump(ideas, handle, indent=4, ensure_ascii=False)

    novel_ideas = [idea for idea in ideas if idea["novel"]]
    run_config = build_run_config(args, task_name)

    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            process = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    args.model,
                    args.writeup,
                    args.improvement,
                    args.review_model,
                    args.engine,
                    gpu_id,
                    run_config,
                    args.experiment,
                    memory_dir,
                    args.skip_writeup,
                    args.skip_review,
                ),
            )
            process.start()
            time.sleep(150)
            processes.append(process)
        try:
            for _ in range(args.parallel):
                queue.put(None)
            for process in processes:
                process.join()
            print("All parallel processes completed.")
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Terminating worker processes...")
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            print("Workers terminated.")
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir=base_dir,
                    results_dir=results_dir,
                    idea=idea,
                    experiment_name=args.experiment,
                    model=args.model,
                    client=client,
                    client_model=client_model,
                    writeup=args.writeup,
                    improvement=args.improvement,
                    review_model=args.review_model,
                    engine=args.engine,
                    run_config=run_config,
                    memory_dir=memory_dir,
                    skip_writeup=args.skip_writeup,
                    skip_review=args.skip_review,
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}")
            except Exception as exc:
                print(f"Failed to evaluate idea {idea['Name']}: {exc}")
                print(traceback.format_exc())
    print("All ideas evaluated.")
