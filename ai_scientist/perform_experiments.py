import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired

MAX_ITERS = 4
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 1500

coder_prompt = """You are a careful Dental Scientist experimenter.
Your goal is to implement and evaluate the following research idea:

Title: {title}
Experiment idea: {idea}

You have a budget of up to {max_runs} experimental runs. You do not need to use all of them.

First, make a compact experimental plan for the runs you intend to use. Prefer a small number of clear, hypothesis-driven comparisons over an unfocused sweep.
Stay faithful to the benchmark protocol, preserve reproducibility, and avoid unnecessary changes outside the current template.

The baseline has already been run, so do not re-run it.
For reference, the baseline results are:

{baseline_results}

Important constraints:
- Keep the work clinically meaningful and mechanically plausible.
- Favor modifications that fit the current template and low-resource setting.
- Record enough detail in `notes.txt` for a later writeup.
- Do not invent extra datasets, labels, or benchmark-breaking changes.

After each code change, we will run `python experiment.py --out_dir=run_i` where `i` is the run number.
YOUR PROPOSED CHANGE MUST SUPPORT THIS EXACT COMMAND FORMAT. DO NOT ADD REQUIRED EXTRA COMMAND LINE ARGS.

Implement the first step of your plan now."""


def _load_task_context(folder_name):
    context_path = osp.join(folder_name, "task_context.json")
    if not osp.exists(context_path):
        return {}
    try:
        with open(context_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _stderr_tail(stderr_output: str) -> str:
    if len(stderr_output) <= MAX_STDERR_OUTPUT:
        return stderr_output
    return "..." + stderr_output[-MAX_STDERR_OUTPUT:]


def run_static_checks(folder_name, timeout=120):
    cwd = osp.abspath(folder_name)
    command = ["python", "-m", "py_compile", "experiment.py", "plot.py"]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            stderr_output = _stderr_tail(result.stderr or "")
            return result.returncode, f"Static validation failed with the following error {stderr_output}"
        return 0, ""
    except TimeoutExpired:
        return 1, f"Static validation timed out after {timeout} seconds"


def run_precheck(folder_name, timeout=600):
    cwd = osp.abspath(folder_name)
    command = ["python", "experiment.py", "--out_dir=run_precheck"]
    env = os.environ.copy()
    env["AI_SCIENTIST_PRECHECK"] = "1"
    env.setdefault("AI_SCIENTIST_PRECHECK_MAX_SAMPLES", "48")
    env.setdefault("AI_SCIENTIST_PRECHECK_MAX_TRAIN_BATCHES", "2")
    env.setdefault("AI_SCIENTIST_PRECHECK_MAX_EVAL_BATCHES", "2")
    env.setdefault("AI_SCIENTIST_PRECHECK_EPOCHS", "1")

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        precheck_dir = osp.join(cwd, "run_precheck")
        if osp.exists(precheck_dir):
            shutil.rmtree(precheck_dir)

        if result.returncode != 0:
            stderr_output = _stderr_tail(result.stderr or "")
            return result.returncode, f"Precheck failed with the following error {stderr_output}"
        return 0, ""
    except TimeoutExpired:
        precheck_dir = osp.join(cwd, "run_precheck")
        if osp.exists(precheck_dir):
            shutil.rmtree(precheck_dir)
        return 1, f"Precheck timed out after {timeout} seconds"


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=7200):
    cwd = osp.abspath(folder_name)
    # COPY CODE SO WE CAN SEE IT.
    shutil.copy(
        osp.join(folder_name, "experiment.py"),
        osp.join(folder_name, f"run_{run_num}.py"),
    )

    static_return_code, static_prompt = run_static_checks(folder_name)
    if static_return_code != 0:
        return static_return_code, static_prompt

    task_context = _load_task_context(folder_name)
    if task_context.get("template") == "dental_cls_v1":
        precheck_return_code, precheck_prompt = run_precheck(folder_name)
        if precheck_return_code != 0:
            return precheck_return_code, precheck_prompt

    # LAUNCH COMMAND
    command = [
        "python",
        "experiment.py",
        f"--out_dir=run_{run_num}",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            stderr_output = _stderr_tail(result.stderr or "")
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            next_prompt = f"""Run {run_num} completed. Here are the results:
{results}

Review the result and decide whether you should continue, refine the plan, or stop.
Prefer a small number of interpretable comparisons that strengthen the dental research story.

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.

Then implement the next step on your plan if more work is needed.
We will next run `python experiment.py --out_dir=run_{run_num + 1}`.
YOUR PROPOSED CHANGE MUST SUPPORT THIS EXACT COMMAND FORMAT. DO NOT ADD REQUIRED EXTRA COMMAND LINE ARGS.
If the experimental story is complete, respond with 'ALL_COMPLETED'."""
        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt


# RUN PLOTTING
def run_plotting(folder_name, timeout=600):
    cwd = osp.abspath(folder_name)
    # LAUNCH COMMAND
    command = [
        "python",
        "plot.py",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Plotting failed with return code {result.returncode}")
            next_prompt = f"Plotting failed with the following error {result.stderr}"
        else:
            next_prompt = ""
        return result.returncode, next_prompt
    except TimeoutExpired:
        print(f"Plotting timed out after {timeout} seconds")
        next_prompt = f"Plotting timed out after {timeout} seconds"
        return 1, next_prompt


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    task_context = _load_task_context(folder_name)
    task_context_block = ""
    if task_context:
        task_context_block = f"""

Structured task context:
{json.dumps(task_context, indent=2, ensure_ascii=False)}
"""
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    ) + task_context_block
    while run < MAX_RUNS + 1:
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break
        coder_out = coder.run(next_prompt)
        print(coder_out)
        if "ALL_COMPLETED" in coder_out:
            break
        return_code, next_prompt = run_experiment(folder_name, run)
        if return_code == 0:
            run += 1
            current_iter = 0
        current_iter += 1
    if current_iter >= MAX_ITERS:
        print("Not all experiments completed.")
        return False

    current_iter = 0
    next_prompt = """
Please modify `plot.py` to generate the most useful plots for the final dental research writeup.

Prioritize figures that help compare the baseline and the most relevant experimental runs.
Only include runs that support the final story, and label them clearly so the figure can be understood without extra context.

We will run `python plot.py` to generate the plots.
"""
    while True:
        _ = coder.run(next_prompt)
        return_code, next_prompt = run_plotting(folder_name)
        current_iter += 1
        if return_code == 0 or current_iter >= MAX_ITERS:
            break
    next_prompt = """
Please update `notes.txt` with a clear description of each generated figure, including the filename and what scientific point the figure supports.

Write this for a later dental research writeup: be specific, faithful to the actual runs, and avoid claims not supported by the logs or figures.
"""
    coder.run(next_prompt)

    return True
