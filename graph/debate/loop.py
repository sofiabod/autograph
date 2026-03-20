"""autonomous experiment loop: debate → run → record → repeat."""

import subprocess
import re
from datetime import UTC, datetime
from pathlib import Path

from graph.db import MemgraphClient
from uuid import UUID

from graph.models.nodes import Experiment, Result, Run
from graph.models.edges import Produced, ImprovedFrom, FailedFrom, PartOf
from graph.models.types import Status, Category
from graph.debate.config import DebateConfig
from graph.debate.debate import run_debate
from graph.ingest import _guess_category


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = REPO_ROOT / "train.py"
RESULTS_TSV = REPO_ROOT / "results" / "baseline" / "results.tsv"
RUN_LOG = REPO_ROOT / "run.log"

# hyperparameter names that are safe to auto-modify in train.py
ALLOWED_PARAMS = {
    "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN",
    "TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR",
    "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS",
    "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
    "DEPTH", "DEVICE_BATCH_SIZE",
}


def _apply_parameters(parameters_changed: dict) -> list[str]:
    """auto-modify train.py hyperparameters based on debate decision.
    returns list of changes applied, or raises if something looks wrong."""
    if not parameters_changed:
        return []

    # params that must be quoted strings in python
    STRING_PARAMS = {"WINDOW_PATTERN"}

    text = TRAIN_PY.read_text()
    changes = []

    for param, values in parameters_changed.items():
        param_upper = param.upper()
        if param_upper not in ALLOWED_PARAMS:
            print(f"  skipping {param} — not in allowed params")
            continue

        new_val = str(values.get("to", ""))
        if not new_val:
            continue

        # validate: must be a valid python literal (no prose, no sentences)
        try:
            compile(f"x = {new_val}", "<check>", "exec")
        except SyntaxError:
            # try quoting it if it's a string param
            if param_upper in STRING_PARAMS:
                new_val = f'"{new_val}"'
                try:
                    compile(f"x = {new_val}", "<check>", "exec")
                except SyntaxError:
                    print(f"  skipping {param_upper} — invalid value: {new_val}")
                    continue
            else:
                print(f"  skipping {param_upper} — invalid value: {new_val}")
                continue

        # preserve quotes on string params if the LLM forgot them
        if param_upper in STRING_PARAMS and not (new_val.startswith('"') or new_val.startswith("'")):
            new_val = f'"{new_val}"'

        # match lines like: MATRIX_LR = 0.04  or DEPTH = 8  (with optional comment)
        pattern = rf'^({param_upper}\s*=\s*)(.+?)(\s*#.*)?$'
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            print(f"  skipping {param_upper} — not found in train.py")
            continue

        old_val = match.group(2).strip()
        old_line = match.group(0)
        comment = match.group(3) or ""
        new_line = f"{match.group(1)}{new_val}{comment}"
        text = text.replace(old_line, new_line, 1)
        changes.append(f"{param_upper}: {old_val} -> {new_val}")
        print(f"  {param_upper}: {old_val} -> {new_val}")

    if changes:
        TRAIN_PY.write_text(text)

    return changes


def _parse_run_log(log_path: Path) -> dict | None:
    """extract val_bpb and peak_vram_mb from run.log."""
    if not log_path.exists():
        return None

    text = log_path.read_text()
    result = {}

    val_match = re.search(r"^val_bpb:\s+([\d.]+)", text, re.MULTILINE)
    if val_match:
        result["val_bpb"] = float(val_match.group(1))

    vram_match = re.search(r"^peak_vram_mb:\s+([\d.]+)", text, re.MULTILINE)
    if vram_match:
        result["peak_vram_mb"] = float(vram_match.group(1))
        result["memory_gb"] = round(result["peak_vram_mb"] / 1024, 1)

    steps_match = re.search(r"^num_steps:\s+(\d+)", text, re.MULTILINE)
    if steps_match:
        result["training_steps"] = int(steps_match.group(1))

    mfu_match = re.search(r"^mfu_percent:\s+([\d.]+)", text, re.MULTILINE)
    if mfu_match:
        result["mfu_percent"] = float(mfu_match.group(1))

    tokens_match = re.search(r"^total_tokens_M:\s+([\d.]+)", text, re.MULTILINE)
    if tokens_match:
        result["tokens_trained_m"] = float(tokens_match.group(1))

    return result if "val_bpb" in result else None


def _get_current_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    return r.stdout.strip()


def _get_next_experiment_id(client: MemgraphClient) -> int:
    rows = client._run("MATCH (e:Experiment) RETURN max(e.experiment_id) AS max_id")
    current = rows[0]["max_id"] if rows and rows[0]["max_id"] is not None else 0
    return current + 1


def _record_result(
    client: MemgraphClient,
    decision: dict,
    run_result: dict | None,
    hypothesis_id: str | None,
    prev_best_bpb: float,
    run_id: str | None = None,
) -> tuple[Experiment, Status]:
    """write experiment result back to memgraph and results.tsv."""
    exp_id = _get_next_experiment_id(client)
    commit = _get_current_commit()
    crashed = run_result is None

    # capture best BEFORE creating the new experiment so lineage edge
    # doesn't accidentally point to the experiment we're about to create
    prev_best = client.get_best()

    if crashed:
        val_bpb = 0.0
        memory_gb = 0.0
        status = Status.crash
    else:
        val_bpb = run_result["val_bpb"]
        memory_gb = run_result.get("memory_gb", 0.0)
        status = Status.keep if val_bpb < prev_best_bpb else Status.discard

    category = _guess_category(decision.get("change_summary", ""))
    change_summary = decision.get("change_summary", "unknown change")

    # create experiment node
    exp = Experiment(
        experiment_id=exp_id,
        commit=commit,
        val_bpb=val_bpb,
        baseline_bpb=prev_best_bpb,
        delta_bpb=val_bpb - prev_best_bpb if not crashed else 0.0,
        memory_gb=memory_gb,
        peak_vram_mb=run_result.get("peak_vram_mb", 0.0) if run_result else 0.0,
        status=status,
        change_summary=change_summary,
        category=Category(category) if category in [c.value for c in Category] else Category.hyperparameter,
        hypothesis=decision.get("hypothesis", ""),
        reasoning=decision.get("reasoning", ""),
        training_steps=run_result.get("training_steps", 0) if run_result else 0,
        mfu_percent=run_result.get("mfu_percent", 0.0) if run_result else 0.0,
        tokens_trained_m=run_result.get("tokens_trained_m", 0.0) if run_result else 0.0,
    )
    client.create_experiment(exp)

    # link experiment to run session
    if run_id:
        client.create_edge(PartOf(source=exp.id, target=UUID(run_id)))

    # create result node
    if crashed:
        result_text = f"{change_summary} crashed"
    elif status == Status.keep:
        result_text = f"{change_summary} improved val_bpb to {val_bpb:.6f}"
    else:
        result_text = f"{change_summary} did not help (val_bpb {val_bpb:.6f})"

    result = Result(
        text=result_text,
        val_bpb=val_bpb,
        delta=val_bpb - prev_best_bpb if not crashed else 0.0,
        kept=status == Status.keep,
        category=exp.category,
        valid_from=datetime.now(UTC),
    )
    client.create_result(result)
    client.create_edge(Produced(source=exp.id, target=result.id))

    # link to hypothesis if we have one
    if hypothesis_id:
        try:
            client._run(
                """
                MATCH (e:Experiment {id: $eid})
                MATCH (h:Hypothesis {id: $hid})
                CREATE (e)-[:TESTED]->(h)
                """,
                {"eid": str(exp.id), "hid": hypothesis_id},
            )
        except Exception:
            pass

    # create lineage edge to previous best (captured before insert)
    if prev_best and prev_best["id"] != str(exp.id):
        if status == Status.keep:
            client.create_edge(ImprovedFrom(
                source=exp.id, target=UUID(prev_best["id"]),
                delta_val_bpb=val_bpb - prev_best_bpb,
            ))
        elif status in (Status.discard, Status.crash):
            client.create_edge(FailedFrom(
                source=exp.id, target=UUID(prev_best["id"]),
                delta_val_bpb=val_bpb - prev_best_bpb if not crashed else 0.0,
            ))

    # append to results.tsv
    if RESULTS_TSV.exists():
        with open(RESULTS_TSV, "a") as f:
            f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb}\t{status.value}\t{change_summary}\n")

    # update hypothesis status
    if hypothesis_id:
        new_status = "confirmed" if status == Status.keep else "rejected"
        client._run(
            "MATCH (h:Hypothesis {id: $hid}) SET h.status = $status, h.valid_until = $now",
            {"hid": hypothesis_id, "status": new_status, "now": datetime.now(UTC).isoformat()},
        )

    print(f"recorded experiment #{exp_id}: {status.value} val_bpb={val_bpb:.6f} — {change_summary}")
    return exp, status


def run_experiment(decision: dict) -> dict | None:
    """run modal, return parsed results or None if crashed."""
    print(f"running experiment: {decision.get('change_summary', '?')}...")

    try:
        r = subprocess.run(
            ["modal", "run", "run_modal.py"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=600,
        )

        # write log
        with open(RUN_LOG, "w") as f:
            f.write(r.stdout)
            if r.stderr:
                f.write(r.stderr)

        result = _parse_run_log(RUN_LOG)
        if result:
            print(f"  val_bpb={result['val_bpb']:.6f} memory={result.get('memory_gb', '?')}gb")
        else:
            print("  crashed — no val_bpb in output")
            # show last 10 lines of log
            lines = (r.stdout + r.stderr).strip().splitlines()
            for line in lines[-10:]:
                print(f"  > {line}")

        return result

    except subprocess.TimeoutExpired:
        print("  timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"  error: {e}")
        return None


def run_loop(max_iterations: int = 100, config: DebateConfig | None = None):
    """the full autonomous loop: debate → run → record → repeat."""
    if config is None:
        config = DebateConfig()

    program_md = REPO_ROOT / "program.md"

    with MemgraphClient() as client:
        # create a run node to group this session's experiments
        run = Run(name=f"debate-loop-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}")
        run_id = client.create_run(run)
        print(f"started run: {run.name}")

        for i in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"iteration {i + 1}/{max_iterations}")
            print(f"{'='*60}\n")

            # get current best
            best = client.get_best()
            prev_best_bpb = best["val_bpb"] if best else 1.0
            print(f"current best: {prev_best_bpb:.6f}")

            # run debate
            print("\nstarting debate...")
            debate_result = run_debate(client, config)
            decision = debate_result.decision

            print(f"\ndecision: run={decision.get('run', False)} confidence={decision.get('confidence', 0)}")
            print(f"  {decision.get('change_summary', '?')}")

            if not decision.get("run", False):
                print("debate rejected this experiment, trying again...")
                continue

            # get the hypothesis id for linking
            hypothesis_id = debate_result.hypothesis_ids[-1] if debate_result.hypothesis_ids else None

            # auto-modify train.py based on debate decision
            params = decision.get("parameters_changed", {})
            print(f"\napplying changes: {decision.get('change_summary', '?')}")
            changes = _apply_parameters(params)

            if not changes:
                print("no parameter changes to apply, skipping experiment")
                continue

            # commit the change
            subprocess.run(
                ["git", "add", "train.py"],
                cwd=REPO_ROOT,
            )
            subprocess.run(
                ["git", "commit", "-m", f"experiment {_get_next_experiment_id(client)}: {decision.get('change_summary', 'unknown')}"],
                cwd=REPO_ROOT,
            )

            # run the experiment
            run_result = run_experiment(decision)

            # record result to graph + tsv
            _, status = _record_result(client, decision, run_result, hypothesis_id, prev_best_bpb, run_id)

            # if it didn't improve, revert
            if status != Status.keep:
                print("reverting train.py...")
                subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=REPO_ROOT)
                # restore program.md which gets wiped by reset
                if not program_md.exists():
                    subprocess.run(["git", "checkout", "master", "--", "program.md"], cwd=REPO_ROOT)

            print(f"\niteration {i + 1} complete: {status.value}")

    print("\nloop finished")


if __name__ == "__main__":
    config = DebateConfig()
    if not config.api_key:
        print("set OPENROUTER_API_KEY in .env")
        exit(1)

    import sys
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_loop(max_iterations=max_iter, config=config)
