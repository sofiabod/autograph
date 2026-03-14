import csv
from pathlib import Path

from graph.db import MemgraphClient
from graph.models.nodes import Experiment, Technique, Result, Run
from graph.models.edges import ImprovedFrom, FailedFrom, Tried, Produced, PartOf, Contradicts
from graph.models.types import Status, Category


# ordered — first match wins, so more specific terms come first
_CATEGORY_KEYWORDS: list[tuple[Category, list[str]]] = [
    (Category.regularization, ["weight decay", "wd ", "dropout", "regulariz", "label smooth", "z-loss"]),
    (Category.normalization, ["norm", "rmsnorm", "layernorm"]),
    (Category.attention, ["attention", "head", "window", "rope", "qk", "gqa", "softcap"]),
    (Category.optimizer, ["muon", "adam", "momentum", "beta"]),
    (Category.schedule, ["warmdown", "warmup", " lr ", "decay", "schedule", "cooldown", "lr_frac"]),
    (Category.architecture, ["depth", "dim", "aspect", "mlp", "expansion", "parallel", "layer", "token shift", "activation", "swiglu", "relu", "gelu", "silu"]),
    (Category.training_loop, ["batch", "compile", "seed", "ema"]),
]

_TECHNIQUE_KEYWORDS: dict[str, str] = {
    "weight decay": "weight decay",
    "wd": "weight decay",
    "token shift": "token shifting",
    "rope": "rope",
    "qk norm": "qk norm",
    "softcap": "logit softcap",
    "value embedding": "value embeddings",
    "ve": "value embeddings",
    "muon": "muon optimizer",
    "swiglu": "swiglu",
    "gqa": "grouped query attention",
    "relu": "relu activation",
    "gelu": "gelu activation",
    "silu": "silu activation",
    "dropout": "dropout",
    "ema": "ema",
    "flash attention": "flash attention",
}


def _guess_category(description: str) -> Category:
    desc = description.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(w in desc for w in keywords):
            return category
    return Category.hyperparameter


def _extract_techniques(description: str) -> list[str]:
    techniques = []
    desc = description.lower()
    for keyword, name in _TECHNIQUE_KEYWORDS.items():
        if keyword in desc and name not in techniques:
            techniques.append(name)
    return techniques


def _generate_result_text(exp_data: dict) -> str:
    # turn an experiment into a human-readable finding
    status = exp_data["status"]
    desc = exp_data["description"]
    val = exp_data["val_bpb"]
    if status == "crash":
        return f"{desc} crashed (OOM or error)"
    elif status == "keep":
        return f"{desc} improved val_bpb to {val:.6f}"
    else:
        return f"{desc} did not help (val_bpb {val:.6f})"


def ingest_tsv(client: MemgraphClient, path: str | Path, run_name: str = "baseline"):
    # load results.tsv into memgraph with full schema
    path = Path(path)
    experiments = {}
    results = {}
    raw_rows = []

    # create the run node
    run = Run(name=run_name)

    # parse all rows first
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_rows.append({
                "commit": row["commit"].strip(),
                "val_bpb": float(row["val_bpb"]),
                "memory_gb": float(row["memory_gb"]),
                "status": row["status"].strip(),
                "description": row["description"].strip(),
            })

    # update run stats
    keeps = [r for r in raw_rows if r["status"] == "keep"]
    run.total_experiments = len(raw_rows)
    run.keep_count = len(keeps)
    run.best_val_bpb = min(r["val_bpb"] for r in keeps) if keeps else float("inf")
    client.create_run(run)

    # pass 1: create experiment nodes, result nodes, technique nodes
    for i, row in enumerate(raw_rows):
        category = _guess_category(row["description"])

        exp = Experiment(
            experiment_id=i + 1,
            commit=row["commit"],
            val_bpb=row["val_bpb"],
            memory_gb=row["memory_gb"],
            status=Status(row["status"]),
            change_summary=row["description"],
            category=category,
        )
        client.create_experiment(exp)
        experiments[i + 1] = exp

        # every experiment belongs to the run
        client.create_edge(PartOf(source=exp.id, target=run.id))

        # every experiment produces a result (what we learned)
        res = Result(
            text=_generate_result_text(row),
            val_bpb=row["val_bpb"],
            delta=0.0,  # computed in pass 2
            kept=row["status"] == "keep",
            category=category,
        )
        client.create_result(res)
        client.create_edge(Produced(source=exp.id, target=res.id))
        results[i + 1] = res

        # technique nodes + tried edges
        for tech_name in _extract_techniques(row["description"]):
            tech = Technique(name=tech_name, category=category)
            client.create_technique(tech)
            client.create_edge(Tried(source=exp.id, target=tech.id))

    # pass 2: lineage edges + contradiction detection
    prev_keep = None
    for eid in sorted(experiments.keys()):
        exp = experiments[eid]
        if prev_keep is not None and eid > 1:
            parent = prev_keep
            delta = exp.val_bpb - parent.val_bpb

            if exp.status == Status.keep and delta <= 0:
                client.create_edge(ImprovedFrom(
                    source=exp.id, target=parent.id, delta_val_bpb=delta,
                ))
            elif exp.status in (Status.discard, Status.crash):
                client.create_edge(FailedFrom(
                    source=exp.id, target=parent.id, delta_val_bpb=delta,
                ))

        if exp.status == Status.keep:
            prev_keep = exp

    # pass 3: detect contradictions between results
    # if the same technique was tried and both kept and discarded, those results contradict
    technique_results: dict[str, list[tuple[Result, bool]]] = {}
    for eid, exp in experiments.items():
        for tech_name in _extract_techniques(exp.change_summary):
            if tech_name not in technique_results:
                technique_results[tech_name] = []
            technique_results[tech_name].append((results[eid], exp.status == Status.keep))

    for tech_name, pairs in technique_results.items():
        kept = [r for r, k in pairs if k]
        discarded = [r for r, k in pairs if not k]
        # if a technique has both keeps and discards, the best keep contradicts the worst discard
        if kept and discarded:
            best_keep = min(kept, key=lambda r: r.val_bpb)
            worst_discard = max(discarded, key=lambda r: r.val_bpb)
            client.create_edge(Contradicts(
                source=best_keep.id,
                target=worst_discard.id,
                explanation=f"{tech_name}: worked sometimes but not always",
            ))

    return len(experiments)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/baseline/results.tsv"
    run_name = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    with MemgraphClient() as client:
        client.create_indexes()
        client.drop_all()
        count = ingest_tsv(client, path, run_name)
        print(f"ingested {count} experiments")
        print(f"nodes: {client.count_nodes()}")
        best = client.get_best()
        if best:
            print(f"best: {best['change_summary']} ({best['val_bpb']})")
        contradictions = client.get_contradictions()
        print(f"contradictions: {len(contradictions)}")
