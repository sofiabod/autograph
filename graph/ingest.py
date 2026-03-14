import csv
from pathlib import Path

from graph.db import MemgraphClient
from graph.models.nodes import Experiment, Technique
from graph.models.edges import ImprovedFrom, FailedFrom, Tried
from graph.models.types import Status, Category


# ordered  first match wins, so more specific terms come first
_CATEGORY_KEYWORDS: list[tuple[Category, list[str]]] = [
    (Category.regularization, ["weight decay", "wd ", "dropout", "regulariz", "label smooth", "z-loss"]),
    (Category.normalization, ["norm", "rmsnorm", "layernorm"]),
    (Category.attention, ["attention", "head", "window", "rope", "qk", "gqa", "softcap"]),
    (Category.optimizer, ["muon", "adam", "momentum", "beta"]),
    (Category.schedule, ["warmdown", "warmup", " lr ", "decay", "schedule", "cooldown", "lr_frac"]),
    (Category.architecture, ["depth", "dim", "aspect", "mlp", "expansion", "parallel", "layer", "token shift", "activation", "swiglu", "relu", "gelu", "silu"]),
    (Category.training_loop, ["batch", "compile", "seed", "ema"]),
]


def _guess_category(description: str) -> Category:
    """best-effort category from description text."""
    desc = description.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(w in desc for w in keywords):
            return category
    return Category.hyperparameter


def _extract_techniques(description: str) -> list[str]:
    """pull technique names from description."""
    techniques = []
    keywords = {
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
    desc = description.lower()
    for keyword, name in keywords.items():
        if keyword in desc and name not in techniques:
            techniques.append(name)
    return techniques


def ingest_tsv(client: MemgraphClient, path: str | Path):
    """load results.tsv into memgraph."""
    path = Path(path)
    experiments = {}

    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            status_str = row["status"].strip()
            val_bpb = float(row["val_bpb"])
            memory_gb = float(row["memory_gb"])
            description = row["description"].strip()

            exp = Experiment(
                experiment_id=i + 1,
                commit=row["commit"].strip(),
                val_bpb=val_bpb,
                memory_gb=memory_gb,
                status=Status(status_str),
                change_summary=description,
                category=_guess_category(description),
            )

            client.create_experiment(exp)
            experiments[i + 1] = exp

            # create technique nodes + tried edges
            for tech_name in _extract_techniques(description):
                tech = Technique(name=tech_name, category=exp.category)
                client.create_technique(tech)
                client.create_edge(Tried(source=exp.id, target=tech.id))

    # build parent edges (sequential lineage)
    prev_keep = None
    for eid in sorted(experiments.keys()):
        exp = experiments[eid]
        if prev_keep is not None and eid > 1:
            parent = prev_keep
            delta = exp.val_bpb - parent.val_bpb

            if exp.status == Status.keep and delta < 0:
                client.create_edge(ImprovedFrom(
                    source=exp.id,
                    target=parent.id,
                    delta_val_bpb=delta,
                ))
            elif exp.status in (Status.discard, Status.crash):
                client.create_edge(FailedFrom(
                    source=exp.id,
                    target=parent.id,
                    delta_val_bpb=delta,
                ))

        if exp.status == Status.keep:
            prev_keep = exp

    return len(experiments)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/baseline/results.tsv"
    with MemgraphClient() as client:
        client.create_indexes()
        client.drop_all()
        count = ingest_tsv(client, path)
        print(f"ingested {count} experiments")
        print(f"nodes: {client.count_nodes()}")
        best = client.get_best()
        if best:
            print(f"best: {best['change_summary']} ({best['val_bpb']})")
