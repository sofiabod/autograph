import json
from pathlib import Path
from uuid import UUID

from expgraph.nodes import Experiment, Technique
from expgraph.edges import ImprovedFrom, FailedFrom, LedTo, Tried
from expgraph.types import Category, Status


def _parse_category(raw: str) -> Category:
    """best-effort category parse, falls back to training_loop."""
    try:
        return Category(raw)
    except ValueError:
        return Category.training_loop


def _technique_key(name: str, user_id: str) -> str:
    return f"{user_id}:{name.lower().strip()}"


def ingest_jsonl(path: str | Path, user_id: str = "default") -> dict:
    """read experiment_log.jsonl, return {nodes, links} for react-force-graph."""
    path = Path(path)
    experiments: dict[int, Experiment] = {}
    techniques: dict[str, Technique] = {}
    links: list = []

    # parse all lines once
    lines = path.read_text().strip().splitlines()
    raw_entries = [json.loads(line) for line in lines]

    # pass 1: build experiment nodes
    for raw in raw_entries:
        exp = Experiment(
            experiment_id=raw["experiment_id"],
            commit=raw["commit"],
            user_id=user_id,
            val_bpb=raw["val_bpb"],
            baseline_bpb=raw["baseline_bpb"],
            delta_bpb=raw["delta_bpb"],
            memory_gb=raw["memory_gb"],
            status=raw["status"],
            hypothesis=raw.get("hypothesis", ""),
            change_summary=raw.get("change_summary", ""),
            category=_parse_category(raw.get("category", "")),
            reasoning=raw.get("reasoning", ""),
            insights=raw.get("insights", []),
            tags=raw.get("tags", []),
            parent_id=raw.get("parent_id"),
            builds_on=raw.get("builds_on", []),
            contradicts=raw.get("contradicts", []),
            supports=raw.get("supports", []),
            components=raw.get("components", []),
            parameters_changed=raw.get("parameters_changed", {}),
            timestamp=raw.get("timestamp"),
        )
        experiments[exp.experiment_id] = exp

        # extract techniques from components + tags
        for name in set(raw.get("components", []) + raw.get("tags", [])):
            key = _technique_key(name, user_id)
            if key not in techniques:
                techniques[key] = Technique(
                    name=name.lower().strip(),
                    user_id=user_id,
                    category=exp.category,
                )

    # pass 2: build edges
    for eid, exp in experiments.items():
        parent_id = exp.parent_id
        if parent_id is not None and parent_id != eid and parent_id in experiments:
            parent = experiments[parent_id]

            if exp.status == Status.keep and exp.delta_bpb < 0:
                links.append(ImprovedFrom(
                    source=exp.id,
                    target=parent.id,
                    user_id=user_id,
                    delta_val_bpb=exp.delta_bpb,
                ))
            elif exp.status in (Status.discard, Status.crash):
                links.append(FailedFrom(
                    source=exp.id,
                    target=parent.id,
                    user_id=user_id,
                    delta_val_bpb=exp.delta_bpb,
                ))

        # led_to: connect to the next experiment in sequence
        next_eid = eid + 1
        if next_eid in experiments and eid < len(raw_entries):
            next_idea = raw_entries[eid].get("next_idea", "")
            if next_idea:
                links.append(LedTo(
                    source=exp.id,
                    target=experiments[next_eid].id,
                    user_id=user_id,
                    rationale=exp.reasoning,
                    next_idea=next_idea,
                ))

        # tried: connect experiment to its techniques
        for name in set(exp.components + exp.tags):
            key = _technique_key(name, user_id)
            if key in techniques:
                links.append(Tried(
                    source=exp.id,
                    target=techniques[key].id,
                    user_id=user_id,
                ))

    # build output
    all_nodes = []
    for exp in experiments.values():
        all_nodes.append({
            "id": str(exp.id),
            "type": "experiment",
            "experiment_id": exp.experiment_id,
            "commit": exp.commit,
            "val_bpb": exp.val_bpb,
            "delta_bpb": exp.delta_bpb,
            "status": exp.status.value,
            "category": exp.category.value,
            "change_summary": exp.change_summary,
            "reasoning": exp.reasoning,
            "hypothesis": exp.hypothesis,
        })
    for tech in techniques.values():
        all_nodes.append({
            "id": str(tech.id),
            "type": "technique",
            "name": tech.name,
            "category": tech.category.value,
        })

    all_links = []
    for link in links:
        entry = {
            "source": str(link.source),
            "target": str(link.target),
            "edge_type": link.edge_type.value,
        }
        if hasattr(link, "delta_val_bpb"):
            entry["delta_val_bpb"] = link.delta_val_bpb
        if hasattr(link, "rationale"):
            entry["rationale"] = link.rationale
            entry["next_idea"] = link.next_idea
        all_links.append(entry)

    return {"nodes": all_nodes, "links": all_links}
