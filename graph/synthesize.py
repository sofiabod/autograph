"""auto-generate synthesis nodes by detecting patterns in the graph."""

from graph.db import MemgraphClient
from graph.models.nodes import Synthesis
from graph.models.edges import SynthesizedFrom, SupportsSynthesis
from graph.models.types import Category


def _detect_sweet_spots(client: MemgraphClient) -> list[Synthesis]:
    # same technique with both keeps and discards = non-monotonic relationship
    # e.g. weight decay 0.1 keeps, weight decay 0.2 discards → sweet spot
    syntheses = []

    techs = client.get_techniques()
    for t in techs:
        stats = client.get_technique_stats(t["name"])
        keeps = stats.get("keep", 0)
        discards = stats.get("discard", 0)

        if keeps >= 2 and discards >= 2:
            history = client.get_technique_history(t["name"])
            best_keep = min(
                [e for e in history if e["status"] == "keep"],
                key=lambda e: e["val_bpb"],
                default=None,
            )
            worst_discard = max(
                [e for e in history if e["status"] == "discard"],
                key=lambda e: e["val_bpb"],
                default=None,
            )

            if best_keep and worst_discard:
                confidence = min(keeps, discards) / (keeps + discards)
                syn = Synthesis(
                    text=f"{t['name']} has a sweet spot — {keeps} experiments kept, {discards} discarded. "
                         f"best keep: #{best_keep['experiment_id']} ({best_keep['val_bpb']:.6f})",
                    confidence=confidence,
                    domain="ml",
                    category=Category(t.get("category", "hyperparameter")),
                    auto_generated=True,
                    source_count=keeps + discards,
                )
                syntheses.append((syn, [e["experiment_id"] for e in history]))

    return syntheses


def _detect_productive_directions(client: MemgraphClient) -> list[Synthesis]:
    # 3+ sequential keeps in the same category = productive direction
    syntheses = []

    for cat in Category:
        history = client.get_category_history(cat.value)
        keeps = [e for e in history if e["status"] == "keep"]

        if len(keeps) >= 3:
            total = len(history)
            keep_rate = len(keeps) / total if total > 0 else 0
            best = min(keeps, key=lambda e: e["val_bpb"])

            syn = Synthesis(
                text=f"{cat.value} is a productive direction — {len(keeps)}/{total} experiments kept "
                     f"({keep_rate:.0%} success rate). best: #{best['experiment_id']} ({best['val_bpb']:.6f})",
                confidence=keep_rate,
                domain="ml",
                category=cat,
                auto_generated=True,
                source_count=len(keeps),
            )
            syntheses.append((syn, [e["experiment_id"] for e in keeps]))

    return syntheses


def _detect_dead_ends(client: MemgraphClient) -> list[Synthesis]:
    # 5+ consecutive discards in a category = dead end
    syntheses = []

    for cat in Category:
        history = client.get_category_history(cat.value)
        if len(history) < 5:
            continue

        # check last 5
        recent = history[-5:]
        if all(e["status"] == "discard" for e in recent):
            syn = Synthesis(
                text=f"{cat.value} appears exhausted — last {len(recent)} experiments all discarded. "
                     f"consider switching to a different category.",
                confidence=0.8,
                domain="ml",
                category=cat,
                auto_generated=True,
                source_count=len(recent),
            )
            syntheses.append((syn, [e["experiment_id"] for e in recent]))

    return syntheses


def _detect_crashes(client: MemgraphClient) -> list[Synthesis]:
    # multiple crashes with similar descriptions = hard constraint
    crashes = client._run("""
        MATCH (e:Experiment {status: 'crash'})
        RETURN e.experiment_id AS experiment_id, e.change_summary AS change_summary
    """)

    if len(crashes) >= 2:
        syn = Synthesis(
            text=f"{len(crashes)} experiments crashed. common causes: "
                 + ", ".join(c["change_summary"][:50] for c in crashes[:5]),
            confidence=0.9,
            domain="ml",
            category=Category.architecture,
            auto_generated=True,
            source_count=len(crashes),
        )
        return [(syn, [c["experiment_id"] for c in crashes])]

    return []


def _detect_compounding_techniques(client: MemgraphClient) -> list[Synthesis]:
    # techniques that appear together in the lineage of the best = they compound
    best = client.get_best()
    if not best:
        return []

    chain = client.get_experiment_chain(best["experiment_id"])
    if len(chain) < 3:
        return []

    # count which categories appear in the improvement chain
    categories = {}
    for exp in chain:
        cat = exp.get("category", "hyperparameter")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(exp)

    # categories that appear 2+ times in the lineage = compounding
    compounding = {cat: exps for cat, exps in categories.items() if len(exps) >= 2}
    if len(compounding) >= 2:
        cats = list(compounding.keys())
        syn = Synthesis(
            text=f"compounding effect: {' + '.join(cats)} improvements stack in the lineage of the best experiment. "
                 f"{len(chain)} total improvements in the chain.",
            confidence=0.7,
            domain="ml",
            category=Category.hyperparameter,
            auto_generated=True,
            source_count=len(chain),
        )
        return [(syn, [e["experiment_id"] for e in chain])]

    return []


def run_synthesis(client: MemgraphClient) -> int:
    """detect patterns in the graph and create synthesis nodes."""
    all_syntheses = []
    all_syntheses.extend(_detect_sweet_spots(client))
    all_syntheses.extend(_detect_productive_directions(client))
    all_syntheses.extend(_detect_dead_ends(client))
    all_syntheses.extend(_detect_crashes(client))
    all_syntheses.extend(_detect_compounding_techniques(client))

    created = 0
    for syn, experiment_ids in all_syntheses:
        # check if a similar synthesis already exists (avoid duplicates)
        existing = client._run(
            "MATCH (s:Synthesis) WHERE s.text CONTAINS $keyword RETURN s.id AS id LIMIT 1",
            {"keyword": syn.text[:50]},
        )
        if existing:
            continue

        client.create_synthesis(syn)
        created += 1

        # link to source experiments
        for eid in experiment_ids:
            exp = client.get_experiment(eid)
            if exp:
                client._run(
                    """
                    MATCH (s:Synthesis {id: $sid})
                    MATCH (e:Experiment {experiment_id: $eid})
                    CREATE (e)-[:SUPPORTS_SYNTHESIS]->(s)
                    """,
                    {"sid": str(syn.id), "eid": eid},
                )

    return created


if __name__ == "__main__":
    with MemgraphClient() as client:
        count = run_synthesis(client)
        print(f"created {count} synthesis nodes")
        print()
        for s in client.get_syntheses():
            print(f"  [{s['confidence']:.1f}] {s['text']}")
