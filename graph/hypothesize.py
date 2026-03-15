"""auto-generate hypotheses from graph patterns and synthesis nodes."""

from graph.db import MemgraphClient
from graph.models.nodes import Hypothesis
from graph.models.edges import MotivatedBy
from graph.models.types import HypothesisStatus, Category


def _combine_winners(client: MemgraphClient) -> list[tuple[Hypothesis, str]]:
    # two keeps in different categories that were never tried together
    keeps = client.get_keeps()
    if len(keeps) < 2:
        return []

    hypotheses = []
    seen_pairs = set()

    # get the top 5 keeps
    top_keeps = keeps[:5]
    for i, a in enumerate(top_keeps):
        for b in top_keeps[i + 1:]:
            if a.get("category") == b.get("category"):
                continue

            pair_key = tuple(sorted([a["experiment_id"], b["experiment_id"]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # check if a combination was already tried (search for both keywords)
            a_words = a["change_summary"].lower().split()[:3]
            b_words = b["change_summary"].lower().split()[:3]
            combined_search = " ".join(a_words)
            existing = client.search_experiments(combined_search)

            # if fewer than 3 matches, this combo is probably novel
            if len(existing) < 3:
                hyp = Hypothesis(
                    text=f"combine #{a['experiment_id']} ({a['change_summary']}) "
                         f"with #{b['experiment_id']} ({b['change_summary']}) — "
                         f"both improved independently, might compound",
                    status=HypothesisStatus.pending,
                    category=Category(a.get("category", "hyperparameter")),
                    auto_generated=True,
                )
                # link to a synthesis if one exists for either category
                synth_id = _find_synthesis_for_category(client, a.get("category", ""))
                hypotheses.append((hyp, synth_id))

    return hypotheses[:3]  # cap at 3 to avoid flooding


def _retry_crashes(client: MemgraphClient) -> list[tuple[Hypothesis, str]]:
    # crashed experiments that might work with less vram
    crashes = client._run("""
        MATCH (e:Experiment {status: 'crash'})
        RETURN e.experiment_id AS eid, e.change_summary AS cs
    """)

    hypotheses = []
    for crash in crashes[:3]:
        desc = crash["cs"].lower()
        # skip if it already mentions reducing batch/size
        if "reduce" in desc or "smaller" in desc or "half" in desc:
            continue

        hyp = Hypothesis(
            text=f"retry #{crash['eid']} ({crash['cs']}) but with device_batch_size=64 "
                 f"or smaller model dim to avoid OOM",
            status=HypothesisStatus.pending,
            category=Category.architecture,
            auto_generated=True,
        )
        synth_id = _find_synthesis_for_text(client, "crashed")
        hypotheses.append((hyp, synth_id))

    return hypotheses


def _cross_category(client: MemgraphClient) -> list[tuple[Hypothesis, str]]:
    # insight from productive category applied to exhausted one
    syntheses = client.get_syntheses()

    productive = [s for s in syntheses if "productive" in s["text"]]
    exhausted = [s for s in syntheses if "exhausted" in s["text"]]

    if not productive or not exhausted:
        return []

    hypotheses = []
    for prod in productive[:2]:
        for dead in exhausted[:2]:
            prod_cat = prod.get("category", "?")
            dead_cat = dead.get("category", "?")
            if prod_cat == dead_cat:
                continue

            hyp = Hypothesis(
                text=f"{prod_cat} techniques are still working ({prod['text'][:80]}). "
                     f"try applying a {prod_cat} approach to the {dead_cat} domain "
                     f"which is currently exhausted",
                status=HypothesisStatus.pending,
                category=Category(prod_cat) if prod_cat in [c.value for c in Category] else Category.hyperparameter,
                auto_generated=True,
            )
            hypotheses.append((hyp, prod.get("id", "")))

    return hypotheses[:2]


def _diminishing_returns_reversal(client: MemgraphClient) -> list[tuple[Hypothesis, str]]:
    # technique that stopped working might work again after architecture changed
    # find techniques that worked early but failed recently
    techs = client.get_techniques()
    hypotheses = []

    for t in techs:
        history = client.get_technique_history(t["name"])
        if len(history) < 4:
            continue

        # split into early and late
        mid = len(history) // 2
        early = history[:mid]
        late = history[mid:]

        early_keeps = [e for e in early if e["status"] == "keep"]
        late_keeps = [e for e in late if e["status"] == "keep"]
        late_discards = [e for e in late if e["status"] == "discard"]

        # worked early, stopped working late
        if len(early_keeps) >= 1 and len(late_keeps) == 0 and len(late_discards) >= 2:
            best_early = min(early_keeps, key=lambda e: e["val_bpb"])
            hyp = Hypothesis(
                text=f"{t['name']} worked early (#{best_early['experiment_id']}, "
                     f"val_bpb={best_early['val_bpb']:.6f}) but failed in last "
                     f"{len(late_discards)} attempts. the architecture has changed since then — "
                     f"retry with current config",
                status=HypothesisStatus.pending,
                category=Category(t.get("category", "hyperparameter")),
                auto_generated=True,
            )
            synth_id = _find_synthesis_for_text(client, t["name"])
            hypotheses.append((hyp, synth_id))

    return hypotheses[:2]


def _find_synthesis_for_category(client: MemgraphClient, category: str) -> str:
    rows = client._run(
        "MATCH (s:Synthesis {category: $cat}) RETURN s.id AS id LIMIT 1",
        {"cat": category},
    )
    return rows[0]["id"] if rows else ""


def _find_synthesis_for_text(client: MemgraphClient, keyword: str) -> str:
    rows = client._run(
        "MATCH (s:Synthesis) WHERE s.text CONTAINS $kw RETURN s.id AS id LIMIT 1",
        {"kw": keyword},
    )
    return rows[0]["id"] if rows else ""


def run_hypothesize(client: MemgraphClient) -> int:
    """generate hypotheses from graph patterns."""
    all_hyps = []
    all_hyps.extend(_combine_winners(client))
    all_hyps.extend(_retry_crashes(client))
    all_hyps.extend(_cross_category(client))
    all_hyps.extend(_diminishing_returns_reversal(client))

    created = 0
    for hyp, synth_id in all_hyps:
        # skip if similar hypothesis already exists
        existing = client._run(
            "MATCH (h:Hypothesis) WHERE h.text CONTAINS $kw RETURN h.id AS id LIMIT 1",
            {"kw": hyp.text[:40]},
        )
        if existing:
            continue

        client.create_hypothesis(hyp)
        created += 1

        # link to synthesis that motivated it
        if synth_id:
            client._run(
                """
                MATCH (h:Hypothesis {id: $hid})
                MATCH (s:Synthesis {id: $sid})
                CREATE (h)-[:MOTIVATED_BY]->(s)
                """,
                {"hid": str(hyp.id), "sid": synth_id},
            )

    return created


if __name__ == "__main__":
    from graph.synthesize import run_synthesis

    with MemgraphClient() as client:
        # run synthesis first so hypotheses have something to work with
        syn_count = run_synthesis(client)
        print(f"syntheses: {syn_count} new")

        hyp_count = run_hypothesize(client)
        print(f"hypotheses: {hyp_count} new")
        print()
        for h in client.get_hypothesis_status("pending"):
            print(f"  {h['text'][:100]}...")
