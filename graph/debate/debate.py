import json
from datetime import UTC, datetime
from dataclasses import dataclass, field

from graph.db import MemgraphClient
from graph.models.nodes import Hypothesis, Agent
from graph.models.edges import Challenged, Refines, Tested
from graph.models.types import HypothesisStatus, Category
from graph.debate.config import DebateConfig
from graph.debate.agents import Proposer, Challenger
from graph.debate.prompts import PROPOSER_INITIAL, CHALLENGER_INITIAL, DECISION_PROMPT


@dataclass
class DebateRound:
    round_num: int
    proposal: str
    challenge: str
    rebuttal: str = ""
    assessment: str = ""
    # ids for graph backtracking
    hypothesis_id: str = ""
    challenge_edge_id: str = ""


@dataclass
class DebateResult:
    rounds: list[DebateRound] = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    raw_decision: str = ""
    # graph node ids for tracing
    proposer_agent_id: str = ""
    challenger_agent_id: str = ""
    hypothesis_ids: list[str] = field(default_factory=list)


def _guess_category(text: str) -> str:
    text = text.lower()
    keywords = {
        "regularization": ["weight decay", "wd", "dropout", "label smooth"],
        "normalization": ["norm", "rmsnorm"],
        "attention": ["attention", "head", "window", "rope", "qk", "gqa", "softcap"],
        "optimizer": ["muon", "adam", "momentum"],
        "schedule": ["warmdown", "warmup", "lr", "decay", "cooldown"],
        "architecture": ["depth", "dim", "mlp", "expansion", "token shift", "activation"],
        "training_loop": ["batch", "compile", "seed"],
    }
    for cat, words in keywords.items():
        if any(w in text for w in words):
            return cat
    return "hyperparameter"


def _write_agents_to_graph(client: MemgraphClient, config: DebateConfig) -> tuple[str, str]:
    # create or merge agent nodes for proposer and challenger
    proposer_agent = Agent(name=f"proposer:{config.proposer_model}")
    challenger_agent = Agent(name=f"challenger:{config.challenger_model}")
    pid = client.create_agent(proposer_agent)
    cid = client.create_agent(challenger_agent)
    return pid, cid


def _write_hypothesis_to_graph(
    client: MemgraphClient,
    text: str,
    category: str,
    status: HypothesisStatus = HypothesisStatus.pending,
) -> Hypothesis:
    # create hypothesis node and write to memgraph
    try:
        cat = Category(category)
    except ValueError:
        cat = Category.hyperparameter
    hyp = Hypothesis(
        text=text,
        status=status,
        category=cat,
        valid_from=datetime.now(UTC),
    )
    client.create_hypothesis(hyp)
    return hyp


def _write_challenge_to_graph(
    client: MemgraphClient,
    challenger_agent_id: str,
    hypothesis: Hypothesis,
    reason: str,
    round_num: int,
    evidence_cited: list[int] | None = None,
) -> str:
    # write the challenge edge: challenger -> hypothesis
    edge = Challenged(
        source=hypothesis.id,  # on the hypothesis
        target=hypothesis.id,  # self-referencing for now, challenger is in the reason
        reason=reason[:500],  # truncate to avoid huge edges
        round=round_num,
        evidence_cited=evidence_cited or [],
    )
    # use raw cypher to link challenger agent to hypothesis
    client._run(
        """
        MATCH (h:Hypothesis {id: $hid})
        MATCH (a:Agent {id: $aid})
        CREATE (a)-[r:CHALLENGED {reason: $reason, round: $round}]->(h)
        RETURN type(r) AS t
        """,
        {
            "hid": str(hypothesis.id),
            "aid": challenger_agent_id,
            "reason": reason[:500],
            "round": round_num,
        },
    )
    return str(edge.id)


def _write_refinement_to_graph(
    client: MemgraphClient,
    old_hypothesis: Hypothesis,
    new_hypothesis: Hypothesis,
    what_changed: str,
    why: str,
) -> None:
    # new hypothesis refines old one
    edge = Refines(
        source=new_hypothesis.id,
        target=old_hypothesis.id,
        what_changed=what_changed[:300],
        why=why[:300],
    )
    client.create_edge(edge)


def _update_hypothesis_status(
    client: MemgraphClient,
    hypothesis: Hypothesis,
    status: str,
) -> None:
    client._run(
        "MATCH (h:Hypothesis {id: $hid}) SET h.status = $status, h.valid_until = $now",
        {
            "hid": str(hypothesis.id),
            "status": status,
            "now": datetime.now(UTC).isoformat(),
        },
    )


def _write_debate_log_to_graph(
    client: MemgraphClient,
    debate_result: DebateResult,
) -> None:
    # write a DebateLog node summarizing the full debate for querying
    decision = debate_result.decision
    rounds_summary = []
    for r in debate_result.rounds:
        rounds_summary.append({
            "round": r.round_num,
            "proposal": r.proposal[:200],
            "challenge": r.challenge[:200],
            "rebuttal": r.rebuttal[:200],
            "assessment": r.assessment[:200],
        })

    client._run(
        """
        CREATE (d:DebateLog {
            id: $id,
            run_decision: $run,
            hypothesis: $hypothesis,
            change_summary: $change_summary,
            confidence: $confidence,
            reasoning: $reasoning,
            rounds: $rounds,
            created_at: $now
        })
        """,
        {
            "id": str(debate_result.hypothesis_ids[0]) if debate_result.hypothesis_ids else "",
            "run": decision.get("run", False),
            "hypothesis": decision.get("hypothesis", ""),
            "change_summary": decision.get("change_summary", ""),
            "confidence": decision.get("confidence", 0.0),
            "reasoning": decision.get("reasoning", ""),
            "rounds": json.dumps(rounds_summary),
            "now": datetime.now(UTC).isoformat(),
        },
    )


def _build_proposer_context(client: MemgraphClient) -> str:
    best = client.get_best()
    keeps = client.get_keeps()
    recent_keeps = keeps[:5] if keeps else []
    recent_keeps_str = "\n".join(
        f"  #{k['experiment_id']} val_bpb={k['val_bpb']:.6f} — {k['change_summary']}"
        for k in recent_keeps
    )

    all_exps = client.get_all_experiments()
    recent_failures = [e for e in reversed(all_exps) if e["status"] == "discard"][:10]
    recent_failures_str = "\n".join(
        f"  #{f['experiment_id']} val_bpb={f['val_bpb']:.6f} — {f['change_summary']}"
        for f in recent_failures
    )

    techs = client.get_techniques()
    tech_stats = []
    for t in techs:
        stats = client.get_technique_stats(t["name"])
        tech_stats.append(f"  {t['name']}: {stats}")
    tech_stats_str = "\n".join(tech_stats)

    contradictions = client.get_contradictions()
    contradictions_str = "\n".join(
        f"  {c['result_a']} vs {c['result_b']} — {c['explanation']}"
        for c in contradictions
    )

    best_id = best["experiment_id"] if best else 1
    lineage = client.get_experiment_chain(best_id)
    lineage_str = "\n".join(
        f"  #{l['experiment_id']} val_bpb={l['val_bpb']:.6f} — {l['change_summary']}"
        for l in lineage
    )

    best_str = f"#{best['experiment_id']} val_bpb={best['val_bpb']:.6f} — {best['change_summary']}" if best else "none"

    # previous debates so we don't repeat ourselves
    prev_debates = client._run("""
        MATCH (d:DebateLog)
        RETURN d.change_summary AS change, d.run_decision AS ran,
               d.confidence AS conf, d.reasoning AS reason
        ORDER BY d.created_at DESC
    """)
    prev_debates_str = "\n".join(
        f"  {'ran' if d['ran'] else 'skipped'} (conf={d['conf']:.1f}): {d['change']} — {d['reason'][:100]}"
        for d in prev_debates
    ) if prev_debates else "none"

    # rejected hypotheses — things the challenger killed
    rejected = client._run("""
        MATCH (h:Hypothesis {status: 'rejected'})
        RETURN h.text AS text
    """)
    rejected_str = "\n".join(
        f"  {r['text'][:100]}"
        for r in rejected
    ) if rejected else "none"

    return PROPOSER_INITIAL.format(
        best=best_str,
        recent_keeps=recent_keeps_str or "none",
        recent_failures=recent_failures_str or "none",
        technique_stats=tech_stats_str or "none",
        contradictions=contradictions_str or "none",
        lineage=lineage_str or "none",
        previous_debates=prev_debates_str,
        rejected_hypotheses=rejected_str,
    )


def _build_challenger_context(client: MemgraphClient, proposal: str) -> str:
    proposal_lower = proposal.lower()

    categories = ["attention", "optimizer", "schedule", "architecture", "regularization", "normalization"]
    category = "hyperparameter"
    for cat in categories:
        if cat in proposal_lower:
            category = cat
            break

    category_history = client.get_category_history(category)
    category_str = "\n".join(
        f"  #{e['experiment_id']} {e['status']} val_bpb={e['val_bpb']:.6f} — {e['change_summary']}"
        for e in category_history[:15]
    )

    techs = client.get_techniques()
    relevant_techs = [t for t in techs if t["name"] in proposal_lower]
    technique_str = ""
    for t in relevant_techs:
        history = client.get_technique_history(t["name"])
        technique_str += f"\n  {t['name']}:\n"
        for h in history[:10]:
            technique_str += f"    #{h['experiment_id']} {h['status']} val_bpb={h['val_bpb']:.6f} — {h['change_summary']}\n"

    failed = [e for e in category_history if e["status"] == "discard"][:10]
    failed_str = "\n".join(
        f"  #{f['experiment_id']} val_bpb={f['val_bpb']:.6f} — {f['change_summary']}"
        for f in failed
    )

    contradictions = client.get_contradictions()
    contradictions_str = "\n".join(
        f"  {c['result_a']} vs {c['result_b']} — {c['explanation']}"
        for c in contradictions
    )

    return CHALLENGER_INITIAL.format(
        proposal=proposal,
        category_history=category_str or "none",
        technique_history=technique_str or "none",
        failed_experiments=failed_str or "none",
        contradictions=contradictions_str or "none",
    )


def _extract_decision(proposer: Proposer, config: DebateConfig, debate: DebateResult) -> dict:
    last_round = debate.rounds[-1]
    prompt = DECISION_PROMPT.format(
        proposal=last_round.rebuttal or last_round.proposal,
        challenge=last_round.challenge,
        rebuttal=last_round.rebuttal,
        final_assessment=last_round.assessment,
    )

    raw = proposer.respond(prompt, 0.1)
    debate.raw_decision = raw

    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError):
        return {"run": False, "error": "failed to parse decision", "raw": raw}


def run_debate(client: MemgraphClient, config: DebateConfig | None = None) -> DebateResult:
    """run a full debate, writing every step to memgraph for backtracking."""
    if config is None:
        config = DebateConfig()

    proposer = Proposer(config)
    challenger = Challenger(config)
    result = DebateResult()

    # write agent nodes to graph
    pid, cid = _write_agents_to_graph(client, config)
    result.proposer_agent_id = pid
    result.challenger_agent_id = cid

    # round 1: proposer proposes
    proposer_context = _build_proposer_context(client)
    proposal = proposer.propose(proposer_context)

    # write initial hypothesis to graph
    category = _guess_category(proposal)
    hyp = _write_hypothesis_to_graph(client, proposal[:500], category)
    result.hypothesis_ids.append(str(hyp.id))

    # challenger challenges
    challenger_context = _build_challenger_context(client, proposal)
    challenge = challenger.challenge(challenger_context)

    # write challenge edge to graph
    challenge_eid = _write_challenge_to_graph(client, cid, hyp, challenge, round_num=1)

    round1 = DebateRound(
        round_num=1, proposal=proposal, challenge=challenge,
        hypothesis_id=str(hyp.id), challenge_edge_id=challenge_eid,
    )

    # proposer rebuts
    rebuttal = proposer.rebut(challenge)
    round1.rebuttal = rebuttal

    # if proposer modified their proposal, create a refined hypothesis
    if "modif" in rebuttal.lower() or "updat" in rebuttal.lower() or "revise" in rebuttal.lower():
        new_hyp = _write_hypothesis_to_graph(client, rebuttal[:500], category)
        _write_refinement_to_graph(client, hyp, new_hyp, "modified after challenge", challenge[:200])
        result.hypothesis_ids.append(str(new_hyp.id))
        hyp = new_hyp

    # challenger final assessment
    assessment = challenger.reassess(rebuttal)
    round1.assessment = assessment

    result.rounds.append(round1)

    # additional rounds
    for i in range(1, config.max_rounds):
        new_rebuttal = proposer.rebut(assessment)

        # write refinement if proposal changed
        if "modif" in new_rebuttal.lower() or "updat" in new_rebuttal.lower():
            new_hyp = _write_hypothesis_to_graph(client, new_rebuttal[:500], category)
            _write_refinement_to_graph(client, hyp, new_hyp, "refined in round " + str(i + 1), assessment[:200])
            result.hypothesis_ids.append(str(new_hyp.id))
            hyp = new_hyp

        new_assessment = challenger.reassess(new_rebuttal)

        # write challenge for this round
        _write_challenge_to_graph(client, cid, hyp, new_assessment, round_num=i + 1)

        round_n = DebateRound(
            round_num=i + 1,
            proposal=new_rebuttal,
            challenge=new_assessment,
            rebuttal=new_rebuttal,
            assessment=new_assessment,
            hypothesis_id=str(hyp.id),
        )
        result.rounds.append(round_n)

        if "APPROVE" in new_assessment.upper():
            _update_hypothesis_status(client, hyp, "confirmed")
            break

        if "REJECT" in new_assessment.upper():
            _update_hypothesis_status(client, hyp, "rejected")
            break

        assessment = new_assessment

    # extract structured decision
    result.decision = _extract_decision(proposer, config, result)

    # update final hypothesis status based on decision
    final_status = "confirmed" if result.decision.get("run", False) else "rejected"
    _update_hypothesis_status(client, hyp, final_status)

    # write full debate log to graph
    _write_debate_log_to_graph(client, result)

    return result


if __name__ == "__main__":
    from graph.debate.config import DebateConfig

    config = DebateConfig()
    if not config.api_key:
        print("set OPENROUTER_API_KEY env var")
        exit(1)

    with MemgraphClient() as client:
        print("starting debate...\n")
        result = run_debate(client, config)

        for r in result.rounds:
            print(f"=== round {r.round_num} ===")
            print(f"\nproposer:\n{r.proposal[:300]}...")
            print(f"\nchallenger:\n{r.challenge[:300]}...")
            if r.rebuttal:
                print(f"\nrebuttal:\n{r.rebuttal[:300]}...")
            if r.assessment:
                print(f"\nassessment:\n{r.assessment[:300]}...")
            print()

        print("=== decision ===")
        print(json.dumps(result.decision, indent=2))
