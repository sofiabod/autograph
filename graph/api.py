from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from graph.db import MemgraphClient

app = FastAPI(title="expgraph-live")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client() -> MemgraphClient:
    return MemgraphClient()


@app.get("/graph")
def graph():
    # serves full graph for react-force-graph — reads from memgraph in real time
    c = _get_client()
    try:
        nodes = []
        links = []

        # experiments
        for e in c.get_all_experiments():
            nodes.append({
                "id": e["id"],
                "type": "experiment",
                "experiment_id": e["experiment_id"],
                "commit": e.get("commit", ""),
                "val_bpb": e["val_bpb"],
                "delta_bpb": e.get("delta_bpb", 0),
                "status": e["status"],
                "category": e.get("category", ""),
                "change_summary": e.get("change_summary", ""),
                "reasoning": e.get("reasoning", ""),
                "hypothesis": e.get("hypothesis", ""),
            })

        # techniques
        for t in c.get_techniques():
            nodes.append({
                "id": t["id"],
                "type": "technique",
                "name": t["name"],
                "category": t.get("category", ""),
            })

        # hypotheses (from debates)
        for h in c._run("MATCH (h:Hypothesis) RETURN h"):
            hp = dict(h["h"])
            nodes.append({
                "id": hp["id"],
                "type": "hypothesis",
                "text": hp.get("text", ""),
                "status": hp.get("status", "pending"),
            })

        # results
        for r in c._run("MATCH (r:Result) RETURN r LIMIT 500"):
            rp = dict(r["r"])
            nodes.append({
                "id": rp["id"],
                "type": "result",
                "text": rp.get("text", ""),
                "val_bpb": rp.get("val_bpb", 0),
                "kept": rp.get("kept", False),
            })

        # debate logs
        for d in c._run("MATCH (d:DebateLog) RETURN d"):
            dp = dict(d["d"])
            nodes.append({
                "id": dp.get("id", ""),
                "type": "debate",
                "change_summary": dp.get("change_summary", ""),
                "run_decision": dp.get("run_decision", False),
                "confidence": dp.get("confidence", 0),
                "reasoning": dp.get("reasoning", ""),
            })

        # agents
        for a in c._run("MATCH (a:Agent) RETURN a"):
            ap = dict(a["a"])
            nodes.append({
                "id": ap["id"],
                "type": "agent",
                "name": ap.get("name", ""),
            })

        # all edges
        for row in c._run("""
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, b.id AS target, type(r) AS edge_type,
                   properties(r) AS props
        """):
            link = {
                "source": row["source"],
                "target": row["target"],
                "edge_type": row["edge_type"],
            }
            # include edge properties
            if row["props"]:
                for k, v in row["props"].items():
                    link[k] = v
            links.append(link)

        return {"nodes": nodes, "links": links}
    finally:
        c.close()


@app.get("/graph/stats")
def graph_stats():
    # quick stats for dashboard
    c = _get_client()
    try:
        counts = c.count_nodes()
        best = c.get_best()
        contradictions = c.get_contradictions()
        pending = c.get_hypothesis_status("pending")
        return {
            "node_counts": counts,
            "best": best,
            "contradictions": len(contradictions),
            "pending_hypotheses": len(pending),
        }
    finally:
        c.close()


@app.get("/graph/debates")
def debates():
    # list all debate logs
    c = _get_client()
    try:
        rows = c._run("""
            MATCH (d:DebateLog)
            RETURN d ORDER BY d.created_at DESC
        """)
        return [dict(r["d"]) for r in rows]
    finally:
        c.close()


@app.get("/graph/lineage/{experiment_id}")
def lineage(experiment_id: int):
    c = _get_client()
    try:
        return c.get_experiment_chain(experiment_id)
    finally:
        c.close()


@app.get("/health")
def health():
    c = _get_client()
    try:
        counts = c.count_nodes()
        return {"status": "ok", "nodes": counts}
    finally:
        c.close()
