from neo4j import GraphDatabase

from graph.models.nodes import Experiment, Technique, Hypothesis, Agent
from graph.models.edges import BaseEdge


class MemgraphClient:
    """thin wrapper around memgraph via bolt protocol."""

    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple | None = None):
        self._driver = GraphDatabase.driver(uri, auth=auth or ("", ""))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _run(self, query: str, params: dict | None = None) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    # -- schema --

    def create_indexes(self):
        """create indexes for fast lookups."""
        queries = [
            "CREATE INDEX ON :Experiment(experiment_id)",
            "CREATE INDEX ON :Experiment(commit)",
            "CREATE INDEX ON :Experiment(status)",
            "CREATE INDEX ON :Technique(name)",
            "CREATE INDEX ON :Agent(name)",
            "CREATE INDEX ON :Hypothesis(id)",
        ]
        for q in queries:
            try:
                self._run(q)
            except Exception:
                pass  # index already exists

    def drop_all(self):
        """clear the entire graph."""
        self._run("MATCH (n) DETACH DELETE n")

    # -- nodes --

    def create_experiment(self, exp: Experiment) -> str:
        query = """
        CREATE (e:Experiment {
            id: $id,
            experiment_id: $experiment_id,
            commit: $commit,
            val_bpb: $val_bpb,
            baseline_bpb: $baseline_bpb,
            delta_bpb: $delta_bpb,
            delta_vs_best: $delta_vs_best,
            delta_vs_own_best: $delta_vs_own_best,
            global_best_at_publish: $global_best_at_publish,
            agent_best_at_publish: $agent_best_at_publish,
            memory_gb: $memory_gb,
            peak_vram_mb: $peak_vram_mb,
            status: $status,
            training_steps: $training_steps,
            mfu_percent: $mfu_percent,
            tokens_trained_m: $tokens_trained_m,
            seed: $seed,
            batch: $batch,
            run_name: $run_name,
            hypothesis: $hypothesis,
            change_summary: $change_summary,
            category: $category,
            reasoning: $reasoning,
            agent_id: $agent_id,
            gpu: $gpu
        })
        RETURN e.id AS id
        """
        params = {
            "id": str(exp.id),
            "experiment_id": exp.experiment_id,
            "commit": exp.commit,
            "val_bpb": exp.val_bpb,
            "baseline_bpb": exp.baseline_bpb,
            "delta_bpb": exp.delta_bpb,
            "delta_vs_best": exp.delta_vs_best,
            "delta_vs_own_best": exp.delta_vs_own_best,
            "global_best_at_publish": exp.global_best_at_publish,
            "agent_best_at_publish": exp.agent_best_at_publish,
            "memory_gb": exp.memory_gb,
            "peak_vram_mb": exp.peak_vram_mb,
            "status": exp.status.value,
            "training_steps": exp.training_steps,
            "mfu_percent": exp.mfu_percent,
            "tokens_trained_m": exp.tokens_trained_m,
            "seed": exp.seed,
            "batch": exp.batch,
            "run_name": exp.run_name,
            "hypothesis": exp.hypothesis,
            "change_summary": exp.change_summary,
            "category": exp.category.value,
            "reasoning": exp.reasoning,
            "agent_id": exp.agent_id,
            "gpu": exp.gpu,
        }
        rows = self._run(query, params)
        return rows[0]["id"]

    def create_technique(self, tech: Technique) -> str:
        query = """
        MERGE (t:Technique {name: $name})
        ON CREATE SET t.id = $id, t.category = $category, t.success_rate = $success_rate
        RETURN t.id AS id
        """
        params = {
            "id": str(tech.id),
            "name": tech.name,
            "category": tech.category.value,
            "success_rate": tech.success_rate,
        }
        rows = self._run(query, params)
        return rows[0]["id"]

    def create_agent(self, agent: Agent) -> str:
        query = """
        MERGE (a:Agent {name: $name})
        ON CREATE SET a.id = $id, a.gpu = $gpu, a.total_runs = $total_runs,
                      a.best_val_bpb = $best_val_bpb, a.keep_rate = $keep_rate
        RETURN a.id AS id
        """
        params = {
            "id": str(agent.id),
            "name": agent.name,
            "gpu": agent.gpu,
            "total_runs": agent.total_runs,
            "best_val_bpb": agent.best_val_bpb,
            "keep_rate": agent.keep_rate,
        }
        rows = self._run(query, params)
        return rows[0]["id"]

    def create_hypothesis(self, hyp: Hypothesis) -> str:
        query = """
        CREATE (h:Hypothesis {
            id: $id, text: $text, debate_rounds: $debate_rounds,
            challenger_agreed: $challenger_agreed, winning_argument: $winning_argument
        })
        RETURN h.id AS id
        """
        params = {
            "id": str(hyp.id),
            "text": hyp.text,
            "debate_rounds": hyp.debate_rounds,
            "challenger_agreed": hyp.challenger_agreed,
            "winning_argument": hyp.winning_argument,
        }
        rows = self._run(query, params)
        return rows[0]["id"]

    # -- edges --

    def create_edge(self, edge: BaseEdge):
        """create an edge between two nodes by uuid."""
        props = {}
        for field in edge.__class__.model_fields:
            if field not in ("id", "edge_type", "source", "target", "created_at"):
                val = getattr(edge, field)
                if val is not None and val != "" and val != 0.0 and val is not False:
                    props[field] = val

        props_str = ", ".join(f"{k}: ${k}" for k in props) if props else ""
        props_clause = f" {{{props_str}}}" if props_str else ""

        query = f"""
        MATCH (a {{id: $source}})
        MATCH (b {{id: $target}})
        CREATE (a)-[r:{edge.edge_type.value}{props_clause}]->(b)
        RETURN type(r) AS edge_type
        """
        params = {"source": str(edge.source), "target": str(edge.target), **props}
        return self._run(query, params)

    # -- queries --

    def get_experiment(self, experiment_id: int) -> dict | None:
        rows = self._run(
            "MATCH (e:Experiment {experiment_id: $eid}) RETURN e",
            {"eid": experiment_id},
        )
        return dict(rows[0]["e"]) if rows else None

    def get_all_experiments(self) -> list[dict]:
        rows = self._run("MATCH (e:Experiment) RETURN e ORDER BY e.experiment_id")
        return [dict(r["e"]) for r in rows]

    def get_keeps(self) -> list[dict]:
        rows = self._run(
            "MATCH (e:Experiment {status: 'keep'}) RETURN e ORDER BY e.val_bpb ASC"
        )
        return [dict(r["e"]) for r in rows]

    def get_best(self) -> dict | None:
        rows = self._run(
            "MATCH (e:Experiment {status: 'keep'}) RETURN e ORDER BY e.val_bpb ASC LIMIT 1"
        )
        return dict(rows[0]["e"]) if rows else None

    def get_techniques(self) -> list[dict]:
        rows = self._run("MATCH (t:Technique) RETURN t ORDER BY t.name")
        return [dict(r["t"]) for r in rows]

    def get_neighbors(self, experiment_id: int) -> list[dict]:
        """get all nodes connected to an experiment."""
        rows = self._run(
            """
            MATCH (e:Experiment {experiment_id: $eid})-[r]-(n)
            RETURN type(r) AS edge_type, properties(n) AS neighbor
            """,
            {"eid": experiment_id},
        )
        return rows

    def get_technique_stats(self, technique_name: str) -> dict:
        """get keep/discard stats for a technique."""
        rows = self._run(
            """
            MATCH (e:Experiment)-[:TRIED]->(t:Technique {name: $name})
            RETURN e.status AS status, count(*) AS count
            """,
            {"name": technique_name},
        )
        return {r["status"]: r["count"] for r in rows}

    def get_experiment_chain(self, experiment_id: int) -> list[dict]:
        """follow IMPROVED_FROM edges backwards to find the lineage."""
        rows = self._run(
            """
            MATCH path = (e:Experiment {experiment_id: $eid})-[:IMPROVED_FROM*]->(ancestor)
            UNWIND nodes(path) AS n
            RETURN DISTINCT n.experiment_id AS experiment_id, n.val_bpb AS val_bpb,
                   n.change_summary AS change_summary, n.status AS status
            ORDER BY n.experiment_id
            """,
            {"eid": experiment_id},
        )
        return rows

    def count_nodes(self) -> dict:
        rows = self._run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(*) AS count
        """)
        return {r["label"]: r["count"] for r in rows}
