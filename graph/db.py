from neo4j import GraphDatabase

from graph.models.nodes import Experiment, Technique, Hypothesis, Result, Run, Agent
from graph.models.edges import BaseEdge


class MemgraphClient:
    # thin wrapper around memgraph via bolt protocol

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
        queries = [
            "CREATE INDEX ON :Experiment(experiment_id)",
            "CREATE INDEX ON :Experiment(commit)",
            "CREATE INDEX ON :Experiment(status)",
            "CREATE INDEX ON :Technique(name)",
            "CREATE INDEX ON :Agent(name)",
            "CREATE INDEX ON :Hypothesis(id)",
            "CREATE INDEX ON :Result(id)",
            "CREATE INDEX ON :Run(name)",
        ]
        for q in queries:
            try:
                self._run(q)
            except Exception:
                pass

    def drop_all(self):
        self._run("MATCH (n) DETACH DELETE n")

    # -- create nodes --

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
            id: $id, text: $text, status: $status, category: $category,
            debate_rounds: $debate_rounds, challenger_agreed: $challenger_agreed,
            winning_argument: $winning_argument
        })
        RETURN h.id AS id
        """
        params = {
            "id": str(hyp.id),
            "text": hyp.text,
            "status": hyp.status.value,
            "category": hyp.category.value,
            "debate_rounds": hyp.debate_rounds,
            "challenger_agreed": hyp.challenger_agreed,
            "winning_argument": hyp.winning_argument,
        }
        rows = self._run(query, params)
        return rows[0]["id"]

    def create_result(self, res: Result) -> str:
        query = """
        CREATE (r:Result {
            id: $id, text: $text, val_bpb: $val_bpb,
            delta: $delta, kept: $kept, category: $category
        })
        RETURN r.id AS id
        """
        params = {
            "id": str(res.id),
            "text": res.text,
            "val_bpb": res.val_bpb,
            "delta": res.delta,
            "kept": res.kept,
            "category": res.category.value,
        }
        rows = self._run(query, params)
        return rows[0]["id"]

    def create_run(self, run: Run) -> str:
        query = """
        CREATE (r:Run {
            id: $id, name: $name, total_experiments: $total_experiments,
            best_val_bpb: $best_val_bpb, keep_count: $keep_count
        })
        RETURN r.id AS id
        """
        params = {
            "id": str(run.id),
            "name": run.name,
            "total_experiments": run.total_experiments,
            "best_val_bpb": run.best_val_bpb,
            "keep_count": run.keep_count,
        }
        rows = self._run(query, params)
        return rows[0]["id"]

    # -- edges --

    def create_edge(self, edge: BaseEdge):
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

    # -- basic lookups --

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
        rows = self._run(
            """
            MATCH (e:Experiment {experiment_id: $eid})-[r]-(n)
            RETURN type(r) AS edge_type, properties(n) AS neighbor
            """,
            {"eid": experiment_id},
        )
        return rows

    def get_technique_stats(self, technique_name: str) -> dict:
        rows = self._run(
            """
            MATCH (e:Experiment)-[:TRIED]->(t:Technique {name: $name})
            RETURN e.status AS status, count(*) AS count
            """,
            {"name": technique_name},
        )
        return {r["status"]: r["count"] for r in rows}

    def count_nodes(self) -> dict:
        rows = self._run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(*) AS count
        """)
        return {r["label"]: r["count"] for r in rows}

    # -- traversals --
    # these are multi-hop queries the debate agents use to build arguments

    def get_experiment_chain(self, experiment_id: int) -> list[dict]:
        # follow IMPROVED_FROM backwards to find full lineage
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

    def get_downstream(self, experiment_id: int, depth: int = 5) -> list[dict]:
        # everything that happened after this experiment
        rows = self._run(
            f"""
            MATCH (e:Experiment {{experiment_id: $eid}})<-[:IMPROVED_FROM|FAILED_FROM*1..{depth}]-(child)
            RETURN child.experiment_id AS experiment_id, child.val_bpb AS val_bpb,
                   child.change_summary AS change_summary, child.status AS status
            ORDER BY child.experiment_id
            """,
            {"eid": experiment_id},
        )
        return rows

    def get_category_history(self, category: str) -> list[dict]:
        # all experiments in a category, with their outcomes
        rows = self._run(
            """
            MATCH (e:Experiment {category: $cat})
            RETURN e.experiment_id AS experiment_id, e.val_bpb AS val_bpb,
                   e.change_summary AS change_summary, e.status AS status
            ORDER BY e.experiment_id
            """,
            {"cat": category},
        )
        return rows

    def get_failed_after(self, experiment_id: int) -> list[dict]:
        # what failed experiments branched from this one?
        rows = self._run(
            """
            MATCH (e:Experiment {experiment_id: $eid})<-[:FAILED_FROM]-(f:Experiment)
            RETURN f.experiment_id AS experiment_id, f.val_bpb AS val_bpb,
                   f.change_summary AS change_summary, f.delta_bpb AS delta_bpb
            ORDER BY f.experiment_id
            """,
            {"eid": experiment_id},
        )
        return rows

    def get_technique_history(self, technique_name: str) -> list[dict]:
        # full history of a technique — every experiment that used it
        rows = self._run(
            """
            MATCH (e:Experiment)-[:TRIED]->(t:Technique {name: $name})
            RETURN e.experiment_id AS experiment_id, e.val_bpb AS val_bpb,
                   e.change_summary AS change_summary, e.status AS status
            ORDER BY e.experiment_id
            """,
            {"name": technique_name},
        )
        return rows

    def get_contradictions(self) -> list[dict]:
        # find all result pairs that contradict each other
        rows = self._run("""
            MATCH (a:Result)-[c:CONTRADICTS]->(b:Result)
            RETURN a.text AS result_a, b.text AS result_b, c.explanation AS explanation
        """)
        return rows

    def get_hypothesis_status(self, status: str = "pending") -> list[dict]:
        # all hypotheses with a given status
        rows = self._run(
            """
            MATCH (h:Hypothesis {status: $status})
            RETURN h.id AS id, h.text AS text, h.debate_rounds AS debate_rounds,
                   h.challenger_agreed AS challenger_agreed
            """,
            {"status": status},
        )
        return rows

    def get_results_for_experiment(self, experiment_id: int) -> list[dict]:
        # what did we learn from a specific experiment?
        rows = self._run(
            """
            MATCH (e:Experiment {experiment_id: $eid})-[:PRODUCED]->(r:Result)
            RETURN r.text AS text, r.val_bpb AS val_bpb, r.delta AS delta, r.kept AS kept
            """,
            {"eid": experiment_id},
        )
        return rows

    def get_run_experiments(self, run_name: str) -> list[dict]:
        # all experiments in a batch run
        rows = self._run(
            """
            MATCH (e:Experiment)-[:PART_OF]->(r:Run {name: $name})
            RETURN e.experiment_id AS experiment_id, e.val_bpb AS val_bpb,
                   e.change_summary AS change_summary, e.status AS status
            ORDER BY e.experiment_id
            """,
            {"name": run_name},
        )
        return rows

    def get_hypothesis_chain(self, hypothesis_id: str) -> list[dict]:
        # follow REFINES edges to see how an idea evolved
        rows = self._run(
            """
            MATCH path = (h:Hypothesis {id: $hid})-[:REFINES*]->(ancestor:Hypothesis)
            UNWIND nodes(path) AS n
            RETURN DISTINCT n.id AS id, n.text AS text, n.status AS status
            """,
            {"hid": hypothesis_id},
        )
        return rows

    def search_experiments(self, keyword: str) -> list[dict]:
        # fuzzy search on change_summary
        rows = self._run(
            """
            MATCH (e:Experiment)
            WHERE e.change_summary CONTAINS $kw
            RETURN e.experiment_id AS experiment_id, e.val_bpb AS val_bpb,
                   e.change_summary AS change_summary, e.status AS status
            ORDER BY e.experiment_id
            """,
            {"kw": keyword},
        )
        return rows
