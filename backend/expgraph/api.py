from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from expgraph.ingest import ingest_jsonl

app = FastAPI(title="expgraph")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "experiment_log.jsonl"


@app.get("/graph")
def graph(
    path: str = Query(default=None),
    user_id: str = Query(default="default"),
):
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {p}")
    return ingest_jsonl(p, user_id)


@app.get("/health")
def health():
    return {"status": "ok"}
