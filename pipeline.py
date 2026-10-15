"""Content-based OSS recommendations with optional implicit-feedback CF."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
CATALOG = {
    "python": ("Python SDK toolkit", "Python client library, typed APIs, tests, packaging and developer documentation.",
               [("pagination", "Add pagination helpers to the Python SDK client"), ("typing", "Improve type hints and async examples in Python")]),
    "data": ("Data pipeline utilities", "Python data engineering, ETL jobs, tabular processing and reproducible pipelines.",
             [("etl", "Add incremental ETL checkpoint support"), ("schema", "Validate dataframe schema before loading data")]),
    "web": ("Web framework starter", "HTTP routing, middleware, web application templates and API integration.",
            [("middleware", "Document middleware ordering for web requests"), ("routes", "Support optional parameters in API routes")]),
    "security": ("Security checks", "Dependency scanning, secure defaults, vulnerability reports and threat modeling.",
                 [("scanner", "Report transitive dependency advisories"), ("secrets", "Add secret detection to pre-commit checks")]),
    "docs": ("Documentation tools", "Markdown, docs sites, examples, accessibility and contribution guides.",
             [("examples", "Add a runnable getting started example"), ("links", "Check links in generated documentation")]),
    "observability": ("Service telemetry", "Metrics, traces, logs, monitoring, alerting and instrumentation libraries.",
                      [("traces", "Propagate trace context across worker tasks"), ("metrics", "Add histogram support to the metrics exporter")]),
    "postgres": ("Postgres helpers", "PostgreSQL drivers, SQL queries, connection pools, migrations and database tooling.",
                 [("pool", "Expose pool wait time in database metrics"), ("migrations", "Make migration dry run output clearer")]),
    "rust": ("Rust async runtime", "Rust systems programming, async tasks, memory safety and runtime utilities.",
             [("runtime", "Improve task cancellation diagnostics"), ("docs", "Add examples for bounded async channels")]),
}


def _items():
    items = []
    for topic, (repo_title, repo_desc, issues) in CATALOG.items():
        items.append({"id": f"repo-{topic}", "kind": "repository", "title": repo_title,
                      "description": repo_desc, "topics": [topic]})
        for slug, title in issues:
            items.append({"id": f"issue-{topic}-{slug}", "kind": "issue", "title": title,
                          "description": f"Open contribution for {topic}: {title.lower()}.", "topics": [topic]})
    return items


def _fit(seed=SEED):
    items = _items()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    docs = [f"{x['title']} {x['description']} {' '.join(x['topics'])}" for x in items]
    return {"items": items, "vectorizer": vectorizer, "matrix": vectorizer.fit_transform(docs)}


def _content_scores(model, interests):
    query = model["vectorizer"].transform([interests])
    return cosine_similarity(query, model["matrix"])[0]


def _collaborative_scores(interactions, user_id, item_ids):
    if not interactions or user_id is None:
        return None
    frame = pd.DataFrame(interactions)
    required = {"user_id", "item_id", "rating"}
    if not required.issubset(frame.columns) or frame.user_id.nunique() < 2 or len(frame) < 4:
        return None
    matrix = frame.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="max", fill_value=0)
    if user_id not in matrix.index or len(matrix) < 2:
        return None
    profile = matrix.loc[[user_id]].to_numpy()
    peers = matrix.drop(index=user_id)
    similarities = cosine_similarity(profile, peers.to_numpy())[0]
    if not np.any(similarities > 0):
        return None
    weights = np.maximum(similarities, 0)
    denom = float(weights.sum())
    if denom <= 0:
        return None
    peer_values = peers.to_numpy(dtype=float)
    raw = (weights @ peer_values) / denom
    consumed = set(matrix.columns[matrix.loc[user_id].to_numpy() > 0])
    values = {item: float(raw[i]) for i, item in enumerate(matrix.columns) if item in item_ids and item not in consumed}
    maximum = max(values.values(), default=0)
    if maximum <= 0:
        return None
    return {item: value / maximum for item, value in values.items()}


def _rank(model, interests, limit=5, interactions=None, user_id=None):
    items = model["items"]
    content = _content_scores(model, interests)
    cf = _collaborative_scores(interactions, user_id, {item["id"] for item in items})
    scores = {}
    for i, item in enumerate(items):
        content_score = float(content[i])
        score = 0.8 * content_score + 0.2 * cf.get(item["id"], 0) if cf is not None else content_score
        scores[item["id"]] = (score, content_score, item)
    ranked = sorted(scores.values(), key=lambda entry: entry[0], reverse=True)[:limit]
    return ranked, cf is not None


def sample_input():
    return {"interests": "Python SDK client library typing async examples and tests", "limit": 5}


def train_model(seed=SEED):
    return _fit(seed)


def predict(model, payload):
    interests = payload["interests"] if isinstance(payload, dict) else str(payload)
    limit = int(payload.get("limit", 5)) if isinstance(payload, dict) else 5
    ranked, used_cf = _rank(model, interests, limit, payload.get("interactions"), payload.get("user_id")) if isinstance(payload, dict) else _rank(model, interests, limit)
    return {"method": "content+collaborative" if used_cf else "content",
            "recommendations": [{"id": item["id"], "kind": item["kind"], "title": item["title"],
                                 "score": round(score, 4), "content_score": round(content, 4)}
                                for score, content, item in ranked]}


def evaluate(seed=SEED):
    model = _fit(seed)
    precisions, recalls = [], []
    for topic in CATALOG:
        ranked, _ = _rank(model, f"{topic} open source repository issues contribution development")
        expected = {item["id"] for item in model["items"] if topic in item["topics"]}
        selected = {item[2]["id"] for item in ranked}
        precisions.append(len(expected & selected) / 5)
        recalls.append(len(expected & selected) / len(expected))
    return {"profiles": len(CATALOG), "precision_at_5": round(float(np.mean(precisions)), 4),
            "recall_at_5": round(float(np.mean(recalls)), 4), "catalog_items": len(model["items"])}
