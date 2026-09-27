"""Match baselines: ESCI native relevance classification, WDC identity pairs.

Per plans/02_commerce_match.md: compare exact-identifier rules, token/attribute
similarity, and a supervised encoder. Never map ESCI E to identity/variant (F02).
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split


def load_esci_relevance(n_dev: int = 20000, seed: int = 42):
    ex = pd.read_parquet("data/esci/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
    pr = pd.read_parquet("data/esci/esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")
    us_small_train = ex[(ex.product_locale == "us") & (ex.small_version == 1) & (ex.split == "train")]
    merged = us_small_train.merge(pr[pr.product_locale == "us"], on=["product_id", "product_locale"], how="left")
    merged = merged.dropna(subset=["product_title"])
    merged = merged.sample(n=min(n_dev, len(merged)), random_state=seed)
    return merged


def esci_label_to_query_relevance(label: str) -> str:
    """Native ESCI label mapped to this project's query_relevance predicate.
    E=Exact->exact, S=Substitute->substitute, C=Complement->complement,
    I=Irrelevant->irrelevant. This is ESCI's OWN native classification task
    (query-product relevance), never used to derive identity/variant labels."""
    return {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}[label]


def run_relevance_baseline(n_dev: int = 20000) -> dict:
    df = load_esci_relevance(n_dev=n_dev)
    df["y"] = df["esci_label"].map(esci_label_to_query_relevance)
    text = (df["query"].fillna("") + " [SEP] " + df["product_title"].fillna("")).tolist()
    y = df["y"].tolist()

    X_train, X_dev, y_train, y_dev = train_test_split(text, y, test_size=0.2, random_state=42, stratify=y)

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xdev = vec.transform(X_dev)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xdev)

    report = classification_report(y_dev, preds, output_dict=True)
    return {
        "n_train": len(X_train),
        "n_dev": len(X_dev),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {k: v for k, v in report.items() if k in ("exact", "substitute", "complement", "irrelevant")},
        "confusion_note": "See classification_report per_class precision/recall for confusion pattern.",
    }


def rules_identity_score(title_a: str, title_b: str) -> float:
    """Deterministic token-overlap Jaccard similarity — a rules baseline for
    identity, not a claim of state-of-the-art blocking."""
    def norm(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s.lower()))
    a, b = norm(title_a), norm(title_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_identity_baseline(threshold: float = 0.5) -> dict:
    with gzip.open("data/wdc/wdcproducts80cc20rnd000un_gs.json.gz") as f:
        rows = [json.loads(line) for line in f]

    y_true = [r["label"] for r in rows]  # 1 = match, 0 = non-match
    scores = [rules_identity_score(r["title_left"], r["title_right"]) for r in rows]
    y_pred = [1 if s >= threshold else 0 for s in scores]

    return {
        "n": len(rows),
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "positive_rate_gold": sum(y_true) / len(y_true),
        "hard_negative_rate": sum(r.get("is_hard_negative", False) for r in rows) / len(rows),
    }


if __name__ == "__main__":
    out = {
        "relevance_esci_native": run_relevance_baseline(n_dev=20000),
        "identity_wdc_rules_jaccard": run_identity_baseline(threshold=0.5),
    }
    out_path = Path("reports/match_baseline_2026-09-27/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
