"""Related-model comparators for HF06 (RexBERT + linear probe), HF07
(RexReranker-0.6B), HF10 (retrieval controls as similarity proxy), and HF11
(general reranking controls) against the SAME ESCI relevance eval set used by
expansion/eval/match_baselines.py.

Per RELATED_MODEL_COMPARISON_MATRIX.md:
- HF06 "scoring an untrained classifier head is invalid" -> we fit a fast
  linear probe on RexBERT-base embeddings (frozen backbone) using the SAME
  train/dev split as the existing TF-IDF+LogisticRegression baseline. This is
  disclosed as "RexBERT+linear-probe", never a zero-shot claim.
- HF07/HF11 "classification accuracy cannot substitute for ranking metrics" ->
  we score native reranker outputs (yes/no probability or cross-encoder logit)
  per (query, product) pair and report BOTH a ranking-style metric (mean
  reciprocal rank / NDCG within same-query groups where possible) and, only as
  a secondary disclosed view, thresholded classification metrics matched to
  the existing baseline's label set.
- HF10 embeddings are NOT a classifier; we report cosine-similarity-threshold
  as an explicitly labeled proxy relevance signal, never claimed as the
  model's intended use.

No score in this file is fabricated: every number below is produced by an
actual forward pass recorded at run time, or the cell is marked NOT_RUN with
the real exception text.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_esci_relevance(n_dev: int = 20000, seed: int = 42) -> pd.DataFrame:
    ex = pd.read_parquet("data/esci/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
    pr = pd.read_parquet("data/esci/esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")
    us_small_train = ex[(ex.product_locale == "us") & (ex.small_version == 1) & (ex.split == "train")]
    merged = us_small_train.merge(pr[pr.product_locale == "us"], on=["product_id", "product_locale"], how="left")
    merged = merged.dropna(subset=["product_title"])
    merged = merged.sample(n=min(n_dev, len(merged)), random_state=seed)
    return merged


LABEL_MAP = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}


def build_split(n_dev: int = 20000, seed: int = 42):
    """Same split methodology as match_baselines.run_relevance_baseline:
    80/20 stratified split of the same sampled subset, so results are
    comparable cell-for-cell."""
    df = load_esci_relevance(n_dev=n_dev, seed=seed)
    df["y"] = df["esci_label"].map(LABEL_MAP)
    text = (df["query"].fillna("") + " [SEP] " + df["product_title"].fillna("")).tolist()
    queries = df["query"].fillna("").tolist()
    products = df["product_title"].fillna("").tolist()
    y = df["y"].tolist()

    idx = np.arange(len(df))
    idx_train, idx_dev = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    train = {"query": [queries[i] for i in idx_train], "product": [products[i] for i in idx_train],
             "y": [y[i] for i in idx_train]}
    dev = {"query": [queries[i] for i in idx_dev], "product": [products[i] for i in idx_dev],
           "y": [y[i] for i in idx_dev]}
    return train, dev


# ---------------------------------------------------------------------------
# HF06: thebajajra/RexBERT-base — frozen embeddings + linear probe
# ---------------------------------------------------------------------------

def run_rexbert_linear_probe(train, dev, batch_size: int = 32) -> dict:
    t0 = time.time()
    try:
        from transformers import AutoTokenizer, AutoModel

        repo = "thebajajra/RexBERT-base"
        tok = AutoTokenizer.from_pretrained(repo)
        enc = AutoModel.from_pretrained(repo)
        enc.eval()

        def embed(queries, products):
            texts = [f"{q} [SEP] {p}" for q, p in zip(queries, products)]
            feats = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    inputs = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                    out = enc(**inputs)
                    hidden = out.last_hidden_state  # (B, T, H)
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    mean_pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                    feats.append(mean_pooled.numpy())
            return np.concatenate(feats, axis=0)

        Xtr = embed(train["query"], train["product"])
        Xdev = embed(dev["query"], dev["product"])

        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xtr, train["y"])
        preds = clf.predict(Xdev)
        report = classification_report(dev["y"], preds, output_dict=True)
        return {
            "comparison_status": "RUN",
            "method": "frozen RexBERT-base mean-pooled embeddings + LogisticRegression head "
                      "(disclosed adaptation: linear probe only, backbone frozen, no gradient into RexBERT)",
            "n_train": len(train["y"]),
            "n_dev": len(dev["y"]),
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {k: v for k, v in report.items() if k in LABEL_MAP.values()},
            "wall_time_sec": time.time() - t0,
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


# ---------------------------------------------------------------------------
# HF07: thebajajra/RexReranker-0.6B — native yes/no pairwise reranking score
# ---------------------------------------------------------------------------

def run_rexreranker(dev, n: int = 300) -> dict:
    t0 = time.time()
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        repo = "thebajajra/RexReranker-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).eval()

        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        max_length = 512  # capped for CPU wall-clock; queries/titles are short

        prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
                  "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
                  "\"no\".<|im_end|>\n<|im_start|>user\n")
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        task = "Given a shopper search query, judge whether the product is what the shopper is looking for"

        def format_instruction(query, doc):
            return f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}"

        queries = dev["query"][:n]
        products = dev["product"][:n]
        gold = dev["y"][:n]

        scores = []
        bs = 8
        for i in range(0, len(queries), bs):
            pairs = [format_instruction(q, d) for q, d in zip(queries[i:i + bs], products[i:i + bs])]
            inputs = tokenizer(pairs, padding=False, truncation="longest_first",
                                return_attention_mask=False,
                                max_length=max_length - len(prefix_tokens) - len(suffix_tokens))
            for j, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][j] = prefix_tokens + ele + suffix_tokens
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
            with torch.no_grad():
                batch_scores = model(**inputs).logits[:, -1, :]
                true_vec = batch_scores[:, token_true_id]
                false_vec = batch_scores[:, token_false_id]
                stacked = torch.stack([false_vec, true_vec], dim=1)
                probs = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().tolist()
            scores.extend(probs)

        # Native use is pairwise yes/no relevance judgment, not 4-way ESCI
        # classification. We report it honestly as a binary "is this an exact
        # match" judgment (native framing) at threshold 0.5, using ESCI E vs
        # not-E as the binary ground truth (disclosed remap, not the model's
        # native output space).
        binary_gold = [1 if g == "exact" else 0 for g in gold]
        binary_pred = [1 if s >= 0.5 else 0 for s in scores]
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        try:
            auc = roc_auc_score(binary_gold, scores)
        except Exception:
            auc = None

        return {
            "comparison_status": "RUN",
            "method": "native yes/no reranker probability at threshold 0.5, remapped to binary "
                      "exact-vs-not-exact (disclosed remap from ESCI 4-way to reranker's native binary "
                      "judgment; NOT the model's native output space, no ranking-metric claim made "
                      "since ESCI here is not grouped into per-query candidate lists of controlled size)",
            "n": len(queries),
            "precision_exact": precision_score(binary_gold, binary_pred, zero_division=0),
            "recall_exact": recall_score(binary_gold, binary_pred, zero_division=0),
            "f1_exact": f1_score(binary_gold, binary_pred, zero_division=0),
            "roc_auc_exact": auc,
            "wall_time_sec": time.time() - t0,
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


# ---------------------------------------------------------------------------
# HF11: general reranking controls (sentence-transformers CrossEncoder)
# ---------------------------------------------------------------------------

def run_cross_encoder(repo: str, dev, n: int = 500) -> dict:
    t0 = time.time()
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(repo, max_length=256)
        queries = dev["query"][:n]
        products = dev["product"][:n]
        gold = dev["y"][:n]

        pairs = list(zip(queries, products))
        scores = model.predict(pairs).tolist()

        binary_gold = [1 if g == "exact" else 0 for g in gold]
        # Threshold at each model's own score median to avoid assuming a
        # fixed native scale (ms-marco and ettin have different score ranges).
        median = float(np.median(scores))
        binary_pred = [1 if s >= median else 0 for s in scores]
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        try:
            auc = roc_auc_score(binary_gold, scores)
        except Exception:
            auc = None

        return {
            "comparison_status": "RUN",
            "method": "native CrossEncoder.predict() relevance score; classification view is a "
                      "disclosed median-threshold remap to exact-vs-not-exact for comparability with "
                      "the existing ESCI baseline, matched to HF11's own warning that classification "
                      "accuracy cannot substitute for ranking metrics -> ROC-AUC over native scores is "
                      "the primary reported metric here, thresholded F1 is secondary",
            "n": len(queries),
            "roc_auc_exact": auc,
            "median_threshold_precision_exact": precision_score(binary_gold, binary_pred, zero_division=0),
            "median_threshold_recall_exact": recall_score(binary_gold, binary_pred, zero_division=0),
            "median_threshold_f1_exact": f1_score(binary_gold, binary_pred, zero_division=0),
            "wall_time_sec": time.time() - t0,
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


# ---------------------------------------------------------------------------
# HF10: retrieval controls — cosine similarity as an explicitly labeled proxy
# ---------------------------------------------------------------------------

def run_embedding_similarity_proxy(repo: str, dev, n: int = 500, query_prefix: str = "",
                                    doc_prefix: str = "") -> dict:
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(repo)
        queries = [query_prefix + q for q in dev["query"][:n]]
        products = [doc_prefix + p for p in dev["product"][:n]]
        gold = dev["y"][:n]

        q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        d_emb = model.encode(products, normalize_embeddings=True, show_progress_bar=False)
        sims = (q_emb * d_emb).sum(axis=1).tolist()

        binary_gold = [1 if g == "exact" else 0 for g in gold]
        median = float(np.median(sims))
        binary_pred = [1 if s >= median else 0 for s in sims]
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        try:
            auc = roc_auc_score(binary_gold, sims)
        except Exception:
            auc = None

        return {
            "comparison_status": "RUN",
            "method": "PROXY comparison, explicitly disclosed as reframing: this model's intended use "
                      "is dense retrieval (rank corpus by cosine similarity to a query), NOT "
                      "classification. We compute cosine(query_embedding, product_title_embedding) and "
                      "report it as a relevance-ranking SIGNAL (ROC-AUC of similarity vs exact-match "
                      "gold) plus a median-threshold binary view for comparability. This is not the "
                      "model's native benchmark (which would be Recall@K over a real corpus/gallery) "
                      "and must not be read as a retrieval-quality claim.",
            "n": len(queries),
            "roc_auc_exact": auc,
            "median_threshold_precision_exact": precision_score(binary_gold, binary_pred, zero_division=0),
            "median_threshold_recall_exact": recall_score(binary_gold, binary_pred, zero_division=0),
            "median_threshold_f1_exact": f1_score(binary_gold, binary_pred, zero_division=0),
            "wall_time_sec": time.time() - t0,
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


if __name__ == "__main__":
    train, dev = build_split(n_dev=20000, seed=42)
    print(f"train={len(train['y'])} dev={len(dev['y'])}")

    out = {}

    print("Running HF06 RexBERT-base linear probe...")
    out["hf06_rexbert_base_linear_probe"] = run_rexbert_linear_probe(train, dev)
    print(json.dumps(out["hf06_rexbert_base_linear_probe"], indent=2)[:2000])

    print("Running HF07 RexReranker-0.6B...")
    out["hf07_rexreranker_0_6b"] = run_rexreranker(dev, n=300)
    print(json.dumps(out["hf07_rexreranker_0_6b"], indent=2)[:2000])

    print("Running HF11 cross-encoder/ms-marco-MiniLM-L6-v2...")
    out["hf11_msmarco_minilm_l6_v2"] = run_cross_encoder("cross-encoder/ms-marco-MiniLM-L6-v2", dev, n=500)
    print(json.dumps(out["hf11_msmarco_minilm_l6_v2"], indent=2)[:2000])

    print("Running HF11 cross-encoder/ettin-reranker-150m-v1...")
    out["hf11_ettin_reranker_150m_v1"] = run_cross_encoder("cross-encoder/ettin-reranker-150m-v1", dev, n=500)
    print(json.dumps(out["hf11_ettin_reranker_150m_v1"], indent=2)[:2000])

    print("Running HF10 Qwen3-Embedding-0.6B (proxy)...")
    out["hf10_qwen3_embedding_0_6b_proxy"] = run_embedding_similarity_proxy(
        "Qwen/Qwen3-Embedding-0.6B", dev, n=300,
        query_prefix="Instruct: Retrieve products relevant to the shopper query\nQuery: ",
    )
    print(json.dumps(out["hf10_qwen3_embedding_0_6b_proxy"], indent=2)[:2000])

    print("Running HF10 intfloat/multilingual-e5-small (proxy)...")
    out["hf10_multilingual_e5_small_proxy"] = run_embedding_similarity_proxy(
        "intfloat/multilingual-e5-small", dev, n=500,
        query_prefix="query: ", doc_prefix="passage: ",
    )
    print(json.dumps(out["hf10_multilingual_e5_small_proxy"], indent=2)[:2000])

    out_path = Path("reports/comparator_baselines_2026-09-27/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Wrote", out_path)
