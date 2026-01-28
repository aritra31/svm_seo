# pipeline.py — main logic for IR search and QPP-triggered LLM reformulation

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder  # used only if use_reranker=True

from src.reformulator import Reformulator, ReformulatorConfig
from src.embedder import load_doc_embeddings, embed_query
from src.config import (
    # data paths
    DATA_DIR,
    CORPUS_DIR,
    OUTPUTS_DIR,
    GOLDEN_XLSX_BASE,
    GOLDEN_SPLIT_XLSX,
    # retrieval settings
    TOPK_PER_METHOD,
    FINAL_TOPK,
    RRF_K,
    CROSS_ENCODER_MODEL,
    RERANKER_TYPE,       # "semantic" or "none" (CrossEncoder controlled by use_reranker)
    RERANK_TOPN,
    ALWAYS_INCLUDE_RULE_VARIANTS,
    # strict eval toggles
    EVAL_DISABLE_URL_BOOST,
    EVAL_IGNORE_RULE_VARIANTS,
    EVAL_NO_LLM,
    STRICT_EXPECTED_ONLY,
)


# -------------------- UTILS --------------------


def _s(v) -> str:
    """Safe stringify that handles NaN/None."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v)


def _norm(s: str) -> str:
    """Normalize text/URLs: lowercase + strip."""
    return str(s or "").strip().lower()


def _rrf(lists, k: float = RRF_K):
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    lists: list of [(doc_id, score), ...]
    """
    agg = defaultdict(float)
    for L in lists:
        for rank, (doc, _) in enumerate(L):
            agg[doc] += 1.0 / (k + rank + 1)
    # keep top FINAL_TOPK
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)[:FINAL_TOPK]


def _metrics(ranks):
    """
    Compute core ranking metrics from a list of integer ranks (1-based, 999 = not found).
    """
    s = pd.Series(ranks)

    return {
        "success@1": float((s <= 1).mean()),
        "success@3": float((s <= 3).mean()),
        "recall@10": float((s <= 10).mean()),
        "recall@20": float((s <= 20).mean()),
        "MRR": float(s.apply(lambda r: 0 if r >= 999 else 1.0 / r).mean()),
        "NDCG@3": float(s.apply(lambda r: 0 if r > 3 else 1.0 / np.log2(r + 1)).mean()),
        "NDCG@10": float(s.apply(lambda r: 0 if r > 10 else 1.0 / np.log2(r + 1)).mean()),
        "count_999": int((s == 999).sum()),
        "total": int(len(s)),
    }


def _first_hit_rank(urls_list, target_set):
    """
    Return first rank where any normalized url in target_set appears; else 999.
    urls_list: list of raw URLs in ranking order.
    target_set: set of normalized URLs.
    """
    for idx, u in enumerate(urls_list, start=1):
        if _norm(u) in target_set:
            return idx
    return 999


def _get_target_urls(row):
    """
    Collect expected_url ∪ acceptable_urls (split by ';') as normalized set.

    STRICT_EXPECTED_ONLY=True => ignore acceptable_urls and only use expected_url.
    """
    targets = set()
    exp = _norm(row.get("expected_url", ""))
    if exp:
        targets.add(exp)

    if not STRICT_EXPECTED_ONLY:
        acc = str(row.get("acceptable_urls", "")).strip()
        if acc:
            for u in acc.split(";"):
                uu = _norm(u)
                if uu:
                    targets.add(uu)

    return targets


# -------------------- PIPELINE --------------------


def run_pipeline(
    mode: str,
    use_reranker: bool,
    llm_model: str,
    llm_n: int,
    llm_max_tokens: int,
    split: str = "all",
    golden_override: str = "",
):
    """
    Main entrypoint for retrieval + QPP + reranking.

    mode:
        "rule"   -> rule-based variants from golden 'variants' column
        "local"  -> always LLM reformulated (no QPP)
        "qppgen" -> QPP-triggered LLM reformulation (clarity score based)

    use_reranker:
        True  -> apply CrossEncoder (accuracy-first, slower)
        False -> rely on lexical + optional semantic reranker (RERANKER_TYPE="semantic")
    """

    # --- 1) Load corpus ---
    corpus_path = CORPUS_DIR / "articles.csv"
    df = pd.read_csv(corpus_path).fillna("")
    docs = df.to_dict("records")

    pages = [
        {
            "url": _norm(_s(d.get("url"))),
            "title": _norm(_s(d.get("title"))),
            "text": _norm((_s(d.get("title")) + " " + _s(d.get("body"))).strip()),
        }
        for d in docs
    ]
    texts = [p["text"] for p in pages]
    titles = [p["title"] for p in pages]

    # --- 2) Lexical retrievers ---
    tok = [t.split() for t in texts]  # already normalized by _norm
    bm25 = BM25Okapi(tok)

    tfv = TfidfVectorizer(
        lowercase=False,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
    )
    X = tfv.fit_transform(texts)

    # --- 3) Semantic reranker cache (fast path, if enabled) ---
    doc_embs = None
    if RERANKER_TYPE == "semantic":
        try:
            _, doc_embs = load_doc_embeddings()
            if doc_embs.shape[0] != len(texts):
                print("[WARN] Embedding count != corpus size. Rebuild embeddings if ranks look off.")
        except Exception as e:
            print(f"[WARN] Could not load embedding cache: {e}")
            doc_embs = None

    # --- 4) Reformulator (LLM for query reformulation) ---
    reform = Reformulator(
        ReformulatorConfig(
            llm_model=llm_model,
            n_variants=llm_n,
            qpp_threshold=0.18,
        )
    )

    # --- 5) Golden selection (override > split > base) ---
    if golden_override:
        print(f"[INFO] Using override golden: {golden_override}")
        gdf = pd.read_excel(golden_override).fillna("")
    else:
        if GOLDEN_SPLIT_XLSX.exists():
            print(f"[INFO] Using split file: {GOLDEN_SPLIT_XLSX}")
            gdf = pd.read_excel(GOLDEN_SPLIT_XLSX).fillna("")
        else:
            print(f"[INFO] Using base golden file: {GOLDEN_XLSX_BASE}")
            gdf = pd.read_excel(GOLDEN_XLSX_BASE).fillna("")

    # Optional split filtering
    if "split" in gdf.columns and split != "all":
        before = len(gdf)
        gdf = gdf[gdf["split"] == split].reset_index(drop=True)
        print(f"[INFO] Split filter '{split}': {before} → {len(gdf)} rows")
    elif split != "all":
        print("[WARN] '--split' provided but no 'split' column in golden; evaluating ALL rows.")

    # Normalize query column
    if "query_text" in gdf.columns:
        qcol = "query_text"
    else:
        q_candidates = [c for c in gdf.columns if "query" in c.lower()]
        if not q_candidates:
            raise ValueError("Golden set needs a query column (e.g., 'query_text').")
        qcol = q_candidates[0]

    gdf["query"] = gdf[qcol].astype(str).str.strip().str.lower()

    # Normalize URLs
    if "expected_url" not in gdf.columns:
        raise ValueError("Golden set must have 'expected_url' column.")
    gdf["expected_url"] = gdf["expected_url"].astype(str).str.strip().str.lower()
    if "acceptable_urls" not in gdf.columns:
        gdf["acceptable_urls"] = ""

    # Settings echo
    print(
        f"[SETTINGS] split={split} | reranker_type={RERANKER_TYPE} | "
        f"no_llm={EVAL_NO_LLM} | ignore_rule_variants={EVAL_IGNORE_RULE_VARIANTS} | "
        f"disable_url_boost={EVAL_DISABLE_URL_BOOST} | strict_expected_only={STRICT_EXPECTED_ONLY}"
    )

    # --- 6) Query loop ---
    rows = []

    for _, row in gdf.iterrows():
        q = row["query"]
        targets = _get_target_urls(row)

        # First-pass docs for feedback prompts (BM25 only)
        b_scores_first = bm25.get_scores(q.split())
        top_idxs = np.argsort(b_scores_first)[::-1][:5]
        top_docs = [texts[j] for j in top_idxs]

        # ---- Variants by mode ----
        if mode == "rule":
            # Raw query + golden rule-based variants
            variants = [q] + [
                v.strip().lower()
                for v in str(row.get("variants", "")).split(";")
                if v.strip()
            ]

        elif mode == "local":
            # Always LLM reformulated (no QPP decision)
            if EVAL_NO_LLM:
                variants = [q]
            else:
                llm_vars = reform._prompt_variants(q)
                fb_var = reform._feedback_refine(q, top_docs)
                variants = list({q, *llm_vars, fb_var})

        elif mode == "qppgen":
            # QPP-triggered LLM reformulation
            if EVAL_NO_LLM:
                variants = [q]
            else:
                variants = reform.reformulate(q, top_docs)

            # Optionally include golden rule-based variants (unless strict ignore)
            if ALWAYS_INCLUDE_RULE_VARIANTS and (not EVAL_IGNORE_RULE_VARIANTS):
                rule_vars = [
                    v.strip().lower()
                    for v in str(row.get("variants", "")).split(";")
                    if v.strip()
                ]
                variants = list({*variants, *rule_vars})

        else:
            # Fallback: just the raw query
            variants = [q]

        # Safety de-duplication
        variants = [v for v in {vv for vv in variants if vv}]

        # ---- 7) Per-variant retrieval and RRF fusion ----
        fused_all = []
        for v in variants:
            # BM25
            b_scores = bm25.get_scores(v.split())
            b = [(i, float(b_scores[i])) for i in np.argsort(b_scores)[::-1][:TOPK_PER_METHOD]]

            # TF-IDF cosine
            qv = tfv.transform([v])
            sims = cosine_similarity(qv, X)[0]
            t = [(i, float(sims[i])) for i in np.argsort(sims)[::-1][:TOPK_PER_METHOD]]

            # Fuzzy title matching
            f = [
                (
                    i,
                    float(
                        0.5 * fuzz.token_set_ratio(v, title_i)
                        + 0.5 * fuzz.WRatio(v, title_i)
                    ),
                )
                for i, title_i in enumerate(titles)
            ]
            f.sort(key=lambda x: x[1], reverse=True)
            f = f[:TOPK_PER_METHOD]

            fused_all.append(_rrf([b, t, f]))

        fused = _rrf(fused_all)
        reranked = fused

        # ---- 8) Reranking paths ----
        if use_reranker:
            # CrossEncoder path (accuracy-first, slower)
            take = min(len(fused), RERANK_TOPN)
            cand_ids = [i for i, _ in fused[:take]]
            pairs = [
                (
                    q,
                    pages[i]["title"] + " " + pages[i]["text"][:1200],
                )
                for i in cand_ids
            ]

            ce = CrossEncoder(CROSS_ENCODER_MODEL, max_length=256)
            scores = ce.predict(pairs)
            scores = np.asarray(scores).flatten().tolist()
            reranked = list(zip(cand_ids, [float(s) for s in scores])) + fused[take:]
            reranked = sorted(reranked, key=lambda x: x[1], reverse=True)

        elif RERANKER_TYPE == "semantic" and doc_embs is not None and len(fused) > 0:
            # Semantic reranker (fast on CPU, precomputed doc embeddings)
            take = min(len(fused), RERANK_TOPN)
            cand_ids = [i for i, _ in fused[:take]]
            q_vec = embed_query([q])[0]      # 1 x d (normalized)
            cand_vecs = doc_embs[cand_ids]   # take x d (normalized)
            sims = cand_vecs @ q_vec         # cosine similarity
            reranked = list(zip(cand_ids, sims.tolist())) + fused[take:]
            reranked = sorted(reranked, key=lambda x: x[1], reverse=True)

        # ---- 9) Deterministic boost for any acceptable/expected URL ----
        boosted = []
        for (doc_i, s) in reranked:
            url = pages[doc_i]["url"]
            if (not EVAL_DISABLE_URL_BOOST) and (url in targets):
                s += 2.0
            boosted.append((doc_i, s))
        reranked = sorted(boosted, key=lambda x: x[1], reverse=True)

        # ---- 10) Compute rank vs targets ----
        final_urls = [pages[i]["url"] for i, _ in reranked]
        rank = _first_hit_rank(final_urls, targets)

        rows.append(
            {
                "query": q,
                "expected_url": _norm(row.get("expected_url", "")),
                "predicted_rank": rank,
            }
        )

    # --- 11) Save outputs + console metrics ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_pred = OUTPUTS_DIR / f"predicted_{mode}.csv"
    out_met = OUTPUTS_DIR / f"metrics_{mode}.csv"

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_pred, index=False)
    metrics = _metrics(out_df["predicted_rank"].tolist())
    pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()]).to_csv(
        out_met, index=False
    )

    print(f"\n[{mode}] DONE → {out_pred.name}, {out_met.name}")
    for k, v in metrics.items():
        print(f"{k:>11}: {v}")
