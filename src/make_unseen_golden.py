# tools/make_unseen_golden.py — create an "unseen" golden from unlabeled corpus pages
import pandas as pd
from pathlib import Path

DATA = Path("data")
CORPUS = DATA / "corpus" / "articles.csv"
BASE = DATA / "SurveyMonkey_Golden_Dataset.xlsx"
SPLIT = DATA / "golden_split.xlsx"
OUT   = DATA / "golden_unseen.xlsx"

def norm(s): return str(s or "").strip().lower()

def to_query(title: str) -> str:
    t = str(title or "").strip()
    if not t:
        return "how do i use this feature in surveymonkey"
    if not t.lower().startswith(("how ", "what ", "why ", "when ", "where ")):
        return f"how to {t}"
    return t

def main(sample=40, seed=42):
    corpus = pd.read_csv(CORPUS).fillna("")
    gold = pd.read_excel(SPLIT if SPLIT.exists() else BASE).fillna("")
    labeled = set(gold["expected_url"].astype(str).map(norm))
    if "acceptable_urls" in gold.columns:
        for cell in gold["acceptable_urls"].astype(str):
            for u in str(cell).split(";"):
                u2 = norm(u)
                if u2: labeled.add(u2)

    corpus["url_norm"] = corpus["url"].map(norm)
    unseen = corpus[~corpus["url_norm"].isin(labeled)].copy()
    if unseen.empty:
        raise SystemExit("No unseen pages available. Crawl more or re-check corpus.")

    if len(unseen) > sample:
        unseen = unseen.sample(n=sample, random_state=seed)

    out = pd.DataFrame({
        "query_text": unseen["title"].map(to_query),
        "expected_url": unseen["url"],
        "acceptable_urls": "",  # keep strict
        "split": "unseen",
    })
    out.to_excel(OUT, index=False)
    print(f"[UNSEEN] Wrote {len(out)} rows → {OUT}")

if __name__ == "__main__":
    main()
