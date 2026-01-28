# main.py
import argparse
from src.pipeline import run_pipeline
from src.crawler import crawl_if_needed
from src.config import USE_RERANKER_DEFAULT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["crawl", "run"], required=True)
    ap.add_argument("--rewrite", choices=["rule", "local", "qppgen"], default="rule")
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--llm-model", default="google/flan-t5-base")
    ap.add_argument("--llm-n", type=int, default=3)
    ap.add_argument("--golden", default="", help="Optional path to golden XLSX for eval")
    ap.add_argument("--split", default="all", help="all|train|test|unseen (if present)")
    ap.add_argument("--llm-max-new-tokens", type=int, default=48)

    args = ap.parse_args()

    if args.step == "crawl":
        crawl_if_needed()
    elif args.step == "run":
        use_rr = USE_RERANKER_DEFAULT and (not args.no_reranker)
        run_pipeline(
            mode=args.rewrite,
            use_reranker=use_rr,
            llm_model=args.llm_model,
            llm_n=args.llm_n,
            llm_max_tokens=args.llm_max_new_tokens,
            split=args.split,
            golden_override=args.golden  
        )
