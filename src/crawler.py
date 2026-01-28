# crawler.py — smart BFS crawling and targeted backfill
import time, random, re
import pandas as pd
from collections import deque
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
from src.config import *


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip())

def _is_help_url(url):
    try:
        return urlparse(url).netloc == urlparse(HELP_BASE).netloc
    except:
        return False

def _fetch(session, url):
    time.sleep(random.uniform(*REQUEST_SLEEP_RANGE))
    r = session.get(url, headers={"User-Agent": "SM-Crawler"}, timeout=20)
    r.raise_for_status()
    return r.text

def _extract_links(base_url, html):
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for tag in soup.select("a[href]"):
        href = tag.get("href", "").strip()
        if href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            continue
        try:
            abs_url = urljoin(base_url, href)
            if _is_help_url(abs_url):
                links.add(abs_url)
        except:
            continue
    return list(links)

def _extract_article(url, html):
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find(["h1", "h2"], string=True)
    title = _norm(title_el.get_text()) if title_el else ""
    parts = [title]
    for tag in soup.select("h1, h2, h3, h4, p, li"):
        text = _norm(tag.get_text(" ", strip=True))
        if text:
            parts.append(text)
    body = _norm(" ".join(parts))
    return {"url": url, "title": title, "body": body}

def crawl_if_needed():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / "articles.csv"
    if out_path.exists():
        print(f"[INFO] Corpus already exists: {out_path.name}")
        return pd.read_csv(out_path)

    visited, queue, records = set(), deque(SEEDS), []
    session = requests.Session()

    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()
        if url in visited: continue
        visited.add(url)
        if not url.startswith(f"{HELP_BASE}/en/"): continue

        try:
            html = _fetch(session, url)
        except Exception as e:
            print(f"[WARN] Failed to fetch: {url} — {e}")
            continue

        for link in _extract_links(url, html):
            if link not in visited:
                queue.append(link)

        article = _extract_article(url, html)
        if len(article["title"]) >= 5 and len(article["body"]) >= 200:
            records.append(article)

    df = pd.DataFrame(list({r["url"]: r for r in records}.values()))
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved corpus: {len(df)} articles")
    return df

def backfill_from_golden():
    df = pd.read_csv(CORPUS_DIR / "articles.csv")
    urls = set(df["url"].astype(str).str.strip())

    golden = pd.read_excel(GOLDEN_XLSX_BASE)
    golden_urls = set(golden["expected_url"].dropna().astype(str).str.strip())
    alt_urls = golden["acceptable_urls"].dropna().astype(str).str.split(";").explode().str.strip()
    golden_urls.update(alt_urls.tolist())

    missing = [u for u in golden_urls if u and u not in urls]
    if not missing:
        print("[INFO] No missing golden URLs.")
        return

    session = requests.Session()
    new_rows = []
    for u in missing:
        try:
            html = _fetch(session, u)
            art = _extract_article(u, html)
            if len(art["title"]) >= 5 and len(art["body"]) >= 200:
                new_rows.append(art)
        except Exception as e:
            print(f"[WARN] Backfill failed: {u} — {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        full_df = pd.concat([df, new_df], ignore_index=True)
        full_df.drop_duplicates(subset="url", inplace=True)
        full_df.to_csv(CORPUS_DIR / "articles.csv", index=False)
        print(f"[INFO] Backfilled {len(new_rows)} articles → total {len(full_df)}")
    else:
        print("[WARN] No backfill articles added.")
