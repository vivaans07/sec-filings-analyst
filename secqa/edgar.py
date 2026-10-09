"""Small EDGAR client: ticker lookup, latest 10-K, filing download and XBRL company facts.

SEC asks automated tools to identify themselves with a contact email and to stay under
10 requests/second (https://www.sec.gov/os/accessing-edgar-data). Set SEC_USER_AGENT, e.g.
    export SEC_USER_AGENT="Your Name you@example.com"
Every response is cached under data/cache/, so re-runs never hit the network twice.
"""
import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import CACHE

MIN_INTERVAL = 0.15  # seconds between requests (~6.7 req/s, under SEC's 10/s limit)
_last = [0.0]


class EdgarError(RuntimeError):
    pass


def user_agent():
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise EdgarError("Set SEC_USER_AGENT to 'Your Name your@email' (SEC requires a contact email).")
    return ua


def _cache_path(url):
    return CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".bin")


def get(url, *, retries=3):
    """GET with on-disk cache, polite rate limiting and gzip handling. Returns bytes."""
    path = _cache_path(url)
    if path.exists():
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": user_agent(), "Accept-Encoding": "gzip"})
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            return body
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise EdgarError(f"{e.code} for {url}") from e
    raise EdgarError(f"gave up on {url}")


def get_json(url):
    return json.loads(get(url))


def ticker_map():
    """{'AAPL': {'cik': 320193, 'title': 'Apple Inc.'}, ...}"""
    raw = get_json("https://www.sec.gov/files/company_tickers.json")
    return {r["ticker"].upper(): {"cik": int(r["cik_str"]), "title": r["title"]} for r in raw.values()}


def latest_10k(cik):
    """Metadata for the most recent 10-K (not 10-K/A) in the company's submissions feed."""
    sub = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    recent = sub["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            acc = recent["accessionNumber"][i]
            return {
                "cik": cik,
                "company": sub.get("name", ""),
                "accession": acc,
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{recent['primaryDocument'][i]}",
            }
    raise EdgarError(f"no 10-K found for CIK {cik}")


def company_facts(cik):
    return get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")


def download_filing(meta):
    """Return the 10-K HTML as text and keep a readable copy under data/filings/."""
    html = get(meta["url"]).decode("utf-8", errors="replace")
    out = Path(CACHE).parent / "filings" / f"{meta['cik']}_{meta['accession']}.htm"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text(html, encoding="utf-8")
    return html
