"""Command line.

  python -m secqa.cli fetch            download each company's latest 10-K + XBRL facts (cached)
  python -m secqa.cli build            parse, chunk, and write data/index/chunks.jsonl + eval/xbrl_questions.jsonl
  python -m secqa.cli eval             score BM25, TF-IDF and hybrid retrieval -> results/retrieval.json
  python -m secqa.cli search "query" --ticker AAPL
"""
import argparse
import json

from . import chunk, edgar, evalset, metrics, parse
from .config import INDEX, RESULTS, ROOT, TICKERS
from .retrieve import Index

EVAL = ROOT / "eval"


def _jsonl_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _jsonl_read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_fetch(a):
    tmap = edgar.ticker_map()
    manifest = []
    for t in a.tickers:
        info = tmap[t]
        meta = edgar.latest_10k(info["cik"])
        edgar.download_filing(meta)
        edgar.company_facts(info["cik"])
        manifest.append({"ticker": t, "title": info["title"], **meta})
        print(f"{t:6} {meta['filing_date']}  {meta['accession']}")
    _jsonl_write(INDEX / "manifest.jsonl", manifest)


def cmd_build(a):
    chunks, questions, stats = [], [], []
    for m in _jsonl_read(INDEX / "manifest.jsonl"):
        sections = parse.parse_10k(edgar.download_filing(m))
        cs = chunk.chunk_sections(m["ticker"], sections)
        qs = evalset.build_questions(m["ticker"], m["title"], m, edgar.company_facts(m["cik"]))
        chunks += [c.to_dict() for c in cs]
        questions += qs
        stats.append((m["ticker"], len(sections), len(cs), len(qs)))
        print(f"{m['ticker']:6} items={len(sections):2}  chunks={len(cs):4}  questions={len(qs):2}")
    _jsonl_write(INDEX / "chunks.jsonl", chunks)
    _jsonl_write(EVAL / "xbrl_questions.jsonl", questions)
    print(f"total: {len(chunks)} chunks, {len(questions)} questions")


def cmd_eval(a):
    idx = Index(_jsonl_read(INDEX / "chunks.jsonl"))
    qs = _jsonl_read(EVAL / "xbrl_questions.jsonl")
    out = {m: metrics.evaluate(idx, qs, m) for m in ("bm25", "tfidf", "hybrid")}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(out, indent=1))
    for m, r in out.items():
        print(f"{m:7} hit@1={r['hit@1']:.3f} hit@5={r['hit@5']:.3f} hit@10={r['hit@10']:.3f} mrr={r['mrr']:.3f}  (n={r['n']})")


def cmd_search(a):
    idx = Index(_jsonl_read(INDEX / "chunks.jsonl"))
    for c in idx.search(a.query, ticker=a.ticker, k=a.k, method=a.method):
        print(f"--- {c['id']}  Item {c['item']} ({c['kind']})\n{c['text'][:600]}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--tickers", nargs="*", default=TICKERS); f.set_defaults(fn=cmd_fetch)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--ticker"); s.add_argument("-k", type=int, default=5)
    s.add_argument("--method", default="hybrid", choices=["bm25", "tfidf", "hybrid"]); s.set_defaults(fn=cmd_search)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
