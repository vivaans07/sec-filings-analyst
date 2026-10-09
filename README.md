# SEC Filings Analyst

Question answering over SEC 10-K filings, built around one question: **how often does the system
find the passage that actually contains the answer, and can you prove it?**

Most filing chatbots are judged by eye. This project measures retrieval against ground truth taken
from the filings themselves: every number in a 10-K's financial statements is also tagged in XBRL,
so for each company the benchmark asks questions like *"What was Apple's total revenue for fiscal
2025?"* and knows the exact answer from the **same accession number** as the document being searched.

> **Status:** in progress (week 1). Retrieval and the XBRL benchmark are implemented; answer
> generation, faithfulness scoring and the dashboard are next. Results below fill in as runs complete.

## How it works

```
EDGAR (latest 10-K + XBRL company facts, cached)
   → parse:    HTML → text and table blocks → split into Items (1, 1A, 7, 7A, 8, ...)
   → chunk:    ~220-word windows inside an Item; tables kept whole with their caption
   → retrieve: BM25 (implemented here) · TF-IDF · hybrid via reciprocal rank fusion
   → evaluate: auto-generated XBRL questions → hit@k and MRR per method and per metric
```

- **25 companies** across tech, financials, energy and agriculture, health care, and consumer and industrials.
- **About 10 metrics per company:** revenue, net income, operating income, diluted EPS, R&D,
  operating cash flow, total assets, cash, equity and long-term debt. Each is only used when the XBRL fact comes from that
  10-K and covers the full fiscal year (or the year-end balance).
- **Answer matching** accepts how filings actually print numbers: in millions, in thousands, in full, or
  as "$X.X billion". It never accepts partial matches such as `1,391,035` for `391,035`.

## Run it

```bash
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name you@example.com"   # SEC requires a contact email
make test
make fetch      # downloads 25 10-Ks + XBRL facts (cached under data/)
make build      # parse, chunk, generate the question set
make eval       # results/retrieval.json
python -m secqa.cli search "supply chain risks" --ticker ADM
```

## Results

*Retrieval benchmark: to be filled in by `make eval`.*

## Roadmap

- [x] EDGAR client with caching and rate limiting
- [x] 10-K parser: Item splitting that skips the table of contents, tables kept as rows
- [x] BM25, TF-IDF and hybrid retrieval, scoped by company
- [x] XBRL-grounded question set + hit@k / MRR
- [ ] Dense embeddings and a reranker; compare against the sparse baselines
- [ ] Answer generation with citations to the chunk, plus numeric exact-match scoring
- [ ] Faithfulness checks: does every number in the answer appear in the cited passage?
- [ ] Hand-labeled narrative questions (risk factors, strategy, segments)
- [ ] Live results dashboard (GitHub Pages)

## Layout

```
secqa/      edgar.py · parse.py · chunk.py · retrieve.py · evalset.py · metrics.py · cli.py
tests/      unit tests for parsing, chunking, matching, fact selection and retrieval
eval/       generated question sets (committed)
results/    benchmark outputs (committed)
data/       downloaded filings and indexes (not committed)
```

## License

MIT
