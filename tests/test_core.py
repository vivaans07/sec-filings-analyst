from secqa import chunk, evalset, parse
from secqa.retrieve import BM25, Index, rrf
import numpy as np

HTML = """<html><body>
<p>Table of Contents</p>
<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 7. Management's Discussion and Analysis</p>
<p>Item 1. Business</p>
<p>The Company designs, manufactures and markets smartphones, personal computers and wearables worldwide.</p>
<p>Item 1A. Risk Factors</p>
<p>Global economic conditions could materially adversely affect the Company's business and results of operations.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Total net sales increased 2% compared to the prior year, driven by higher Services revenue across every region.</p>
<p>Net sales by category</p>
<table><tr><td>Total net sales</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr>
<tr><td>Net income</td><td>93,736</td><td>96,995</td></tr></table>
</body></html>"""


def test_items_skip_table_of_contents():
    secs = parse.parse_10k(HTML.replace("worldwide.", "worldwide. " + "More detail. " * 40)
                           .replace("operations.", "operations. " + "More risk. " * 40)
                           .replace("region.", "region. " + "More analysis. " * 40))
    items = [s["item"] for s in secs]
    assert items == ["1", "1A", "7"]
    mdna = next(s for s in secs if s["item"] == "7")
    assert any(b["kind"] == "table" and "Total net sales | 391,035 | 383,285" in b["text"] for b in mdna["blocks"])


def test_table_chunks_keep_caption():
    secs = [{"item": "7", "title": "MD&A", "blocks": [
        {"kind": "text", "text": "Net sales by category"},
        {"kind": "table", "text": "Total net sales | 391,035 | 383,285"}]}]
    cs = chunk.chunk_sections("AAPL", secs)
    t = [c for c in cs if c.kind == "table"][0]
    assert t.text.startswith("Net sales by category") and "391,035" in t.text


def test_answer_strings_and_matching():
    ans = evalset.answer_strings(391_035_000_000, "USD")
    assert "391,035" in ans and "391.0 billion" in ans
    assert evalset.contains_answer("Total net sales | 391,035 | 383,285", ans)
    assert not evalset.contains_answer("Total net sales | 1,391,035", ans)   # no partial-number matches
    assert evalset.answer_strings(6.08, "USD/shares") == ["6.08"]


def test_find_fact_uses_same_filing_and_full_year():
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2024-06-30", "end": "2024-09-28", "val": 1, "accn": "A", "form": "10-K", "fy": 2024},   # quarter
        {"start": "2023-10-01", "end": "2024-09-28", "val": 5, "accn": "OLD", "form": "10-K", "fy": 2024},  # other filing
        {"start": "2023-10-01", "end": "2024-09-28", "val": 9, "accn": "A", "form": "10-K", "fy": 2024},
    ]}}}}}
    f = evalset.find_fact(facts, ["Revenues"], "A", "duration", "2024-09-28")
    assert f["value"] == 9


def test_short_name():
    assert evalset.short_name("Apple Inc.") == "Apple"
    assert evalset.short_name("JPMORGAN CHASE & CO") == "Jpmorgan Chase"
    assert evalset.short_name("Microsoft Corp") == "Microsoft"


def test_bm25_ranks_relevant_doc_first():
    docs = ["revenue grew in services", "risk factors include competition", "net income and revenue rose"]
    s = BM25(docs).scores("revenue services")
    assert int(np.argmax(s)) == 0


def test_rrf_and_ticker_scoping():
    chunks = [{"ticker": "AAPL", "title": "", "text": "total net sales 391,035"},
              {"ticker": "MSFT", "title": "", "text": "total revenue 245,122"}]
    idx = Index(chunks)
    assert idx.search("total revenue", ticker="AAPL", k=2)[0]["ticker"] == "AAPL"
    fused = rrf(np.array([3.0, 1.0]), np.array([1.0, 2.0]))
    assert fused.shape == (2,)
