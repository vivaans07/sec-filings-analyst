"""Retrieval metrics: hit@k (does any of the top k chunks contain the gold answer?) and MRR."""
from collections import defaultdict

from .evalset import contains_answer


def evaluate(index, questions, method, ks=(1, 3, 5, 10)):
    kmax = max(ks)
    hits = {k: 0 for k in ks}
    rr_total = 0.0
    per_metric = defaultdict(lambda: [0, 0])
    misses = []
    for q in questions:
        results = index.search(q["question"], ticker=q["ticker"], k=kmax, method=method)
        rank = next((i + 1 for i, c in enumerate(results) if contains_answer(c["text"], q["answers"])), None)
        for k in ks:
            hits[k] += bool(rank and rank <= k)
        rr_total += 1.0 / rank if rank else 0.0
        per_metric[q["metric"]][0] += bool(rank and rank <= 5)
        per_metric[q["metric"]][1] += 1
        if not rank:
            misses.append(q["id"])
    n = len(questions) or 1
    return {
        "method": method, "n": len(questions),
        **{f"hit@{k}": round(hits[k] / n, 3) for k in ks},
        "mrr": round(rr_total / n, 3),
        "hit@5_by_metric": {m: round(h / t, 3) for m, (h, t) in sorted(per_metric.items())},
        "misses": misses,
    }
