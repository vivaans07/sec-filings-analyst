"""Retrievers over chunks: BM25 (implemented here), TF-IDF cosine, and a hybrid that fuses the
two rankings with reciprocal rank fusion (RRF). With an encoder (see dense.py) the index also
offers dense embedding search and a three-way hybrid of BM25, TF-IDF and dense. All accept an
optional ticker to scope search to one company's filing, which is how a user would ask
("What was Apple's revenue...").
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with
what which who how did does do our we us their they""".split())


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.postings[t].append((i, f))

    def scores(self, query, allowed=None):
        s = np.zeros(len(self.tf))
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s[i] += idf * f * (self.k1 + 1) / denom
        if allowed is not None:
            s[~allowed] = -np.inf
        return s


class Tfidf:
    def __init__(self, docs):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, min_df=1, ngram_range=(1, 2))
        self.m = self.vec.fit_transform(docs)

    def scores(self, query, allowed=None):
        q = self.vec.transform([query])
        s = (self.m @ q.T).toarray().ravel()
        if allowed is not None:
            s[~allowed] = -np.inf
        return s


def rrf(*score_arrays, k=60):
    """Reciprocal rank fusion: sum of 1/(k + rank) across rankers."""
    fused = np.zeros(len(score_arrays[0]))
    for s in score_arrays:
        order = np.argsort(-s)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(s) + 1)
        fused += np.where(np.isfinite(s), 1.0 / (k + ranks), 0.0)
    return fused


SPARSE_METHODS = ("bm25", "tfidf", "hybrid")
DENSE_METHODS = ("dense", "hybrid_dense")


class Index:
    def __init__(self, chunks, encoder=None, cache_dir=None):
        self.chunks = chunks
        texts = [f"{c['title']} {c['text']}" for c in chunks]
        self.tickers = np.array([c["ticker"] for c in chunks])
        self.bm25 = BM25(texts)
        self.tfidf = Tfidf(texts)
        self.dense = None
        if encoder is not None:
            from .dense import Dense
            self.dense = Dense(texts, encoder, ids=[c.get("id", str(i)) for i, c in enumerate(chunks)],
                               cache_dir=cache_dir)

    @property
    def methods(self):
        return SPARSE_METHODS + (DENSE_METHODS if self.dense else ())

    def search(self, query, ticker=None, k=5, method="hybrid"):
        allowed = (self.tickers == ticker) if ticker else None
        if method in DENSE_METHODS and self.dense is None:
            raise ValueError(f"method {method!r} needs an index built with an encoder")
        if method == "bm25":
            s = self.bm25.scores(query, allowed)
        elif method == "tfidf":
            s = self.tfidf.scores(query, allowed)
        elif method == "dense":
            # cosine can be <= 0 and still be the best match, so only scoping removes candidates
            s = self.dense.scores(query, allowed)
            top = np.argsort(-s)[:k]
            return [self.chunks[i] for i in top if np.isfinite(s[i])]
        elif method in ("hybrid", "hybrid_dense"):
            runs = [self.bm25.scores(query, allowed), self.tfidf.scores(query, allowed)]
            if method == "hybrid_dense":
                runs.append(self.dense.scores(query, allowed))
            s = rrf(*runs)
            if allowed is not None:
                s[~allowed] = -np.inf
        else:
            raise ValueError(method)
        top = np.argsort(-s)[:k]
        return [self.chunks[i] for i in top if np.isfinite(s[i]) and s[i] > 0]
