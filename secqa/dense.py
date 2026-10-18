"""Dense retrieval: embed every chunk once with a sentence-embedding model, then score a query by
cosine similarity. Sparse methods match exact words ("net sales"); dense ones also match meaning
("revenue" ~ "net sales"), which is the gap this module is here to measure.

The default model is BAAI/bge-small-en-v1.5 (33M parameters, 384 dimensions, runs on CPU). BGE
models expect a short instruction in front of queries but not passages, so the two are encoded
differently. sentence-transformers is an optional dependency (pip install -r requirements-dense.txt)
and is only imported when an encoder is actually built, so the sparse pipeline and the tests run
without it.

Chunk embeddings are cached under data/index/, keyed by model name and a hash of the chunk ids and
texts, so `eval` and `search` reuse them instead of re-encoding thousands of chunks each run.
"""
import hashlib
import re

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder:
    """Wraps a sentence-transformers model. encode() returns L2-normalized float32 rows, so a dot
    product is cosine similarity."""

    def __init__(self, model_name=DEFAULT_MODEL, batch_size=64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("Dense retrieval needs sentence-transformers: "
                              "pip install -r requirements-dense.txt") from e
        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def encode(self, texts, *, query=False):
        if query:
            texts = [self.query_prefix + t for t in texts]
        v = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 1000, convert_to_numpy=True)
        return v.astype(np.float32)


def _fingerprint(texts, ids):
    h = hashlib.sha1()
    for i, t in zip(ids, texts):
        h.update(i.encode())
        h.update(b"\0")
        h.update(t.encode())
        h.update(b"\1")
    return h.hexdigest()[:16]


def _cache_file(cache_dir, model_name, fp):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-")
    return cache_dir / f"emb-{slug}-{fp}.npy"


class Dense:
    def __init__(self, texts, encoder, ids=None, cache_dir=None):
        self.encoder = encoder
        ids = ids or [str(i) for i in range(len(texts))]
        path = None
        if cache_dir is not None:
            path = _cache_file(cache_dir, getattr(encoder, "name", "model"), _fingerprint(texts, ids))
        if path is not None and path.exists():
            self.m = np.load(path)
        else:
            self.m = encoder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.m)

    def scores(self, query, allowed=None):
        q = self.encoder.encode([query], query=True)[0]
        s = (self.m @ q).astype(float)
        if allowed is not None:
            s[~allowed] = -np.inf
        return s
