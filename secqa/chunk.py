"""Section-aware chunking. Text is packed into ~target-word windows with overlap inside a single
Item; tables are never split mid-row and carry the nearest preceding caption line so a chunk
like "Total net sales | 391,035" still knows it came from the income statement."""
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    id: str
    ticker: str
    item: str
    title: str
    kind: str      # "text" or "table"
    text: str

    def to_dict(self):
        return asdict(self)


def _words(s):
    return len(s.split())


def chunk_sections(ticker, sections, target=220, overlap=40, table_rows=25):
    chunks, n = [], 0

    def emit(kind, text, sec):
        nonlocal n
        chunks.append(Chunk(f"{ticker}-{n:05d}", ticker, sec["item"], sec["title"], kind, text.strip()))
        n += 1

    for sec in sections:
        buf, last_caption = [], ""
        for b in sec["blocks"]:
            if b["kind"] == "table":
                if buf:
                    emit("text", " ".join(buf), sec)
                    buf = []
                rows = b["text"].split("\n")
                for k in range(0, len(rows), table_rows):
                    head = (last_caption + "\n") if last_caption else ""
                    emit("table", head + "\n".join(rows[k:k + table_rows]), sec)
                continue
            last_caption = b["text"] if _words(b["text"]) <= 25 else last_caption
            buf.append(b["text"])
            if sum(map(_words, buf)) >= target:
                emit("text", " ".join(buf), sec)
                tail = " ".join(buf).split()[-overlap:]
                buf = [" ".join(tail)]
        if buf and _words(" ".join(buf)) > overlap:
            emit("text", " ".join(buf), sec)
    return chunks
