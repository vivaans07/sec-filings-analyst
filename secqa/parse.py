"""Turn a 10-K's HTML into plain text blocks and split it into its standard Items.

Tables are kept as single blocks with cells joined by " | " so a row like
"Total net sales | 391,035 | 383,285" stays together for retrieval.
"""
import re
from bs4 import BeautifulSoup

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Reserved",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
ITEM_RE = re.compile(r"^\s*item\s*(1[0-6]|[1-9][abc]?)\s*[.:\-—–]?\s*(.*)$", re.I)
WS = re.compile(r"[ \t  ​]+")


def _clean(s):
    return WS.sub(" ", s).strip()


def _table_text(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in ("$", "%", ")")]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def html_to_blocks(html):
    """Ordered list of {'kind': 'text'|'table', 'text': str}."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="display: none"]'):
        hidden.decompose()  # inline-XBRL header blocks
    blocks = []
    body = soup.body or soup
    for el in body.find_all(["table", "p", "div", "span", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        if el.find_parent("table") is not None:
            continue
        if el.name == "table":
            t = _table_text(el)
            if t:
                blocks.append({"kind": "table", "text": t})
            el.decompose()
            continue
        if el.find(["p", "div", "table", "li"]):  # only leaf-ish containers carry text
            continue
        t = _clean(el.get_text(" "))
        if t and not (blocks and blocks[-1]["kind"] == "text" and blocks[-1]["text"].endswith(t)):
            blocks.append({"kind": "text", "text": t})
    return blocks


def split_items(blocks):
    """Assign blocks to 10-K Items. The table of contents also lists every Item, so for each
    Item we keep the heading occurrence that is followed by the most text."""
    heads = []
    for i, b in enumerate(blocks):
        if b["kind"] != "text" or len(b["text"]) > 200:
            continue
        m = ITEM_RE.match(b["text"])
        if m:
            heads.append((i, m.group(1).upper()))
    if not heads:
        return [{"item": "ALL", "title": "Full filing", "blocks": blocks}]
    spans = []
    for n, (i, item) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(blocks)
        size = sum(len(b["text"]) for b in blocks[i + 1:end])
        spans.append((item, i, end, size))
    best = {}
    for item, i, end, size in spans:
        if item not in best or size > best[item][3]:
            best[item] = (i, end, item, size)
    sections = []
    for item, (i, end, _, size) in sorted(best.items(), key=lambda kv: kv[1][0]):
        if size < 200:
            continue
        sections.append({"item": item, "title": ITEM_TITLES.get(item, ""), "blocks": blocks[i + 1:end]})
    return sections


def parse_10k(html):
    return split_items(html_to_blocks(html))
