"""Build a ground-truth question set from XBRL facts filed with the same 10-K.

Every number in a 10-K's financial statements is also tagged in XBRL, so for each company we can
ask "What was <company>'s <metric> for fiscal <year>?" and know the exact answer, taken from the
same accession number as the document we index. That gives hundreds of checkable questions with
no hand labeling. Narrative questions (risk factors, strategy) are hand-written in
eval/narrative.jsonl and scored separately.
"""
import re
from datetime import date

# metric -> (question phrase, candidate us-gaap concepts in priority order, period type)
METRICS = {
    "revenue": ("total revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                                  "SalesRevenueNet", "RevenuesNetOfInterestExpense"], "duration"),
    "net_income": ("net income", ["NetIncomeLoss", "ProfitLoss"], "duration"),
    "operating_income": ("operating income", ["OperatingIncomeLoss"], "duration"),
    "eps_diluted": ("diluted earnings per share", ["EarningsPerShareDiluted"], "duration"),
    "rnd": ("research and development expense", ["ResearchAndDevelopmentExpense"], "duration"),
    "op_cash_flow": ("cash provided by operating activities",
                     ["NetCashProvidedByUsedInOperatingActivities"], "duration"),
    "total_assets": ("total assets", ["Assets"], "instant"),
    "cash": ("cash and cash equivalents", ["CashAndCashEquivalentsAtCarryingValue"], "instant"),
    "equity": ("total shareholders' equity", ["StockholdersEquity"], "instant"),
    "lt_debt": ("long-term debt", ["LongTermDebtNoncurrent", "LongTermDebt"], "instant"),
}
SUFFIX = re.compile(r",?\s+(inc|corp|corporation|co|company|plc|ltd|limited|holdings?|group|n\.?v\.?)\.?$", re.I)


def short_name(title):
    name = title.strip()
    for _ in range(2):
        name = SUFFIX.sub("", name).rstrip(" &,")
    return name.title() if name.isupper() else name


def _days(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def find_fact(facts, concepts, accession, period, report_date):
    """The fact for the filing's own fiscal year, reported in that exact 10-K."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        units = gaap.get(concept, {}).get("units", {})
        for unit, rows in units.items():
            for r in rows:
                if r.get("accn") != accession or r.get("form") != "10-K" or r.get("end") != report_date:
                    continue
                if period == "duration" and not (r.get("start") and 350 <= _days(r["start"], r["end"]) <= 380):
                    continue
                if period == "instant" and r.get("start"):
                    continue
                return {"concept": concept, "unit": unit, "value": r["val"], "fy": r.get("fy"), "end": r["end"]}
    return None


def answer_strings(value, unit):
    """Ways the number can appear in the filing's text or tables."""
    v = abs(value)
    if unit.startswith("USD/"):  # per-share
        return sorted({f"{v:.2f}", f"{v:.2f}".rstrip("0").rstrip(".")})
    out = set()
    for scale in (1e6, 1e3, 1):
        x = v / scale
        if x >= 1:
            out.add(f"{round(x):,}")
            if scale == 1e6:
                out.add(f"{x:,.1f}")
    if v >= 1e9:
        out.add(f"{v / 1e9:,.1f} billion")
    return sorted(out)


def build_questions(ticker, title, meta, facts):
    name, fy = short_name(title), meta["report_date"][:4]
    qs = []
    for key, (phrase, concepts, period) in METRICS.items():
        f = find_fact(facts, concepts, meta["accession"], period, meta["report_date"])
        if not f:
            continue
        when = f"as of the end of fiscal {fy}" if period == "instant" else f"for fiscal {fy}"
        qs.append({
            "id": f"{ticker}-{key}", "ticker": ticker, "metric": key,
            "question": f"What was {name}'s {phrase} {when}?",
            "answer_value": f["value"], "unit": f["unit"], "concept": f["concept"],
            "answers": answer_strings(f["value"], f["unit"]),
            "accession": meta["accession"], "period_end": f["end"],
        })
    return qs


def contains_answer(text, answers):
    """True if any accepted answer string appears as a whole number token in text."""
    for a in answers:
        if re.search(r"(?<![\d.,])" + re.escape(a) + r"(?![\d]|,\d|\.\d)", text):
            return True
    return False
