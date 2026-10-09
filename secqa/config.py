from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
INDEX = DATA / "index"
RESULTS = ROOT / "results"

# 25 large US filers across sectors, so the benchmark isn't dominated by one industry's layout.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "ORCL", "CRM", "INTU",   # tech
    "JPM", "BAC", "GS", "V", "MA",                                           # financials
    "XOM", "CVX", "ADM", "BG",                                               # energy / agriculture
    "JNJ", "PFE",                                                            # health care
    "WMT", "COST", "KO", "PG", "CAT",                                        # consumer / industrial
]
