import re
from pathlib import Path
from typing import Dict, List, Set

from .config import RAGConfig

YEAR_RE = re.compile(r"(19|20)\d{2}")

def _extract_year_from_name(name: str) -> str:
    m = YEAR_RE.search(name)
    return m.group(0) if m else ""

def build_catalog(cfg: RAGConfig) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
      {
        "AAPL": {"10-k": ["2021","2022",...], "10-q": ["2021",...]}
      }
    """
    root = cfg.processed_dir
    catalog: Dict[str, Dict[str, Set[str]]] = {}

    if not root.exists():
        return {}

    for ticker_dir in root.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        catalog.setdefault(ticker, {})

        for form_dir in ticker_dir.iterdir():
            if not form_dir.is_dir():
                continue
            form = form_dir.name.lower()  # "10-k" / "10-q"
            years: Set[str] = set()

            for fp in form_dir.glob("*.json"):
                y = _extract_year_from_name(fp.name)
                if y:
                    years.add(y)

            if years:
                catalog[ticker][form] = years

    # sort years
    sorted_catalog: Dict[str, Dict[str, List[str]]] = {}
    for t, forms in catalog.items():
        sorted_catalog[t] = {}
        for form, years in forms.items():
            sorted_catalog[t][form] = sorted(years)

    return sorted_catalog

def list_tickers(cfg: RAGConfig) -> List[str]:
    cat = build_catalog(cfg)
    return sorted(cat.keys())

def list_years(cfg: RAGConfig, ticker: str, form: str) -> List[str]:
    cat = build_catalog(cfg)
    return cat.get(ticker.upper(), {}).get(form.lower(), [])