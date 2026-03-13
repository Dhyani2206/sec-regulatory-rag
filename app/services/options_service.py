from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.api.v1.schemas.options import CompanyOption, OptionsResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_JSONL_PATH = PROJECT_ROOT / "storage" / "chunks.jsonl"

CHUNK_ID_RE = re.compile(
    r"^(?P<company>[^|]+)\|(?P<form>10K|10Q)\|(?P<year>\d{4})\|(?P<section>[^|]+)\|",
    re.IGNORECASE,
)

def _normalize_ticker(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None

def _normalize_form(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text == "10K":
        return "10-K"
    if text == "10Q":
        return "10-Q"
    return text or None

def _normalize_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _extract_from_chunk_id(chunk_id: Any) -> Dict[str, Any]:
    if chunk_id is None:
        return {}

    text = str(chunk_id).strip()
    m = CHUNK_ID_RE.match(text)
    if not m:
        return {}
    return {
        "company": _normalize_ticker(m.group("company")),
        "form": _normalize_form(m.group("form")),
        "year": _normalize_year(m.group("year")),
    }

def _load_chunk_records() -> List[Dict[str, Any]]:
    path = CHUNKS_JSONL_PATH
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    records.append(row)
    except OSError:
        return []
    return records

def get_available_options() -> OptionsResponse:
    records = _load_chunk_records()

    company_map: Dict[str, Dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for row in records:
        company = _normalize_ticker(
            row.get("ticker")
            or row.get("company")
            or row.get("symbol")
        )
        form = _normalize_form(
            row.get("form")
            or row.get("form_folder")
            or row.get("filing_type")
        )
        year = _normalize_year(row.get("year"))

        # Fallback to chunk_id parsing if needed
        if not company or not form or year is None:
            parsed = _extract_from_chunk_id(row.get("chunk_id"))
            company = company or parsed.get("company")
            form = form or parsed.get("form")
            year = year if year is not None else parsed.get("year")

        if not company or not form or year is None:
            continue

        company_map[company][form].add(year)

    companies: List[CompanyOption] = []

    for company in sorted(company_map.keys()):
        available_forms = {
            form: sorted(years)
            for form, years in sorted(company_map[company].items())
        }
        companies.append(
            CompanyOption(
                ticker=company,
                available_forms=available_forms,
            )
        )

    return OptionsResponse(companies=companies)

def is_valid_scope(
    company: Optional[str],
    form: Optional[str],
    year: Optional[int],
) -> bool:
    if company is None or form is None or year is None:
        return True

    company_norm = _normalize_ticker(company)
    form_norm = _normalize_form(form)
    year_norm = _normalize_year(year)

    if company_norm is None or form_norm is None or year_norm is None:
        return False

    options = get_available_options()

    for item in options.companies:
        if item.ticker != company_norm:
            continue
        valid_years = item.available_forms.get(form_norm, [])
        return year_norm in valid_years

    return False