"""
Module: compliance_summary.py

Purpose
-------
Builds a high-level compliance summary from a gap report.

Responsibilities
----------------
- Read a gap report JSON file
- Aggregate obligations by status and severity
- Highlight obligations requiring human review
- Compute a simple overall readiness score
- Save a client-friendly summary JSON
- Provide guided CLI prompts for company, form, and year selection

Inputs
------
outputs/gap_reports/gap_report_<COMPANY>_<FORM>_<YEAR>.json

Outputs
-------
outputs/gap_reports/compliance_summary_<COMPANY>_<FORM>_<YEAR>.json

Pipeline Position
-----------------
Reporting layer on top of the compliance gap analysis workflow.

Notes
-----
This module does not determine legal compliance.
It summarizes evidence-backed gap report results for easier review.
The score is only a workflow aid, not a legal conclusion.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from .config import RAGConfig
from .catalog import build_catalog

cfg = RAGConfig()


def load_gap_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Gap report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_score(results: List[Dict[str, Any]]) -> float:
    """
    Simple readiness score:
    - PASS = 1.0
    - WARN = 0.5
    - REVIEW = 0.25
    - FAIL = 0.0

    This is not a compliance score in the legal sense.
    It is only a workflow prioritization metric.
    """
    if not results:
        return 0.0

    weights = {
        "PASS": 1.0,
        "WARN": 0.5,
        "REVIEW": 0.25,
        "FAIL": 0.0,
    }

    total = 0.0
    for r in results:
        total += weights.get(r.get("status", "REVIEW"), 0.25)

    return round((total / len(results)) * 100, 2)


def summarize_by_severity(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out = {
        "HIGH": {"PASS": 0, "WARN": 0, "REVIEW": 0, "FAIL": 0},
        "MED": {"PASS": 0, "WARN": 0, "REVIEW": 0, "FAIL": 0},
        "LOW": {"PASS": 0, "WARN": 0, "REVIEW": 0, "FAIL": 0},
    }

    for r in results:
        sev = str(r.get("severity", "MED")).upper()
        status = str(r.get("status", "REVIEW")).upper()

        if sev not in out:
            sev = "MED"
        if status not in out[sev]:
            status = "REVIEW"

        out[sev][status] += 1

    return out


def extract_attention_items(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return the most important items for human review:
    - FAIL first
    - then REVIEW
    - then WARN on HIGH severity only
    """
    attention = []

    for r in results:
        status = str(r.get("status", "")).upper()
        severity = str(r.get("severity", "")).upper()

        if status == "FAIL":
            attention.append(r)
        elif status == "REVIEW":
            attention.append(r)
        elif status == "WARN" and severity == "HIGH":
            attention.append(r)

    status_rank = {"FAIL": 0, "REVIEW": 1, "WARN": 2, "PASS": 3}
    severity_rank = {"HIGH": 0, "MED": 1, "LOW": 2}

    attention.sort(
        key=lambda x: (
            status_rank.get(str(x.get("status", "REVIEW")).upper(), 9),
            severity_rank.get(str(x.get("severity", "MED")).upper(), 9),
            str(x.get("rule_id", "")),
        )
    )

    trimmed = []
    for r in attention:
        trimmed.append({
            "rule_id": r.get("rule_id"),
            "rule_name": r.get("rule_name"),
            "severity": r.get("severity"),
            "status": r.get("status"),
            "reason": r.get("reason"),
            "evidence_reason": r.get("evidence_reason"),
        })

    return trimmed


def build_compliance_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    results = report.get("results", [])

    summary = {
        "company": report.get("company"),
        "form": report.get("form"),
        "year": report.get("year"),
        "processed_file": report.get("processed_file"),
        "status_counts": report.get("summary", {}),
        "severity_breakdown": summarize_by_severity(results),
        "overall_readiness_score": safe_score(results),
        "attention_items": extract_attention_items(results),
        "notes": [
            "This summary is a workflow aid, not a legal compliance opinion.",
            "Items marked REVIEW require human verification.",
            "Items marked FAIL indicate a deterministic missing requirement in the processed filing context.",
        ],
    }

    return summary


def save_summary(summary: Dict[str, Any]) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / (
        f"compliance_summary_{summary['company']}_{summary['form'].replace('-', '')}_{summary['year']}.json"
    )
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    cat = build_catalog(cfg)
    if not cat:
        print("No processed filings found. Check cfg.processed_dir:", cfg.processed_dir)
        raise SystemExit(1)

    # show companies
    print("\nAvailable companies (tickers):")
    print(", ".join(sorted(cat.keys())))

    company = input("\nCompany ticker (e.g., VISA): ").strip().upper()
    if company not in cat:
        print(f"Unknown ticker '{company}'. Choose from: {', '.join(sorted(cat.keys()))}")
        raise SystemExit(1)

    # show forms
    forms = sorted(cat[company].keys())
    print(f"\nAvailable forms for {company}: {', '.join(forms)}")
    form_folder = input("Form (10-k or 10-q): ").strip().lower()
    if form_folder not in cat[company]:
        print(f"Invalid form '{form_folder}'. Choose from: {', '.join(forms)}")
        raise SystemExit(1)

    # show years
    years = cat[company][form_folder]
    print(f"\nAvailable years for {company} {form_folder}: {', '.join(years)}")
    year = input("Year: ").strip()
    if year not in years:
        print(f"Invalid year '{year}'. Choose from: {', '.join(years)}")
        raise SystemExit(1)

    form_display = "10-K" if form_folder == "10-k" else "10-Q"
    gap_report_path = cfg.reports_dir / f"gap_report_{company}_{form_display.replace('-', '')}_{year}.json"

    rep = load_gap_report(gap_report_path)
    summary = build_compliance_summary(rep)
    out = save_summary(summary)

    print(f"\nSaved summary -> {out}")
    print("Status counts:", summary["status_counts"])
    print("Overall readiness score:", summary["overall_readiness_score"])
    print("Attention items:", len(summary["attention_items"]))