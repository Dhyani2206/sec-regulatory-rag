"""
Module: batch_evaluation_runner.py

Purpose
-------
Runs evaluation across all available company / form / year combinations
in the processed filings catalog.

Responsibilities
----------------
- Discover all available filing scopes from the catalog
- Run evaluation_runner for each scope
- Aggregate pass/fail counts across the dataset
- Summarize check-level reliability
- Save a batch evaluation report JSON

Inputs
------
- processed filings catalog
- evaluation_runner.py
- all retrieval / reporting dependencies used by evaluation_runner

Outputs
-------
outputs/evaluation/batch_evaluation_report.json

Pipeline Position
-----------------
System-wide QA / validation layer after individual report generation
and evaluation workflows.

Notes
-----
This module evaluates pipeline behavior across the dataset.
It does not judge legal correctness of the filings themselves.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from rag.config import RAGConfig
from rag.catalog import build_catalog
from .evaluation_runner import build_evaluation_report

cfg = RAGConfig()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_check_stats() -> Dict[str, Dict[str, int]]:
    return {
        "rule_citation_routing": {"PASS": 0, "FAIL": 0},
        "scoped_filing_retrieval": {"PASS": 0, "FAIL": 0},
        "gap_report_consistency": {"PASS": 0, "FAIL": 0},
        "gap_report_evidence_integrity": {"PASS": 0, "FAIL": 0},
        "summary_consistency": {"PASS": 0, "FAIL": 0},
        "attention_items_logic": {"PASS": 0, "FAIL": 0},
    }


def build_scope_list(catalog: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, str]]:
    scopes = []
    for company in sorted(catalog.keys()):
        for form_folder in sorted(catalog[company].keys()):
            for year in sorted(catalog[company][form_folder]):
                scopes.append({
                    "company": company,
                    "form_folder": form_folder,
                    "year": str(year),
                })
    return scopes


def aggregate_results(run_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_runs = len(run_reports)
    passed_runs = sum(1 for r in run_reports if r["overall_status"] == "PASS")
    failed_runs = total_runs - passed_runs

    check_stats = init_check_stats()
    failed_scopes = []

    for rep in run_reports:
        if rep["overall_status"] == "FAIL":
            failed_scopes.append({
                "company": rep["company"],
                "form": rep["form"],
                "year": rep["year"],
            })

        for check in rep.get("results", []):
            name = check.get("check_name")
            status = check.get("status", "FAIL")
            if name not in check_stats:
                check_stats[name] = {"PASS": 0, "FAIL": 0}
            if status not in check_stats[name]:
                check_stats[name][status] = 0
            check_stats[name][status] += 1

    check_pass_rates = {}
    for check_name, stats in check_stats.items():
        total = stats.get("PASS", 0) + stats.get("FAIL", 0)
        rate = round((stats.get("PASS", 0) / total) * 100, 2) if total else 0.0
        check_pass_rates[check_name] = rate

    return {
        "total_runs": total_runs,
        "passed_runs": passed_runs,
        "failed_runs": failed_runs,
        "dataset_pass_rate": round((passed_runs / total_runs) * 100, 2) if total_runs else 0.0,
        "check_stats": check_stats,
        "check_pass_rates": check_pass_rates,
        "failed_scopes": failed_scopes,
    }


def build_batch_evaluation_report() -> Dict[str, Any]:
    catalog = build_catalog(cfg)
    scopes = build_scope_list(catalog)

    run_reports = []
    run_errors = []

    for scope in scopes:
        company = scope["company"]
        form_folder = scope["form_folder"]
        year = scope["year"]

        print(f"[batch-eval] Running: {company} | {form_folder} | {year}")

        try:
            rep = build_evaluation_report(company, form_folder, year)
            run_reports.append(rep)
        except Exception as e:
            run_errors.append({
                "company": company,
                "form_folder": form_folder,
                "year": year,
                "error": f"{type(e).__name__}: {e}",
            })

    aggregate = aggregate_results(run_reports)

    report = {
        "overall_status": "PASS" if aggregate["failed_runs"] == 0 and not run_errors else "FAIL",
        "catalog_size": len(scopes),
        "successful_runs": len(run_reports),
        "run_errors_count": len(run_errors),
        "aggregate": aggregate,
        "run_errors": run_errors[:50],
        "notes": [
            "This batch evaluation checks pipeline behavior across all available scopes.",
            "A FAIL indicates at least one scope failed evaluation or raised an execution error.",
            "This is a system validation report, not a legal compliance determination.",
        ],
    }

    return report


def save_batch_report(report: Dict[str, Any]) -> Path:
    out_dir = cfg.reports_dir.parent / "evaluation"
    ensure_dir(out_dir)

    out = out_dir / "batch_evaluation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    report = build_batch_evaluation_report()
    out = save_batch_report(report)

    print(f"\nSaved batch evaluation report -> {out}")
    print("Overall status:", report["overall_status"])
    print("Catalog size:", report["catalog_size"])
    print("Successful runs:", report["successful_runs"])
    print("Run errors:", report["run_errors_count"])
    print("Dataset pass rate:", report["aggregate"]["dataset_pass_rate"])