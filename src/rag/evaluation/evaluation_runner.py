"""
Module: evaluation_runner.py

Purpose
-------
Runs deterministic evaluation checks for the regulatory RAG pipeline.

Responsibilities
----------------
- Validate rule retrieval correctness
- Validate scoped filing retrieval correctness
- Validate gap report consistency
- Validate compliance summary consistency
- Save an evaluation report JSON for audit/debugging

Inputs
------
- storage/faiss.index
- storage/faiss_rules.index
- outputs/gap_reports/gap_report_<...>.json
- outputs/gap_reports/compliance_summary_<...>.json

Outputs
-------
outputs/evaluation/evaluation_report_<COMPANY>_<FORM>_<YEAR>.json

Pipeline Position
-----------------
System evaluation / QA layer after retrieval and reporting workflows.

Notes
-----
This module does not test legal correctness.
It tests system behavior, evidence alignment, and output consistency.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from .config import RAGConfig
from .catalog import build_catalog
from .retrieve_filings import retrieve_filings
from .rules_router import retrieve_rules_routed
from .gap_report import gap_report
from .compliance_summary import build_compliance_summary
from .filing_evidence import normalize_form, load_scoped_section_chunk_evidence 

cfg = RAGConfig()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_form(value: str) -> str:
    v = str(value or "").strip().lower().replace("_", "-")
    if v in {"10k", "10-k"}:
        return "10-k"
    if v in {"10q", "10-q"}:
        return "10-q"
    return v


def make_result(name: str, passed: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "check_name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


# ---------------------------------------------------------
# Check 1: rule citation routing
# ---------------------------------------------------------
def check_rule_citation_routing() -> Dict[str, Any]:
    query = "What does 17 CFR 229.105 require?"
    hits = retrieve_rules_routed(query, top_k=10)

    passed = bool(hits) and str(hits[0]["chunk"].get("citation", "")).strip() == "17 CFR 229.105"

    return make_result(
        "rule_citation_routing",
        passed,
        {
            "query": query,
            "top_hit_citation": hits[0]["chunk"].get("citation") if hits else None,
            "top_hit_score": hits[0]["score"] if hits else None,
            "hit_count": len(hits),
        },
    )


# ---------------------------------------------------------
# Check 2: scoped filing retrieval
# ---------------------------------------------------------
def check_scoped_filing_retrieval(company: str, form_folder: str, year: str) -> Dict[str, Any]:
    """
    Validate scoped filing evidence availability using deterministic chunk lookup.

    This is a structural integrity check, not a semantic retrieval check.
    """

    form_norm = normalize_form(form_folder)

    if form_norm == "10-k":
        sections_to_check = [
            {"label": "Risk Factors", "section": "ITEM 1A"},
            {"label": "MD&A", "section": "ITEM 7"},
        ]
    elif form_norm == "10-q":
        sections_to_check = [
            {"label": "MD&A", "section": "ITEM 2"},
        ]
    else:
        return make_result(
            "scoped_filing_retrieval",
            False,
            {"error": f"Unsupported form for evaluation: {form_folder}"},
        )

    all_passed = True
    test_results = []
    problems = []

    for t in sections_to_check:
        hits = load_scoped_section_chunk_evidence(
            company=company,
            form_folder=form_folder,
            year=year,
            section=t["section"],
            max_hits=5,
        )

        test_ok = len(hits) > 0
        if not test_ok:
            all_passed = False
            problems.append({
                "label": t["label"],
                "section": t["section"],
                "issue": "No scoped deterministic chunk evidence found",
            })

        test_results.append({
            "label": t["label"],
            "section": t["section"],
            "hit_count": len(hits),
            "status": "PASS" if test_ok else "FAIL",
        })

    return make_result(
        "scoped_filing_retrieval",
        all_passed,
        {
            "requested_scope": {
                "company": company,
                "form": form_folder,
                "year": year,
            },
            "tests": test_results,
            "problems": problems,
        },
    )

# ---------------------------------------------------------
# Check 3: gap report basic consistency
# ---------------------------------------------------------
def check_gap_report_consistency(rep: Dict[str, Any]) -> Dict[str, Any]:
    results = rep.get("results", [])
    summary = rep.get("summary", {})

    recomputed = {"PASS": 0, "WARN": 0, "REVIEW": 0, "FAIL": 0}
    for r in results:
        status = str(r.get("status", "REVIEW")).upper()
        if status not in recomputed:
            status = "REVIEW"
        recomputed[status] += 1

    passed = recomputed == summary

    return make_result(
        "gap_report_consistency",
        passed,
        {
            "reported_summary": summary,
            "recomputed_summary": recomputed,
            "results_count": len(results),
        },
    )


# ---------------------------------------------------------
# Check 4: gap report evidence integrity
# ---------------------------------------------------------
def check_gap_report_evidence_integrity(rep: Dict[str, Any], company: str, form_folder: str, year: str) -> Dict[str, Any]:
    problems = []

    for r in rep.get("results", []):
        # filing evidence must be in scope
        for fe in r.get("filing_evidence", []):
            hit_company = str(fe.get("company", "")).upper()
            hit_form = normalize_form(fe.get("form"))
            hit_year = str(fe.get("year", ""))

            if not (
                hit_company == company.upper()
                and hit_form == normalize_form(form_folder)
                and hit_year == str(year)
            ):
                problems.append({
                    "rule_id": r.get("rule_id"),
                    "type": "filing_scope_mismatch",
                    "evidence": fe,
                })

        # if explicit rule citation exists in row evidence, exact matches should be preferred
        # we only flag if a row has rule evidence but none of them match exact citation
    rule_evidence = r.get("rule_evidence", [])
    rule_query = str(r.get("rule_query", ""))
    evidence_mode = str(r.get("evidence_mode", "rule_strict")).lower()
    expected_citation = None

    if rule_query.startswith("17 CFR "):
        expected_citation = rule_query.split(" disclosure requirements")[0].strip()

    # Only enforce exact citation for strict rule-backed obligations
    if evidence_mode == "rule_strict" and expected_citation and rule_evidence:
        exact = [
            x for x in rule_evidence
            if str(x.get("citation", "")).strip() == expected_citation
        ]
        if not exact:
            problems.append({
                "rule_id": r.get("rule_id"),
                "type": "rule_citation_not_exact",
                "expected_citation": expected_citation,
                "rule_evidence": rule_evidence,
            })
    passed = len(problems) == 0

    return make_result(
        "gap_report_evidence_integrity",
        passed,
        {
            "problem_count": len(problems),
            "problems": problems[:10],
        },
    )


# ---------------------------------------------------------
# Check 5: summary consistency
# ---------------------------------------------------------
def check_summary_consistency(rep: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    passed = True
    issues = []

    if summary.get("company") != rep.get("company"):
        passed = False
        issues.append("company mismatch")

    if summary.get("form") != rep.get("form"):
        passed = False
        issues.append("form mismatch")

    if summary.get("year") != rep.get("year"):
        passed = False
        issues.append("year mismatch")

    if summary.get("status_counts") != rep.get("summary"):
        passed = False
        issues.append("status_counts mismatch")

    return make_result(
        "summary_consistency",
        passed,
        {
            "issues": issues,
            "summary_status_counts": summary.get("status_counts"),
            "gap_report_summary": rep.get("summary"),
        },
    )


# ---------------------------------------------------------
# Check 6: attention items correctness
# ---------------------------------------------------------
def check_attention_items(summary: Dict[str, Any]) -> Dict[str, Any]:
    bad_items = []

    for item in summary.get("attention_items", []):
        status = str(item.get("status", "")).upper()
        severity = str(item.get("severity", "")).upper()

        allowed = (
            status == "FAIL"
            or status == "REVIEW"
            or (status == "WARN" and severity == "HIGH")
        )

        if not allowed:
            bad_items.append(item)

    return make_result(
        "attention_items_logic",
        len(bad_items) == 0,
        {
            "bad_items_count": len(bad_items),
            "bad_items": bad_items,
        },
    )


def build_evaluation_report(company: str, form_folder: str, year: str) -> Dict[str, Any]:
    rep = gap_report(company=company, form_folder=form_folder, year=year)
    summary = build_compliance_summary(rep)

    results = [
        check_rule_citation_routing(),
        check_scoped_filing_retrieval(company, form_folder, year),
        check_gap_report_consistency(rep),
        check_gap_report_evidence_integrity(rep, company, form_folder, year),
        check_summary_consistency(rep, summary),
        check_attention_items(summary),
    ]

    checks_run = len(results)
    checks_passed = sum(1 for r in results if r["status"] == "PASS")
    checks_failed = checks_run - checks_passed

    overall_status = "PASS" if checks_failed == 0 else "FAIL"

    return {
        "company": company.upper(),
        "form": "10-K" if form_folder == "10-k" else "10-Q",
        "year": str(year),
        "overall_status": overall_status,
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "results": results,
        "notes": [
            "This evaluation checks system behavior, not legal correctness.",
            "Failed checks indicate pipeline or logic issues that should be fixed before relying on outputs.",
        ],
    }


def save_evaluation_report(report: Dict[str, Any]) -> Path:
    out_dir = cfg.reports_dir.parent / "evaluation"
    ensure_dir(out_dir)

    out = out_dir / (
        f"evaluation_report_{report['company']}_{report['form'].replace('-', '')}_{report['year']}.json"
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    cat = build_catalog(cfg)
    if not cat:
        print("No processed filings found. Check cfg.processed_dir:", cfg.processed_dir)
        raise SystemExit(1)

    print("\nAvailable companies (tickers):")
    print(", ".join(sorted(cat.keys())))

    company = input("\nCompany ticker (e.g., VISA): ").strip().upper()
    if company not in cat:
        print(f"Unknown ticker '{company}'. Choose from: {', '.join(sorted(cat.keys()))}")
        raise SystemExit(1)

    forms = sorted(cat[company].keys())
    print(f"\nAvailable forms for {company}: {', '.join(forms)}")
    form_folder = input("Form (10-k or 10-q): ").strip().lower()
    if form_folder not in cat[company]:
        print(f"Invalid form '{form_folder}'. Choose from: {', '.join(forms)}")
        raise SystemExit(1)

    years = cat[company][form_folder]
    print(f"\nAvailable years for {company} {form_folder}: {', '.join(years)}")
    year = input("Year: ").strip()
    if year not in years:
        print(f"Invalid year '{year}'. Choose from: {', '.join(years)}")
        raise SystemExit(1)

    report = build_evaluation_report(company, form_folder, year)
    out = save_evaluation_report(report)

    print(f"\nSaved evaluation report -> {out}")
    print("Overall status:", report["overall_status"])
    print("Checks run:", report["checks_run"])
    print("Checks passed:", report["checks_passed"])
    print("Checks failed:", report["checks_failed"])