"""
Module: semantic_evaluation_runner.py

Purpose
-------
Runs semantic evaluation cases for the regulatory RAG pipeline.

Responsibilities
----------------
- Load a curated semantic evaluation dataset
- Evaluate rule-only retrieval behavior
- Evaluate scoped filing retrieval behavior
- Support strict, heuristic, and refusal-style cases
- Save a semantic evaluation report JSON

Inputs
------
- tests/semantic_eval_cases.json
- storage/faiss.index
- storage/faiss_rules.index
- retrieval modules

Outputs
-------
outputs/evaluation/semantic_evaluation_report.json

Pipeline Position
-----------------
Semantic QA / retrieval quality validation layer on top of the
core pipeline integrity checks.

Notes
-----
This module evaluates retrieval behavior and grounding quality.
It does not make legal judgments and should not be interpreted
as a compliance certification.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import RAGConfig
from .rules_router import retrieve_rules_routed
from .retrieve_filing_sections import retrieve_filing_sections
from .query_router import route_filing_query
from .filing_evidence import load_scoped_section_chunk_evidence

cfg = RAGConfig()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Semantic evaluation cases file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Semantic evaluation cases file must contain a JSON list.")
    return data


def normalize_form(value: str) -> str:
    v = str(value or "").strip().lower().replace("_", "-")
    if v in {"10k", "10-k"}:
        return "10-k"
    if v in {"10q", "10-q"}:
        return "10-q"
    return v


def make_case_result(
    case: Dict[str, Any],
    passed: bool,
    observed: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "mode": case.get("mode"),
        "type": case.get("type"),
        "status": "PASS" if passed else "FAIL",
        "query": case.get("query"),
        "reason": reason,
        "expected": {
            "expected_rule_citation": case.get("expected_rule_citation"),
            "expected_section": case.get("expected_section"),
            "company": case.get("company"),
            "form_folder": case.get("form_folder"),
            "year": case.get("year"),
            "allow_no_hits": case.get("allow_no_hits", False),
        },
        "observed": observed,
    }


def evaluate_rule_only_case(case: Dict[str, Any]) -> Dict[str, Any]:
    query = case["query"]
    expected_citation = case.get("expected_rule_citation", "")

    hits = retrieve_rules_routed(query, top_k=10)

    top_hit_citation = None
    top_hit_score = None
    if hits:
        top_hit_citation = hits[0]["chunk"].get("citation")
        top_hit_score = hits[0]["score"]

    exact_in_top3 = any(
        str(h["chunk"].get("citation", "")).strip() == expected_citation
        for h in hits[:3]
    )

    passed = bool(expected_citation) and exact_in_top3

    observed = {
        "hit_count": len(hits),
        "top_hit_citation": top_hit_citation,
        "top_hit_score": top_hit_score,
        "top3_citations": [h["chunk"].get("citation") for h in hits[:3]],
    }

    reason = (
        f"Expected citation found in top-3: {expected_citation}"
        if passed else
        f"Expected citation not found in top-3: {expected_citation}"
    )

    return make_case_result(case, passed, observed, reason)


def evaluate_filing_scoped_case(case: Dict[str, Any]) -> Dict[str, Any]:
    query = case["query"]
    company = case.get("company")
    form_folder = case.get("form_folder")
    year = str(case.get("year"))
    expected_section = str(case.get("expected_section", "")).upper().strip()
    allowed_sections = [str(x).upper().strip() for x in case.get("allowed_sections", [])]
    mode = str(case.get("mode", "strict")).lower()
    allow_no_hits = bool(case.get("allow_no_hits", False))

    route = route_filing_query(query, form_folder)
    accepted_sections = [expected_section] + allowed_sections if expected_section else allowed_sections

    # 1) Not applicable / refusal path
    if route["intent"] == "not_applicable":
        passed = allow_no_hits or mode == "refusal"
        return make_case_result(
            case,
            passed,
            observed={
                "route": route,
                "hit_count": 0,
                "top_hit_section": None,
                "top_hit_score": None,
                "top5_sections": [],
                "top5_ranked_sections": [],
            },
            reason="Query routed to not_applicable path."
        )

    # 2) Structural section lookup path
    if route["intent"] == "structural_section_lookup":
        target_section = str(route["target_section"]).upper().strip()
        hits = load_scoped_section_chunk_evidence(
            company=company,
            form_folder=form_folder,
            year=year,
            section=target_section,
            max_hits=5,
        )

        observed_sections = [target_section] if hits else []
        passed = bool(hits) and target_section in accepted_sections

        observed = {
            "route": route,
            "hit_count": len(hits),
            "top_hit_section": target_section if hits else None,
            "top_hit_score": 1.0 if hits else None,
            "top5_sections": observed_sections,
            "top5_ranked_sections": [
                {
                    "company": company,
                    "form": form_folder,
                    "year": year,
                    "section": target_section,
                    "score": 1.0,
                    "adjusted_score": 1.0,
                }
            ] if hits else [],
        }

        reason = (
            f"Structural route resolved to {target_section} with deterministic evidence."
            if passed else
            f"Structural route resolved to {target_section}, but no deterministic evidence was found."
        )

        return make_case_result(case, passed, observed, reason)

    # 3) Semantic topic lookup path
    hits = retrieve_filing_sections(
        query=query,
        company=company,
        form=form_folder,
        year=year,
        top_k=5,
    )

    observed_sections = [str(h.get("section", "")).upper().strip() for h in hits]
    top_hit_section = observed_sections[0] if observed_sections else None
    top_hit_score = hits[0]["adjusted_score"] if hits else None

    if allow_no_hits and len(hits) == 0:
        passed = True
        reason = "No hits allowed for this refusal-style case."
    else:
        if mode == "strict":
            passed = bool(hits) and any(sec in observed_sections[:3] for sec in accepted_sections)
            reason = (
                f"Expected/allowed section found in top-3: {accepted_sections}"
                if passed else
                f"Expected/allowed section not found in top-3: {accepted_sections}"
            )
        elif mode == "heuristic":
            passed = bool(hits) and any(sec in observed_sections for sec in accepted_sections)
            reason = (
                f"Expected/allowed section found somewhere in ranked sections: {accepted_sections}"
                if passed else
                f"Expected/allowed section not found in ranked sections: {accepted_sections}"
            )
        elif mode == "refusal":
            passed = (len(hits) == 0) or any(sec in observed_sections[:3] for sec in accepted_sections)
            reason = (
                "Refusal-style case behaved acceptably."
                if passed else
                "Refusal-style case returned unrelated scoped sections."
            )
        else:
            passed = False
            reason = f"Unsupported evaluation mode: {mode}"

    observed = {
        "route": route,
        "hit_count": len(hits),
        "top_hit_section": top_hit_section,
        "top_hit_score": top_hit_score,
        "top5_sections": observed_sections[:5],
        "top5_ranked_sections": hits[:5],
    }

    return make_case_result(case, passed, observed, reason)


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case_type = str(case.get("type", "")).lower()

    if case_type == "rule_only":
        return evaluate_rule_only_case(case)

    if case_type == "filing_scoped":
        return evaluate_filing_scoped_case(case)

    return make_case_result(
        case,
        False,
        observed={},
        reason=f"Unsupported case type: {case_type}",
    )


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r["status"] == "PASS")
    failed_cases = total_cases - passed_cases

    by_mode = {}
    by_type = {}

    for r in results:
        mode = str(r.get("mode", "unknown")).lower()
        case_type = str(r.get("type", "unknown")).lower()
        status = r.get("status", "FAIL")

        if mode not in by_mode:
            by_mode[mode] = {"PASS": 0, "FAIL": 0}
        if case_type not in by_type:
            by_type[case_type] = {"PASS": 0, "FAIL": 0}

        by_mode[mode][status] += 1
        by_type[case_type][status] += 1

    mode_pass_rates = {}
    for mode, stats in by_mode.items():
        total = stats["PASS"] + stats["FAIL"]
        mode_pass_rates[mode] = round((stats["PASS"] / total) * 100, 2) if total else 0.0

    type_pass_rates = {}
    for case_type, stats in by_type.items():
        total = stats["PASS"] + stats["FAIL"]
        type_pass_rates[case_type] = round((stats["PASS"] / total) * 100, 2) if total else 0.0

    failures = [r for r in results if r["status"] == "FAIL"]

    return {
        "overall_status": "PASS" if failed_cases == 0 else "FAIL",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "overall_pass_rate": round((passed_cases / total_cases) * 100, 2) if total_cases else 0.0,
        "by_mode": by_mode,
        "by_type": by_type,
        "mode_pass_rates": mode_pass_rates,
        "type_pass_rates": type_pass_rates,
        "failures": failures,
    }


def build_semantic_evaluation_report(cases_path: Optional[Path] = None) -> Dict[str, Any]:
    if cases_path is None:
        cases_path = Path("tests") / "semantic_eval_cases.json"

    cases = load_cases(cases_path)
    results = [evaluate_case(case) for case in cases]
    summary = summarize_results(results)

    return {
        "cases_file": str(cases_path),
        "overall_status": summary["overall_status"],
        "total_cases": summary["total_cases"],
        "passed_cases": summary["passed_cases"],
        "failed_cases": summary["failed_cases"],
        "overall_pass_rate": summary["overall_pass_rate"],
        "by_mode": summary["by_mode"],
        "by_type": summary["by_type"],
        "mode_pass_rates": summary["mode_pass_rates"],
        "type_pass_rates": summary["type_pass_rates"],
        "results": results,
        "notes": [
            "Strict cases require strong expected evidence alignment.",
            "Heuristic cases allow looser matching within scoped retrieval.",
            "Refusal cases allow no-hit behavior when evidence is weak or unsupported.",
            "This report evaluates semantic retrieval behavior, not legal compliance correctness.",
        ],
    }


def save_semantic_report(report: Dict[str, Any]) -> Path:
    out_dir = cfg.reports_dir.parent / "evaluation"
    ensure_dir(out_dir)

    out = out_dir / "semantic_evaluation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    report = build_semantic_evaluation_report()
    out = save_semantic_report(report)

    print(f"\nSaved semantic evaluation report -> {out}")
    print("Overall status:", report["overall_status"])
    print("Total cases:", report["total_cases"])
    print("Passed cases:", report["passed_cases"])
    print("Failed cases:", report["failed_cases"])
    print("Overall pass rate:", report["overall_pass_rate"])