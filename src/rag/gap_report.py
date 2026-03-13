"""
Module: gap_report.py

Purpose
-------
Generates compliance gap analysis reports by comparing regulatory
obligations against evidence found in company filings and SEC rules.

Responsibilities
----------------
- Load obligations dataset
- Evaluate filing sections deterministically
- Retrieve rule evidence for each obligation
- Retrieve filing evidence for each obligation
- Use deterministic section evidence for presence-type obligations
- Use scoped semantic retrieval for keyword-type obligations
- Assign compliance workflow status (PASS / WARN / REVIEW / FAIL)
- Produce an audit-ready evidence report

Inputs
------
data/obligations/obligations_sec_v1.csv
storage/chunks.jsonl
storage/faiss.index
storage/faiss_rules.index

Outputs
-------
Structured compliance gap report JSON.

Pipeline Position
-----------------
Compliance workflow layer built on top of dual-corpus RAG retrieval.

Notes
-----
This module does not assert legal compliance. It provides evidence-backed
signals for human review and highlights possible disclosure gaps.

Design Decisions
----------------
- Presence checks use deterministic scoped section chunk evidence first.
- Keyword checks use scoped semantic retrieval.
- Rule evidence is filtered to exact citation when rule_citation is known.
- If evidence is missing or weak, status is downgraded to REVIEW.
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

from .catalog import build_catalog
from .config import RAGConfig
from .evidence_pack import evidence_pack_for_obligation
from .filing_evidence import normalize_form, load_scoped_section_chunk_evidence

cfg = RAGConfig()


@dataclass
class Obligation:
    rule_id: str
    rule_name: str
    applies_to: str
    mapped_section: str
    check_type: str
    min_chars: int
    keywords: str
    severity: str
    rule_citation: str
    rule_query: str
    filing_query: str
    notes: str
    evidence_mode: str


def load_obligations(csv_path: Path) -> List[Obligation]:
    """
    Load obligations from CSV.
    """
    obs: List[Obligation] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs.append(
                Obligation(
                    rule_id=row["rule_id"].strip(),
                    rule_name=row["rule_name"].strip(),
                    applies_to=row["applies_to"].strip().upper(),
                    mapped_section=row["mapped_section"].strip().upper(),
                    check_type=row["check_type"].strip().lower(),
                    min_chars=int(row["min_chars"] or 0),
                    keywords=(row.get("keywords") or "").strip(),
                    severity=(row.get("severity") or "MED").strip().upper(),
                    rule_citation=(row.get("rule_citation") or "").strip(),
                    rule_query=(row.get("rule_query") or "").strip(),
                    filing_query=(row.get("filing_query") or "").strip(),
                    evidence_mode=(row.get("evidence_mode") or "rule_strict").strip().lower(),
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return obs


def find_processed_file(company: str, form_folder: str, year: str) -> Path:
    """
    Pick the best matching processed JSON for (company, form_folder, year).
    If multiple files exist, choose the largest one as the best proxy for completeness.
    """
    root = cfg.processed_dir / company.upper() / form_folder.lower()
    if not root.exists():
        raise FileNotFoundError(f"Missing folder: {root}")

    candidates = list(root.glob(f"*{year}*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No processed JSON found for {company} {form_folder} {year} in {root}"
        )

    return max(candidates, key=lambda p: p.stat().st_size)


def load_sections(fp: Path) -> Dict[str, Dict[str, str]]:
    """
    Load extracted sections from a processed filing JSON.
    """
    data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
    sections = data.get("sections", data)

    if not isinstance(sections, dict):
        return {}

    norm = {}
    for k, v in sections.items():
        if isinstance(v, dict):
            txt = v.get("text", "")
            ttl = v.get("title", "")
            if isinstance(txt, str):
                norm[str(k).upper().strip()] = {
                    "title": str(ttl),
                    "text": txt,
                }
    return norm


def normalize_form(value: str) -> str:
    """
    Normalize form strings to a stable format.
    """
    v = str(value or "").strip().lower().replace("_", "-")
    if v in {"10k", "10-k"}:
        return "10-k"
    if v in {"10q", "10-q"}:
        return "10-q"
    return v


def load_scoped_section_chunk_evidence(
    company: str,
    form_folder: str,
    year: str,
    section: str,
    max_hits: int = 3,
) -> List[dict]:
    """
    Deterministically load chunk evidence for a specific filing scope + section.

    This is preferred for presence checks because it avoids semantic retrieval noise.
    """
    hits = []

    if not cfg.chunks_path.exists() or cfg.chunks_path.stat().st_size == 0:
        return hits

    with cfg.chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            rec_company = str(rec.get("company", "")).upper()
            rec_year = str(rec.get("year", ""))
            rec_section = str(rec.get("section", "")).upper().strip()

            rec_form = rec.get("form")
            if rec_form is None:
                rec_form = rec.get("form_type") or rec.get("doc_type") or rec.get("form_folder")
            rec_form = normalize_form(rec_form)

            if rec_company != company.upper():
                continue
            if rec_form != normalize_form(form_folder):
                continue
            if rec_year != str(year):
                continue
            if rec_section != str(section).upper().strip():
                continue

            # write normalized fields back for clean output
            rec["company"] = rec_company
            rec["year"] = rec_year
            rec["section"] = rec_section
            rec["form"] = rec_form

            hits.append({
                "score": 1.0,  # deterministic evidence, not semantic
                "chunk": rec,
            })

            if len(hits) >= max_hits:
                break

    return hits


def evaluate_presence(section_text: str, min_chars: int) -> Tuple[str, str]:
    """
    Deterministic evaluation for section presence.
    """
    if not section_text:
        return "FAIL", "Section missing"
    if min_chars and len(section_text) < min_chars:
        return "WARN", f"Section present but short ({len(section_text)} chars < {min_chars})"
    return "PASS", "Section present"


def evaluate_keyword(section_text: str, keyword_pattern: str) -> Tuple[str, str]:
    """
    Deterministic evaluation for keyword-based obligations.
    """
    if not section_text:
        return "REVIEW", "Section missing; cannot evaluate keyword"
    if not keyword_pattern:
        return "REVIEW", "No keywords configured"

    pat = re.compile(keyword_pattern, flags=re.IGNORECASE)
    if pat.search(section_text):
        return "PASS", f"Keyword match for pattern: {keyword_pattern}"
    return "REVIEW", f"No keyword match for pattern: {keyword_pattern}"


def filter_rule_hits_by_citation(rules_hits: List[dict], rule_citation: str) -> List[dict]:
    """
    Keep exact citation matches first when a rule citation is explicitly known.
    """
    if not rule_citation:
        return rules_hits

    exact = [
        h for h in rules_hits
        if str(h["chunk"].get("citation", "")).strip().lower() == rule_citation.strip().lower()
    ]

    if exact:
        return exact

    return rules_hits


def summarize_rule_evidence(rules_hits: List[dict]) -> List[dict]:
    """
    Keep the top few rule evidence hits in a compact audit-friendly format.
    """
    out = []
    for h in rules_hits[:3]:
        c = h["chunk"]
        out.append({
            "score": round(float(h.get("score", 0.0)), 4),
            "chunk_id": c.get("chunk_id"),
            "citation": c.get("citation"),
            "heading": c.get("heading"),
        })
    return out


def summarize_filing_evidence(filings_hits: List[dict]) -> List[dict]:
    """
    Keep the top few filing evidence hits in a compact audit-friendly format.
    """
    out = []
    for h in filings_hits[:3]:
        c = h["chunk"]
        out.append({
            "score": round(float(h.get("score", 0.0)), 4),
            "chunk_id": c.get("chunk_id"),
            "company": c.get("company"),
            "form": c.get("form"),
            "year": c.get("year"),
            "section": c.get("section"),
        })
    return out


def apply_evidence_guardrails(
    base_status: str,
    rules_hits: List[dict],
    filings_hits: List[dict],
) -> Tuple[str, str]:
    """
    Upgrade/downgrade status based on evidence availability and quality.
    """
    if not rules_hits:
        return "REVIEW", "No rule evidence retrieved"

    if not filings_hits:
        return "REVIEW", "No filing evidence retrieved"

    top_filing_score = float(filings_hits[0].get("score", 0.0))

    if base_status == "PASS" and top_filing_score < 0.40:
        return "REVIEW", f"Filing evidence weak (top score={top_filing_score:.3f})"

    if base_status == "WARN":
        if top_filing_score >= 0.55:
            return "WARN", f"Section short but filing evidence strong (top score={top_filing_score:.3f})"
        return "REVIEW", f"Section short and filing evidence weak (top score={top_filing_score:.3f})"

    return base_status, "Evidence available"


def gap_report(company: str, form_folder: str, year: str) -> Dict[str, Any]:
    """
    Generate an evidence-grounded gap report for a single filing scope.
    """
    if not company or not form_folder or not year:
        raise ValueError("company, form_folder (10-k/10-q), and year are required")

    form_folder = form_folder.lower()
    if form_folder not in {"10-k", "10-q"}:
        raise ValueError("form_folder must be '10-k' or '10-q'")

    applies_to = "10-K" if form_folder == "10-k" else "10-Q"

    obligations = load_obligations(cfg.obligations_csv)
    obligations = [o for o in obligations if o.applies_to == applies_to]

    filing_fp = find_processed_file(company, form_folder, year)
    sections = load_sections(filing_fp)

    report = {
        "company": company.upper(),
        "form": applies_to,
        "year": str(year),
        "processed_file": str(filing_fp),
        "summary": {"PASS": 0, "WARN": 0, "REVIEW": 0, "FAIL": 0},
        "results": [],
        "notes": [],
    }

    if not sections:
        report["notes"].append("No sections loaded from processed filing. Check extraction pipeline/QC.")
        return report

    for ob in obligations:
        sec = ob.mapped_section
        sec_obj = sections.get(sec, {})
        sec_text = sec_obj.get("text", "") if isinstance(sec_obj, dict) else ""

        # 1) Deterministic local evaluation
        if ob.check_type == "presence":
            base_status, base_reason = evaluate_presence(sec_text, ob.min_chars)
        elif ob.check_type == "keyword":
            base_status, base_reason = evaluate_keyword(sec_text, ob.keywords)
        else:
            base_status, base_reason = "REVIEW", f"Unknown check_type: {ob.check_type}"

        # 2) Retrieve evidence pack (rule + semantic filing retrieval)
        pack = evidence_pack_for_obligation(
            description=ob.rule_name,
            mapped_section=ob.mapped_section,
            rule_citation=ob.rule_citation,
            rule_query=ob.rule_query,
            filing_query=ob.filing_query,
            company=company,
            form=form_folder,
            year=year,
            k_rules=4,
            k_filings=6,
        )

        rules_hits = filter_rule_hits_by_citation(pack["rules_hits"], ob.rule_citation)

        # 3) Filing evidence strategy:
        #    - presence -> deterministic scoped section chunks first
        #    - keyword  -> semantic scoped retrieval
        if ob.check_type == "presence":
            filings_hits = load_scoped_section_chunk_evidence(
                company=company,
                form_folder=form_folder,
                year=year,
                section=ob.mapped_section,
                max_hits=3,
            )

            # fallback only if deterministic section evidence is missing
            if not filings_hits:
                filings_hits = pack["filings_hits"]
        else:
            filings_hits = pack["filings_hits"]

        # 4) Apply evidence guardrails
        final_status, evidence_reason = apply_evidence_guardrails(
            base_status, rules_hits, filings_hits
        )

        report["summary"][final_status] += 1

        report["results"].append({
            "rule_id": ob.rule_id,
            "rule_name": ob.rule_name,
            "severity": ob.severity,
            "mapped_section": ob.mapped_section,
            "check_type": ob.check_type,
            "status": final_status,
            "reason": base_reason,
            "evidence_reason": evidence_reason,
            "rule_query": pack["rule_query"],
            "filing_query": pack["filing_query"],
            "rule_evidence": summarize_rule_evidence(rules_hits),
            "filing_evidence": summarize_filing_evidence(filings_hits),
            "evidence_mode": ob.evidence_mode,
            "notes": ob.notes,
        })

    return report


def save_report(rep: Dict[str, Any]) -> Path:
    """
    Save the gap report to disk.
    """
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / f"gap_report_{rep['company']}_{rep['form'].replace('-', '')}_{rep['year']}.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    cat = build_catalog(cfg)
    if not cat:
        print("No processed filings found. Check cfg.processed_dir:", cfg.processed_dir)
        raise SystemExit(1)

    print("\nAvailable companies (tickers):")
    print(", ".join(sorted(cat.keys())))

    company = input("\nCompany ticker (e.g., AAPL): ").strip().upper()
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

    rep = gap_report(company=company, form_folder=form_folder, year=year)
    out = save_report(rep)
    print(f"\nSaved report -> {out}")
    print("Summary:", rep["summary"])