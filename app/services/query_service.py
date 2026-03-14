from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.api.v1.schemas.query import (
    EvidenceItem,
    EvidencePreviewItem,
    QueryRequest,
    QueryResponse,
    ReviewResult,
    SourceItem,
    TechnicalTrace,
)
from app.services.options_service import get_available_options, is_valid_scope
from src.rag.answer_engine import answer_query


TEXT_KEYS = [
    "text",
    "content",
    "chunk_text",
    "body",
    "passage",
    "section_text",
    "raw_text",
]

FORM_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b10[\s\-]?k\b", re.IGNORECASE), "10-K"),
    (re.compile(r"\b10[\s\-]?q\b", re.IGNORECASE), "10-Q"),
)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_text(item: Dict[str, Any]) -> Optional[str]:
    for key in TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_upper_form(form: Optional[str]) -> Optional[str]:
    return form.upper() if isinstance(form, str) and form.strip() else form


def _clean_whitespace(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return " ".join(text.split())


def _make_snippet(text: Optional[str], max_chars: int = 320) -> Optional[str]:
    text = _clean_whitespace(text)
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _build_source_label(
    company: Optional[str],
    form: Optional[str],
    year: Optional[int],
    section: Optional[str],
) -> str:
    parts: List[str] = []

    if company:
        parts.append(str(company))
    if form:
        parts.append(_safe_upper_form(form) or str(form))
    if year:
        parts.append(str(year))
    if section:
        parts.append(str(section))

    return ", ".join(parts) if parts else "Retrieved evidence"


def _normalize_route(route: Any) -> Any:
    if isinstance(route, (str, dict)) or route is None:
        return route
    return str(route)


def _normalize_guard_reason(guard_reason: Any) -> Any:
    if isinstance(guard_reason, (str, dict)) or guard_reason is None:
        return guard_reason
    return str(guard_reason)


def _to_evidence_item(item: Any, bucket: Optional[str] = None) -> EvidenceItem:
    if isinstance(item, EvidenceItem):
        return item

    if isinstance(item, dict):
        text_value = _extract_text(item)

        known_fields = {
            "source_type",
            "citation",
            "section",
            "company",
            "form",
            "form_folder",
            "year",
            "chunk_id",
            "score",
            "text",
            "content",
            "chunk_text",
            "body",
            "passage",
            "section_text",
            "raw_text",
        }

        metadata = {k: v for k, v in item.items() if k not in known_fields}
        if bucket:
            metadata["bucket"] = bucket

        return EvidenceItem(
            source_type=item.get("source_type"),
            citation=item.get("citation"),
            section=item.get("section"),
            company=item.get("company"),
            form=item.get("form") or item.get("form_folder"),
            year=item.get("year"),
            chunk_id=item.get("chunk_id"),
            score=item.get("score"),
            text=text_value,
            metadata=metadata,
        )

    return EvidenceItem(
        text=str(item),
        metadata={
            "raw_type": type(item).__name__,
            **({"bucket": bucket} if bucket else {}),
        },
    )


def _normalize_evidence(raw_evidence: Any) -> List[EvidenceItem]:
    normalized: List[EvidenceItem] = []

    if raw_evidence is None:
        return normalized

    if isinstance(raw_evidence, dict):
        for bucket_name, bucket_value in raw_evidence.items():
            if bucket_value is None:
                continue

            if isinstance(bucket_value, list):
                for item in bucket_value:
                    normalized.append(_to_evidence_item(item, bucket=bucket_name))
            else:
                normalized.append(_to_evidence_item(bucket_value, bucket=bucket_name))

        return normalized

    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            normalized.append(_to_evidence_item(item))
        return normalized

    normalized.append(_to_evidence_item(raw_evidence))
    return normalized


def _dedupe_sources(evidence: List[EvidenceItem], max_sources: int = 3) -> List[SourceItem]:
    sources: List[SourceItem] = []
    seen = set()

    for item in evidence:
        label = _build_source_label(item.company, item.form, item.year, item.section)
        key = (label, item.company, item.form, item.year, item.section)

        if key in seen:
            continue
        seen.add(key)

        sources.append(
            SourceItem(
                label=label,
                section=item.section,
                company=item.company,
                form=_safe_upper_form(item.form),
                year=item.year,
            )
        )

        if len(sources) >= max_sources:
            break

    return sources


def _build_evidence_preview(
    evidence: List[EvidenceItem],
    max_items: int = 3,
) -> List[EvidencePreviewItem]:
    previews: List[EvidencePreviewItem] = []

    for item in evidence[:max_items]:
        previews.append(
            EvidencePreviewItem(
                label=_build_source_label(item.company, item.form, item.year, item.section),
                snippet=_make_snippet(item.text),
                section=item.section,
                score=item.score,
            )
        )

    return previews


def _infer_evidence_strength(evidence: List[EvidenceItem], guard_reason: Any) -> str:
    guard_text = str(guard_reason).lower() if guard_reason is not None else ""

    if "deterministic" in guard_text and evidence:
        return "strong"

    if len(evidence) >= 3:
        return "strong"

    if len(evidence) >= 1:
        return "moderate"

    return "not_found"


def _infer_confidence(evidence_strength: str, status: str) -> str:
    status_upper = (status or "").upper()

    if status_upper == "PASS" and evidence_strength == "strong":
        return "high"
    if status_upper == "PASS" and evidence_strength == "moderate":
        return "medium"
    if evidence_strength == "not_found":
        return "low"
    return "medium"


def _infer_review_result(
    status: str,
    route: Any,
    guard_reason: Any,
    evidence: List[EvidenceItem],
) -> ReviewResult:
    route_intent = None
    if isinstance(route, dict):
        route_intent = route.get("intent")
    elif isinstance(route, str):
        route_intent = route

    evidence_strength = _infer_evidence_strength(evidence, guard_reason)
    confidence = _infer_confidence(evidence_strength, status)

    if status.upper() == "REFUSE" and route_intent == "invalid_scope":
        return ReviewResult(
            finding="invalid_scope",
            label="Requested filing scope is not available",
            confidence="high",
            evidence_strength="not_found",
        )

    if evidence_strength == "not_found":
        return ReviewResult(
            finding="manual_review_recommended",
            label="No strong supporting evidence surfaced",
            confidence="low",
            evidence_strength="not_found",
        )

    if route_intent == "structural_section_lookup" and evidence_strength == "strong":
        return ReviewResult(
            finding="section_found",
            label="Relevant disclosure section located",
            confidence="high",
            evidence_strength="strong",
        )

    if route_intent == "rule_only" and evidence_strength in {"strong", "moderate"}:
        return ReviewResult(
            finding="rule_found",
            label="Relevant regulatory rule evidence located",
            confidence=confidence,
            evidence_strength=evidence_strength,
        )

    if evidence_strength == "strong":
        return ReviewResult(
            finding="supported_answer",
            label="Answer supported by retrieved evidence",
            confidence=confidence,
            evidence_strength=evidence_strength,
        )

    return ReviewResult(
        finding="manual_review_recommended",
        label="Relevant evidence found, but human review is recommended",
        confidence=confidence,
        evidence_strength=evidence_strength,
    )


def _build_plain_english_explanation(
    route: Any,
    guard_reason: Any,
    review_result: ReviewResult,
    evidence: List[EvidenceItem],
) -> str:
    route_intent = None
    target_section = None

    if isinstance(route, dict):
        route_intent = route.get("intent")
        target_section = route.get("target_section")
    elif isinstance(route, str):
        route_intent = route

    if route_intent == "structural_section_lookup" and target_section and not evidence:
        return (
            f"The system treated this as a filing-structure question and mapped it to "
            f"{target_section}, but it did not surface strong supporting evidence in the selected filing."
        )

    if "deterministic" in str(guard_reason).lower():
        return (
            "The answer is grounded in deterministic retrieved evidence rather than a free-form model guess."
        )

    if evidence:
        return (
            "The answer is based on retrieved regulatory or filing evidence and presented conservatively."
        )

    return (
        "The system could not surface strong evidence, so this result should be treated as review guidance only."
    )


def _build_why_it_matters(query: str, route: Any, evidence: List[EvidenceItem]) -> str:
    q = (query or "").lower()

    if "risk factor" in q or "risk factors" in q:
        return (
            "This helps compliance, legal, audit, and research teams quickly verify where "
            "material risk disclosures appear in official filings."
        )

    route_intent = route.get("intent") if isinstance(route, dict) else str(route).lower() if route else ""

    if "structural" in str(route_intent):
        return (
            "This reduces manual searching across long SEC filings by pointing reviewers to the most relevant section first."
        )

    if evidence:
        return (
            "This helps users trace answers back to filing or rule evidence instead of relying on unsupported summaries."
        )

    return (
        "This helps users identify when human review is needed instead of over-trusting an unsupported answer."
    )


def _build_review_guidance(
    review_result: ReviewResult,
    route: Any,
    evidence: List[EvidenceItem],
) -> List[str]:
    guidance: List[str] = []

    route_intent = route.get("intent") if isinstance(route, dict) else route

    if review_result.finding == "section_found":
        guidance.append("Review the identified section to confirm the disclosure is specific, current, and complete.")
        guidance.append("Compare the language against prior filings if change detection is important.")
    elif review_result.finding == "rule_found":
        guidance.append("Cross-check the rule language against the filing text before making a compliance conclusion.")
    elif review_result.finding == "manual_review_recommended":
        guidance.append("Escalate this question for manual review because strong evidence was not clearly surfaced.")
    elif review_result.finding == "invalid_scope":
        guidance.append("Choose a company, form, and year combination from the available options.")
        guidance.append("If you want a rule-only answer, submit the query without filing scope.")

    if route_intent == "structural_section_lookup" and evidence:
        guidance.append("Use the filing section as the primary review anchor before falling back to broader semantic search.")

    return guidance[:3]


def _build_answer(
    raw_answer: str,
    review_result: ReviewResult,
    evidence: List[EvidenceItem],
) -> str:
    answer = (raw_answer or "").strip()

    if answer:
        return answer

    if review_result.finding == "section_found" and evidence:
        first = evidence[0]
        label = _build_source_label(first.company, first.form, first.year, first.section)
        return f"The relevant disclosure appears to be located in {label}."

    if review_result.finding == "manual_review_recommended":
        return "The system could not surface enough strong evidence to provide a confident answer."

    return "Relevant evidence was retrieved, but the result should be reviewed by a human before use."


def _build_invalid_scope_response(payload: QueryRequest) -> QueryResponse:
    return QueryResponse(
        status="REFUSE",
        review_result=ReviewResult(
            finding="invalid_scope",
            label="Requested filing scope is not available",
            confidence="high",
            evidence_strength="not_found",
        ),
        answer="The selected company, form, and year combination is not available in the indexed dataset.",
        plain_english_explanation=(
            "The system checks available filing metadata before retrieval, and this specific scope combination was not found."
        ),
        why_it_matters=(
            "This prevents unsupported answers by ensuring the system only runs against filings that actually exist in the indexed dataset."
        ),
        review_guidance=[
            "Choose a company, form, and year combination from the available options.",
            "If you want a broader answer, submit the query without filing scope.",
        ],
        sources=[],
        evidence_preview=[],
        technical_trace=None,
        notes=[
            "Query stopped before retrieval because the requested scope is not available."
        ],
        evidence=[],
    )


# ------------------------------------------------------------------
# Minimal, grounded inline scope extraction
# ------------------------------------------------------------------
def _extract_form_from_query(query: str) -> Optional[str]:
    q = str(query or "")
    for pattern, canonical in FORM_PATTERNS:
        if pattern.search(q):
            return canonical
    return None


def _extract_year_from_query(query: str) -> Optional[int]:
    q = str(query or "")
    m = YEAR_RE.search(q)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _extract_company_ticker_from_query(query: str) -> Optional[str]:
    """
    Conservative ticker extraction only.

    This intentionally matches only real tickers from the indexed options data.
    It does NOT guess company names like 'Microsoft' -> 'MSFT' unless the query
    explicitly contains the ticker token.
    """
    q = str(query or "").upper()

    options = get_available_options()
    tickers = sorted((c.ticker for c in options.companies), key=len, reverse=True)

    for ticker in tickers:
        if re.search(rf"\b{re.escape(ticker)}\b", q):
            return ticker

    return None


def _resolve_payload_scope(payload: QueryRequest) -> QueryRequest:
    """
    Preserve explicitly provided scope.
    Only fill missing company/form/year from the query text when the extraction
    is conservative and grounded in real indexed metadata.
    """
    if payload.company and payload.form_folder and payload.year is not None:
        return payload

    extracted_company = payload.company or _extract_company_ticker_from_query(payload.query)
    extracted_form = payload.form_folder or _extract_form_from_query(payload.query)
    extracted_year = payload.year if payload.year is not None else _extract_year_from_query(payload.query)

    return payload.model_copy(
        update={
            "company": extracted_company,
            "form_folder": extracted_form,
            "year": extracted_year,
        }
    )


def run_query_service(payload: QueryRequest) -> QueryResponse:
    resolved_payload = _resolve_payload_scope(payload)

    if not is_valid_scope(resolved_payload.company, resolved_payload.form_folder, resolved_payload.year):
        return _build_invalid_scope_response(resolved_payload)

    result = answer_query(
        query=resolved_payload.query,
        company=resolved_payload.company,
        form_folder=resolved_payload.form_folder,
        year=resolved_payload.year,
    )

    status = result.get("status", "ERROR")
    route = _normalize_route(result.get("route"))
    guard_reason = _normalize_guard_reason(result.get("guard_reason"))
    evidence = _normalize_evidence(result.get("evidence"))
    review_result = _infer_review_result(status, route, guard_reason, evidence)
    answer = _build_answer(result.get("answer", ""), review_result, evidence)

    return QueryResponse(
        status=status,
        review_result=review_result,
        answer=answer,
        plain_english_explanation=_build_plain_english_explanation(
            route=route,
            guard_reason=guard_reason,
            review_result=review_result,
            evidence=evidence,
        ),
        why_it_matters=_build_why_it_matters(
            query=resolved_payload.query,
            route=route,
            evidence=evidence,
        ),
        review_guidance=_build_review_guidance(
            review_result=review_result,
            route=route,
            evidence=evidence,
        ),
        sources=_dedupe_sources(evidence),
        evidence_preview=_build_evidence_preview(evidence),
        technical_trace=TechnicalTrace(
            route=route,
            guard_reason=guard_reason,
        ),
        notes=result.get("notes", []),
        evidence=evidence,
    )