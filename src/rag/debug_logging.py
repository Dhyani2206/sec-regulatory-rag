from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL_DEBUG_LOG = LOG_DIR / "retrieval_debug.jsonl"


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _safe_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe_jsonable(v) for v in value]
        return str(value)


def _extract_chunk_ids(items: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not items:
        return []
    ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        chunk = item.get("chunk") if isinstance(item.get("chunk"), dict) else item
        chunk_id = chunk.get("chunk_id")
        if chunk_id is not None:
            ids.append(str(chunk_id))
    return ids


def log_retrieval_event(
    *,
    query: str,
    company: Optional[str],
    form_folder: Optional[str],
    year: Optional[int],
    route: Any,
    stage: str,
    structural_hits: Optional[List[Dict[str, Any]]] = None,
    semantic_hits: Optional[List[Dict[str, Any]]] = None,
    used_semantic_fallback: bool = False,
    guard_reason: Optional[str] = None,
    status: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> None:
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "scope": {
            "company": company,
            "form_folder": form_folder,
            "year": year,
        },
        "route": _safe_jsonable(route),
        "stage": stage,
        "used_semantic_fallback": used_semantic_fallback,
        "structural_hit_count": len(structural_hits or []),
        "structural_chunk_ids": _extract_chunk_ids(structural_hits),
        "semantic_hit_count": len(semantic_hits or []),
        "semantic_chunk_ids": _extract_chunk_ids(semantic_hits),
        "guard_reason": guard_reason,
        "status": status,
        "notes": notes or [],
    }

    with RETRIEVAL_DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")