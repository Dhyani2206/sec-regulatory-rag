from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

from bs4 import BeautifulSoup


def _clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


HEADING_START_RE = re.compile(
    r"^\s*"
    r"(?:(?:PART)\s+([IVX]+)\b\s*[\.\:\-\u2013\u2014]?\s*)?"
    r"(?:(?:OTHER\s+INFORMATION|FINANCIAL\s+INFORMATION|"
    r"MANAGEMENT\S*|RISK\s+FACTORS|LEGAL\s+PROCEEDINGS|"
    r"CONTROLS\s+AND\s+PROCEDURES|UNRESOLVED\s+STAFF\s+COMMENTS)\s*)?"
    r"ITEM\s+(\d{1,2})([A-Z]?)\b"
    r"\s*[\.\:\-\u2013\u2014]?\s*(.*)$",
    flags=re.IGNORECASE
)

TOC_MARK_RE = re.compile(r"\bTABLE\s+OF\s+CONTENTS\b", re.IGNORECASE)

# Strong cue phrases
MDA_RE = re.compile(r"MANAGEMENT[’'S]*\s+DISCUSSION\s+AND\s+ANALYSIS", re.IGNORECASE)
RISK_FACTORS_RE = re.compile(r"RISK\s+FACTORS", re.IGNORECASE)


def _normalize_item_key(num: str, letter: str) -> str:
    return f"ITEM {int(num)}{(letter or '').upper()}"


def _is_heading_line(line: str) -> Tuple[bool, str, str]:
    s = re.sub(r"\s+", " ", line).strip()
    if len(s) < 4 or len(s) > 220:
        return (False, "", "")
    m = HEADING_START_RE.match(s)
    if not m:
        return (False, "", "")
    if s.endswith(".") and len(s) > 140:
        return (False, "", "")
    item_key = _normalize_item_key(m.group(2), m.group(3))
    title = _clean(m.group(4) or "")
    return (True, item_key, title)


def _html_to_lines(soup: BeautifulSoup) -> List[str]:
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    raw = soup.get_text("\n").replace("\xa0", " ")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return [_clean(x) for x in raw.split("\n") if _clean(x)]


def _strip_toc_lines(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    toc_idx = None
    for i, ln in enumerate(lines):
        if TOC_MARK_RE.search(ln):
            toc_idx = i
            break
    start = toc_idx if toc_idx is not None else 0
    for j in range(start, len(lines)):
        ok, item_key, _ = _is_heading_line(lines[j])
        if ok and item_key == "ITEM 1":
            return lines[j:]
    return lines


def _score_candidate(item_key: str, title: str, text: str) -> int:
    """
    Choose real body section vs TOC stub.
    """
    head = (text or "")[:8000]
    head_u = head.upper()
    title_u = (title or "").upper()

    score = min(len(text), 250000)

    # Penalize TOC
    if "TABLE OF CONTENTS" in head_u:
        score -= 200000

    if item_key == "ITEM 1A":
        if "RISK" in title_u:
            score += 25000
        if RISK_FACTORS_RE.search(head):
            score += 40000

    if item_key == "ITEM 7":
        # This is the big fix for BAC/PFE/Visa TOC stubs:
        # prefer sections that actually contain MD&A cue text
        if MDA_RE.search(head):
            score += 60000
        if "MANAGEMENT" in title_u or "DISCUSSION" in title_u or "ANALYSIS" in title_u:
            score += 15000

    if item_key == "ITEM 2":
        # 10-Q MD&A
        if MDA_RE.search(head):
            score += 40000

    return score


def _slice_section(lines: List[str], start_i: int, end_i: int) -> str:
    body = []
    for j in range(start_i, end_i):
        ok2, _, _ = _is_heading_line(lines[j])
        if ok2:
            break
        body.append(lines[j])
    return _clean("\n".join(body))


def _keyword_fallback(lines: List[str], start_pat: re.Pattern, stop_item_pat: re.Pattern) -> str:
    """
    Find a section that starts at a keyword heading (like MD&A) and stops at the next ITEM heading.
    """
    for i, ln in enumerate(lines):
        if start_pat.search(ln):
            body = []
            for j in range(i + 1, len(lines)):
                if stop_item_pat.match(re.sub(r"\s+", " ", lines[j]).strip()):
                    break
                body.append(lines[j])
            text = _clean("\n".join(body))
            if len(text) >= 800:
                return text
    return ""


def extract_sections_from_file(path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    lines = _strip_toc_lines(_html_to_lines(soup))

    # headings: (line_index, item_key, title)
    headings: List[Tuple[int, str, str]] = []
    for i, ln in enumerate(lines):
        ok, item_key, title = _is_heading_line(ln)
        if ok:
            headings.append((i, item_key, title))

    best: Dict[str, Dict[str, str]] = {}
    best_score: Dict[str, int] = {}

    # slice by headings
    for h_idx, (i, item_key, title) in enumerate(headings):
        start = i + 1
        end = headings[h_idx + 1][0] if h_idx + 1 < len(headings) else len(lines)
        text = _slice_section(lines, start, end)
        if len(text) < 500:
            continue

        sc = _score_candidate(item_key, title, text)
        if item_key not in best_score or sc > best_score[item_key]:
            best[item_key] = {"title": title, "text": text}
            best_score[item_key] = sc

    # -----------------------
    # FALLBACKS (real-world)
    # -----------------------

    # 10-K fallback for ITEM 7 if missing or obviously stubby
    # (fixes JPM/XOM missing ITEM 7 and BAC/PFE/Visa stubby ITEM 7 cases) :contentReference[oaicite:8]{index=8}
    if "ITEM 7" not in best or len(best["ITEM 7"]["text"]) < 2000:
        stop_item_line = re.compile(r"^\s*(?:PART\s+[IVX]+\s*)?ITEM\s+\d{1,2}[A-Z]?\b", re.IGNORECASE)
        text = _keyword_fallback(lines, MDA_RE, stop_item_line)
        if text:
            best["ITEM 7"] = {"title": "Management’s Discussion and Analysis", "text": text}

    # 10-Q fallback for ITEM 2 if too short (JPM/PFE show very short Item 2) :contentReference[oaicite:9]{index=9}
    if "ITEM 2" in best and len(best["ITEM 2"]["text"]) < 1200:
        stop_item_line = re.compile(r"^\s*(?:PART\s+[IVX]+\s*)?ITEM\s+\d{1,2}[A-Z]?\b", re.IGNORECASE)
        text = _keyword_fallback(lines, MDA_RE, stop_item_line)
        if text:
            best["ITEM 2"] = {"title": "Management’s Discussion and Analysis", "text": text}

    # Visa-specific (or generic) extra fallback when Item 7 is still short:
    if "ITEM 7" in best and len(best["ITEM 7"]["text"]) < 2500:
        stop_item_line = re.compile(r"^\s*(?:PART\s+[IVX]+\s*)?ITEM\s+\d{1,2}[A-Z]?\b", re.IGNORECASE)

        # Alternate MD&A cue phrases often used in some filings
        ALT_MDA_RE = re.compile(
            r"(?:MANAGEMENT[’'S]*\s+DISCUSSION\s+AND\s+ANALYSIS|MD&A|RESULTS\s+OF\s+OPERATIONS|FINANCIAL\s+CONDITION)",
            re.IGNORECASE
        )

        text = _keyword_fallback(lines, ALT_MDA_RE, stop_item_line)
        if text and len(text) > len(best["ITEM 7"]["text"]):
            best["ITEM 7"] = {"title": "Management’s Discussion and Analysis", "text": text}

    return best