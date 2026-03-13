import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.rag.config import RAGConfig

CFG = RAGConfig()

# Primary sources (eCFR)

REG_SK_TOC = "https://www.ecfr.gov/current/title-17/chapter-II/part-229?toc=1"
REG_SX_TOC = "https://www.ecfr.gov/current/title-17/chapter-II/part-210?toc=1"

# guardrails
USER_AGENT = "sec-regulatory-poc/1.0 (compliance-rag; contact: internal)"
REQ_TIMEOUT = 30
SLEEP_SECONDS = 0.7         # throttle to avoid triggering bot detection
MAX_RETRIES = 4
BACKOFF = 2.0               # exponential backoff base
CACHE_DIR = CFG.regulations_processed_dir / "_cache_html"

# Seed set (cost-effective MVP)
SEED_SK = {
    "229.101", "229.103", "229.105", "229.106",
    "229.303", "229.307", "229.308", "229.401",
    "229.404", "229.405", "229.406", "229.407",
    "229.408", "229.601",
}
SEED_SX = {
    "210.1-01", "210.3-01", "210.3-02", "210.3-03",
    "210.3-04", "210.3-05", "210.3-10", "210.3-11",
    "210.4-01", "210.4-08", "210.5-02", "210.8-01",
}

SEC_CIT_RE = re.compile(r"§\s*(\d+\.\d+(?:-\d+)?)", re.IGNORECASE)  # matches 229.105 or 210.3-01


@dataclass
class RuleDoc:
    part: str              # "229" or "210"
    section: str           # "229.105" or "210.3-01"
    citation: str          # "17 CFR 229.105"
    title: str             # heading text
    text: str              # clean body text
    source_url: str
    retrieved_at_utc: str


def _utc_now() -> str:
    # keep it simple, ISO-like string (no timezone libs)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cache_path_for_url(url: str) -> Path:
    # stable filename
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")
    return CACHE_DIR / f"{safe}.html"


def fetch_html(url: str, use_cache: bool = True) -> str:
    _safe_mkdir(CACHE_DIR)
    cache_fp = _cache_path_for_url(url)

    if use_cache and cache_fp.exists() and cache_fp.stat().st_size > 0:
        return cache_fp.read_text(encoding="utf-8", errors="ignore")

    headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=REQ_TIMEOUT)
            if r.status_code == 200 and "Request Access" not in r.text:
                cache_fp.write_text(r.text, encoding="utf-8", errors="ignore")
                time.sleep(SLEEP_SECONDS)
                return r.text

            last_err = RuntimeError(f"HTTP {r.status_code} or blocked page for {url}")
        except Exception as e:
            last_err = e

        time.sleep(BACKOFF ** attempt)

    raise RuntimeError(f"Failed to fetch {url}. Last error: {last_err}")


def parse_toc_for_section_links(toc_html: str, base_url: str) -> List[str]:
    """
    Extract section page links (e.g., .../section-229.105 or .../section-210.3-01).
    """
    soup = BeautifulSoup(toc_html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/section-" in href:
            links.append(urljoin(base_url, href))
    # dedupe preserve order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def extract_rule_from_section_page(section_url: str, part: str, html: str) -> Optional[RuleDoc]:
    soup = BeautifulSoup(html, "lxml")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    # Find the first "§ 229.xxx" / "§ 210.x-xx" occurrence
    m = SEC_CIT_RE.search(text)
    if not m:
        return None

    section = m.group(1)
    citation = f"17 CFR {section}"

    # Try to capture title: usually appears right after citation in page text
    # We'll find a line that contains the citation and use following text as title candidate.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    title = ""
    for i, ln in enumerate(lines[:80]):  # look near top
        if section in ln:
            # Example: "§ 229.105 (Item 105) Risk factors."
            title = ln
            # If title line too short, fallback to next line
            if len(title) < 10 and i + 1 < len(lines):
                title = lines[i + 1]
            break

    # Body text: keep full cleaned text (simple MVP). Later narrow to main content region.
    body = text

    return RuleDoc(
        part=part,
        section=section,
        citation=citation,
        title=title[:300],
        text=body,
        source_url=section_url,
        retrieved_at_utc=_utc_now(),
    )


def ingest_part(toc_url: str, part: str, seed_sections: Optional[set] = None, use_cache: bool = True) -> Tuple[List[RuleDoc], Dict]:
    """
    Returns: (rules, manifest)
    """
    toc_html = fetch_html(toc_url, use_cache=use_cache)
    section_urls = parse_toc_for_section_links(toc_html, base_url=toc_url)

    rules: List[RuleDoc] = []
    issues: List[Dict] = []

    for u in section_urls:
        # quick seed filter by URL containing section id
        if seed_sections:
            # section page URL contains 'section-229.105' etc.
            if not any(f"section-{s}" in u for s in seed_sections):
                continue

        try:
            html = fetch_html(u, use_cache=use_cache)
            rule = extract_rule_from_section_page(u, part=part, html=html)
            if rule is None:
                issues.append({"url": u, "error": "Could not parse section/citation"})
                continue
            rules.append(rule)
        except Exception as e:
            issues.append({"url": u, "error": str(e)})

    manifest = {
        "part": part,
        "toc_url": toc_url,
        "retrieved_at_utc": _utc_now(),
        "requested_mode": "seed" if seed_sections else "all",
        "total_section_urls_seen": len(section_urls),
        "rules_extracted": len(rules),
        "issues": issues[:50],  # cap in manifest
        "issues_count": len(issues),
    }
    return rules, manifest


def write_jsonl(out_path: Path, rules: List[RuleDoc]) -> None:
    _safe_mkdir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def write_manifest(out_path: Path, manifest: Dict) -> None:
    _safe_mkdir(out_path.parent)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(mode: str = "seed", use_cache: bool = True) -> None:
    _safe_mkdir(CFG.regulations_processed_dir)

    if mode not in {"seed", "all"}:
        raise ValueError("mode must be 'seed' or 'all'")

    seed_sk = SEED_SK if mode == "seed" else None
    seed_sx = SEED_SX if mode == "seed" else None

    print(f"[RULES] Ingesting Reg S-K (Part 229) mode={mode} ...")
    sk_rules, sk_manifest = ingest_part(REG_SK_TOC, part="229", seed_sections=seed_sk, use_cache=use_cache)
    write_jsonl(CFG.regulations_processed_dir / "regsk_229.jsonl", sk_rules)
    write_manifest(CFG.regulations_processed_dir / "regsk_229_manifest.json", sk_manifest)
    print(f"[RULES] Part 229 extracted: {len(sk_rules)} rules, issues: {sk_manifest['issues_count']}")

    print(f"[RULES] Ingesting Reg S-X (Part 210) mode={mode} ...")
    sx_rules, sx_manifest = ingest_part(REG_SX_TOC, part="210", seed_sections=seed_sx, use_cache=use_cache)
    write_jsonl(CFG.regulations_processed_dir / "regsx_210.jsonl", sx_rules)
    write_manifest(CFG.regulations_processed_dir / "regsx_210_manifest.json", sx_manifest)
    print(f"[RULES] Part 210 extracted: {len(sx_rules)} rules, issues: {sx_manifest['issues_count']}")

    total = len(sk_rules) + len(sx_rules)
    print(f"[RULES] DONE. Total rules saved: {total}")
    print(f"[RULES] Output dir: {CFG.regulations_processed_dir}")


if __name__ == "__main__":
    # modes:
    #   seed = fast, cost-effective MVP
    #   all  = full parts (may trigger anti-bot on eCFR; use cautiously)
    main(mode="seed", use_cache=True)