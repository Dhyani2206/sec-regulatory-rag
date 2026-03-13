import json
import re
from pathlib import Path

PROCESSED_DIR = Path("processed_data")

# Real-world thresholds (still strict enough to catch broken extraction)
MIN_LEN_10K = {
    "ITEM 1A": 2500,
    "ITEM 7":  2500,
}

# 10-Q: Item 2 (MD&A) is the reliability signal.
# Item 1A may be absent legitimately.
MIN_LEN_10Q = {
    "ITEM 2":  1200,
}

# Heading bleed detectors (true bad split)
BAD_1A_BLEED_RE = re.compile(r"^\s*ITEM\s+(1B|7A?|8|9)\b", re.IGNORECASE | re.MULTILINE)


def normalize_key(k: str) -> str:
    k = k.upper().strip()
    k = k.replace(".", "")
    k = " ".join(k.split())
    return k


def coerce_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if isinstance(v.get("text"), str):
            return v["text"]
        parts = [x for x in v.values() if isinstance(x, str)]
        return "\n".join(parts)
    return ""


def qc_one(file_path: Path):
    data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))

    # wrapper or direct sections dict
    form_type = ""
    if isinstance(data, dict):
        form_type = str(data.get("form_type", "")).upper()
        sections = data.get("sections", data)
    else:
        sections = {}

    if not isinstance(sections, dict):
        sections = {}

    norm = {normalize_key(k): v for k, v in sections.items()}

    # decide kind
    p = str(file_path).lower()
    if "10-k" in p or "10k" in form_type:
        kind = "10-K"
        rules = MIN_LEN_10K
    elif "10-q" in p or "10q" in form_type:
        kind = "10-Q"
        rules = MIN_LEN_10Q
    else:
        kind = "UNKNOWN"
        rules = {}

    issues = []
    warnings = []

    # required checks
    for target, minlen in rules.items():
        if target not in norm:
            issues.append(f"Missing {target}")
            continue

        text = coerce_text(norm[target])
        if len(text) < minlen:
            issues.append(f"{target} too short: {len(text)} chars")

        # Only for 10-K Item 1A: detect true heading bleed (not mere references)
        if kind == "10-K" and target == "ITEM 1A":
            head = text[:6000]
            if BAD_1A_BLEED_RE.search(head):
                issues.append("ITEM 1A contains another ITEM heading near start (bad split)")

    # 10-Q Item 1A: warning only (often legitimately absent)
    if kind == "10-Q" and "ITEM 1A" not in norm:
        warnings.append("10-Q: ITEM 1A not present (often OK)")

    return {"file": str(file_path), "kind": kind, "issues": issues, "warnings": warnings}


def main():
    reports = []
    for fp in PROCESSED_DIR.rglob("*.json"):
        reports.append(qc_one(fp))

    bad = [r for r in reports if r["issues"]]

    print(f"Total filings checked: {len(reports)}")
    print(f"Filings with issues:   {len(bad)}")

    out = Path("qc_report.json")
    out.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()