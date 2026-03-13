import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from src.rag.config import RAGConfig

CFG = RAGConfig()

# ---------- SETTINGS ----------
YEAR = "2025"         # start with one year; later loop 2021..2025
TITLE = "17"          # CFR Title 17
OUT_DIR = CFG.regulations_processed_dir

# GovInfo bulk CFR base
BULK_BASE = "https://www.govinfo.gov/bulkdata/CFR"

# Parts we want
TARGET_PARTS = {"210", "229"}  # Reg S-X, Reg S-K

# Regex helpers
PART_RE = re.compile(r"\bPART\s+(\d+)\b", re.IGNORECASE)


@dataclass
class RuleDoc:
    year: str
    title: str
    part: str
    section: str
    citation: str
    heading: str
    text: str
    source: str


def download_zip(year: str, title: str, dest: Path) -> Path:
    """
    Downloads govinfo CFR bulk zip for given year+title.
    govinfo uses: CFR-{year}-title-{title}.zip  (note the hyphen)
    """
    dest.mkdir(parents=True, exist_ok=True)

    url = f"https://www.govinfo.gov/bulkdata/CFR/{year}/title-{title}/CFR-{year}-title-{title}.zip"
    out = dest / f"CFR-{year}-title-{title}.zip"

    if out.exists() and out.stat().st_size > 0:
        return out

    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed download: HTTP {r.status_code} for {url}")

    with out.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return out


def iter_xml_files_from_zip(zip_path: Path) -> Iterable[Tuple[str, bytes]]:
    """
    Yields (name, content_bytes) for XML files inside the zip.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.lower().endswith(".xml"):
                yield name, z.read(name)


def normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_part_and_section_from_heading(heading: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempts to infer CFR part + section from a heading string.
    Example section heading in CFR XML often includes '§ 229.105' or similar.
    """
    part = None
    section = None

    # Try to find "§ 229.105" or "§ 210.3-01"
    m = re.search(r"§\s*(\d+)\.(\d+(?:-\d+)?)", heading)
    if m:
        part = m.group(1)
        section = f"{m.group(1)}.{m.group(2)}"
        return part, section

    return None, None


def parse_cfr_xml(xml_bytes: bytes) -> List[RuleDoc]:
    """
    Parses one CFR XML file and returns rule docs for target parts.
    """
    soup = BeautifulSoup(xml_bytes, "xml")

    docs: List[RuleDoc] = []

    # CFR structure varies; we look for SECTION tags which typically contain SECTNO + SUBJECT + P
    for section_tag in soup.find_all(["SECTION", "SECT"]):
        sectno = section_tag.find(["SECTNO", "SECTNUM"])
        subject = section_tag.find(["SUBJECT", "SECTSUBJ"])

        sectno_text = normalize_text(sectno.get_text(" ")) if sectno else ""
        subject_text = normalize_text(subject.get_text(" ")) if subject else ""

        heading = " ".join([x for x in [sectno_text, subject_text] if x]).strip()
        if not heading:
            continue

        part, section = extract_part_and_section_from_heading(heading)
        if not part or part not in TARGET_PARTS or not section:
            continue

        # Extract body text
        body = normalize_text(section_tag.get_text(" "))
        if len(body) < 200:
            continue

        citation = f"{TITLE} CFR {section}"
        docs.append(
            RuleDoc(
                year=YEAR,
                title=TITLE,
                part=part,
                section=section,
                citation=citation,
                heading=heading[:300],
                text=body,
                source=f"govinfo:CFR-{YEAR}-title{TITLE}",
            )
        )

    return docs


def write_jsonl(path: Path, rows: List[RuleDoc]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "_govinfo_cache"
    zip_path = download_zip(YEAR, TITLE, cache_dir)
    print(f"[govinfo] Downloaded: {zip_path} ({zip_path.stat().st_size} bytes)")

    all_docs: List[RuleDoc] = []
    for name, xml_bytes in iter_xml_files_from_zip(zip_path):
        docs = parse_cfr_xml(xml_bytes)
        if docs:
            all_docs.extend(docs)

    regsk = [d for d in all_docs if d.part == "229"]
    regsx = [d for d in all_docs if d.part == "210"]

    out_sk = OUT_DIR / f"regsk_229_{YEAR}.jsonl"
    out_sx = OUT_DIR / f"regsx_210_{YEAR}.jsonl"

    write_jsonl(out_sk, regsk)
    write_jsonl(out_sx, regsx)

    manifest = {
        "source": f"govinfo:CFR-{YEAR}-title{TITLE}",
        "year": YEAR,
        "title": TITLE,
        "total_rule_sections": len(all_docs),
        "regsk_229": len(regsk),
        "regsx_210": len(regsx),
        "outputs": {"regsk": str(out_sk), "regsx": str(out_sx)},
    }
    (OUT_DIR / f"govinfo_manifest_{YEAR}_title{TITLE}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[govinfo] Extracted Reg S-K (Part 229): {len(regsk)} sections -> {out_sk}")
    print(f"[govinfo] Extracted Reg S-X (Part 210): {len(regsx)} sections -> {out_sx}")
    print("[govinfo] DONE.")


if __name__ == "__main__":
    main()