# count_filings.py
from pathlib import Path
from collections import defaultdict

RAW = Path("companies_data")
PROC = Path("processed_data")

def count_raw():
    counts = defaultdict(lambda: {"10-k":0, "10-q":0})
    for cdir in RAW.iterdir():
        if not cdir.is_dir(): 
            continue
        for ft in ["10-k","10-q"]:
            p = cdir/ft
            if not p.exists(): 
                continue
            files = list(p.glob("*.html")) + list(p.glob("*.htm"))
            counts[cdir.name][ft] = len(files)
    return counts

def count_processed():
    counts = defaultdict(lambda: {"10-k":0, "10-q":0})
    for fp in PROC.rglob("*.json"):
        s = str(fp).lower()
        if "\\10-k\\" in s or "/10-k/" in s:
            counts[fp.parts[-3]]["10-k"] += 1
        elif "\\10-q\\" in s or "/10-q/" in s:
            counts[fp.parts[-3]]["10-q"] += 1
    return counts

print("RAW counts:")
for k,v in sorted(count_raw().items()):
    print(k, v)

print("\nPROCESSED counts:")
for k,v in sorted(count_processed().items()):
    print(k, v)