import json
from pathlib import Path

from section_extractor import extract_sections_from_file

BASE_DIR = Path("companies_data")
OUTPUT_DIR = Path("processed_data")


def process_all_filings():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for company_dir in BASE_DIR.iterdir():
        if not company_dir.is_dir():
            continue

        company = company_dir.name

        for form_type in ["10-k", "10-q"]:
            form_path = company_dir / form_type
            if not form_path.exists():
                continue

            for html_file in list(form_path.glob("*.html")) + list(form_path.glob("*.htm")):
                print(f"Processing {company} {form_type} {html_file.name}")

                sections = extract_sections_from_file(html_file)

                out_dir = OUTPUT_DIR / company / form_type
                out_dir.mkdir(parents=True, exist_ok=True)

                out_file = out_dir / (html_file.stem + ".json")

                payload = {
                    "company": company,
                    "form_type": form_type.upper().replace("-", ""),
                    "source_file": str(html_file),
                    "sections": sections
                }

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

    print("Extraction complete.")


if __name__ == "__main__":
    process_all_filings()