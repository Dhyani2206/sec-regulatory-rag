"""
Human-readable application documentation.

Serves ``docs/application-guide.md`` as HTML. The app registers **two** URLs:

- ``GET /documentation`` — primary bookmark URL
- ``GET /api/v1/documentation`` — same content (works behind gateways that only expose ``/api/v1``)
"""

from __future__ import annotations

from pathlib import Path

import markdown
from fastapi import HTTPException
from fastapi.responses import HTMLResponse

_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  line-height: 1.6;
  max-width: 52rem;
  margin: 0 auto;
  padding: 1.5rem 1.25rem 3rem;
  color: #1a1a1a;
  background: #fafafa;
}
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e8; background: #121212; }
}
h1 { font-size: 1.75rem; border-bottom: 1px solid #ccc; padding-bottom: 0.35rem; }
h2 { font-size: 1.35rem; margin-top: 2rem; }
h3 { font-size: 1.1rem; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }
pre {
  background: #f0f0f0;
  padding: 1rem;
  overflow-x: auto;
  border-radius: 6px;
}
@media (prefers-color-scheme: dark) {
  pre { background: #1e1e1e; }
}
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95rem; }
th, td { border: 1px solid #888; padding: 0.5rem 0.65rem; text-align: left; }
th { background: #e8e8e8; }
@media (prefers-color-scheme: dark) {
  th { background: #2a2a2a; }
}
a { color: #0b57d0; }
header { margin-bottom: 1.5rem; font-size: 0.9rem; color: #555; }
@media (prefers-color-scheme: dark) {
  header { color: #aaa; }
}
"""


def _resolve_guide_path() -> Path:
    """
    Locate ``docs/application-guide.md`` by walking ancestors of this file.

    Avoids off-by-one ``parents[N]`` mistakes across refactors and works if the
    endpoint file moves within ``app/``.
    """
    here = Path(__file__).resolve()
    for d in [here.parent, *here.parents]:
        candidate = d / "docs" / "application-guide.md"
        if candidate.is_file():
            return candidate
    # Sensible default for error messages (repo root for typical layout)
    return here.parents[4] / "docs" / "application-guide.md"


def get_documentation_response() -> HTMLResponse:
    """
    Render ``docs/application-guide.md`` as HTML.

    Raises HTTPException 503 if the Markdown source is missing.
    """
    guide = _resolve_guide_path()
    if not guide.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Documentation source missing: {guide}",
        )

    md_text = guide.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        md_text,
        extensions=[
            "markdown.extensions.fenced_code",
            "markdown.extensions.tables",
            "markdown.extensions.nl2br",
        ],
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>SEC Regulatory RAG — Application guide</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    API reference: <a href="/docs">OpenAPI (Swagger)</a>
    · <a href="/redoc">ReDoc</a>
    · Source: <code>docs/application-guide.md</code>
  </header>
  <article>
  {body_html}
  </article>
</body>
</html>"""

    return HTMLResponse(content=html, status_code=200)
