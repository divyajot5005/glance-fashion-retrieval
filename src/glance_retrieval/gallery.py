from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image


def _thumbnail_data(path: str, max_size: tuple[int, int] = (420, 300)) -> str | None:
    try:
        with Image.open(Path(path).expanduser()) as image:
            image = image.convert("RGB")
            image.thumbnail(max_size)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except OSError:
        return None


def write_gallery(query: str, results, output: Path, variant: str = "proposed") -> None:
    cards = []
    for result in results:
        image_data = _thumbnail_data(result.path)
        visual = (
            f'<img src="{image_data}" alt="Result {result.rank}">'
            if image_data
            else '<div class="missing">Image unavailable</div>'
        )
        cards.append(
            f"""
            <article class="card">
              {visual}
              <div class="body">
                <div class="rank">#{result.rank} &nbsp; score {result.score:.3f}</div>
                <div class="id">{html.escape(result.image_id)}</div>
                <dl>
                  <dt>Fashion global</dt><dd>{result.fashion_global:.3f}</dd>
                  <dt>Atomic</dt><dd>{result.fashion_atomic:.3f}</dd>
                  <dt>Context</dt><dd>{result.context:.3f}</dd>
                  <dt>Metadata</dt><dd>{result.metadata_bonus:.3f}</dd>
                </dl>
              </div>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Glance retrieval results</title>
<style>
body{{margin:0;background:#f4f7fa;color:#22313f;font:15px/1.45 system-ui,sans-serif}}
header{{background:#17324d;color:white;padding:34px max(5vw,24px)}}
header p{{margin:7px 0 0;color:#cfe4ef}}main{{padding:30px max(5vw,24px)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px}}
.card{{background:white;border:1px solid #d6e1ea;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px #17324d12}}
.card img,.missing{{width:100%;height:230px;object-fit:contain;background:#eef3f6}}
.missing{{display:grid;place-items:center;color:#66788a}}.body{{padding:16px}}.rank{{font-weight:700;color:#168c8c}}
.id{{margin:4px 0 12px;font-family:ui-monospace,monospace;overflow-wrap:anywhere}}
dl{{display:grid;grid-template-columns:1fr auto;gap:4px 12px;margin:0}}dt{{color:#66788a}}dd{{margin:0;font-variant-numeric:tabular-nums}}
</style></head><body><header><h1>Fashion &amp; context retrieval</h1>
<p>{html.escape(query)} &nbsp; | &nbsp; variant: {html.escape(variant)}</p></header>
<main><div class="grid">{''.join(cards)}</div></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
