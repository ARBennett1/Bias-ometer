#!/usr/bin/env python3
"""
prescreen_audit.py
──────────────────
Diagnostic tool for investigating the frame prescreen pipeline.

Scans a video between START and END at a configurable interval,
scores every frame with both Tesseract and the pixel-contrast heuristic,
saves all frames, and writes:
  <output_dir>/report.csv   — per-frame scores table
  <output_dir>/report.html  — visual grid with pass/fail highlights

Usage:
  python prescreen_audit.py <video> --start 0 --end 120
  python prescreen_audit.py <video> --start 45 --end 105 --interval 0.5 --output output/audit_run1
  python prescreen_audit.py <video> --end 120 --pixel-threshold 25 --min-chars 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Optional dependency checks
# ──────────────────────────────────────────────────────────────────────────────

try:
    from PIL import Image
    import numpy as np
except ImportError:
    sys.exit(
        "ERROR: Pillow and numpy are required.\n"
        "Run: pip install Pillow numpy"
    )

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    _TESSERACT_OK = True
except Exception:
    _TESSERACT_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Frame extraction  (matches screen_capture._extract_frame exactly)
# ──────────────────────────────────────────────────────────────────────────────

def extract_frame(source: str, timestamp: float, out_path: Path) -> bool:
    """
    Extract a single frame using the same ffmpeg invocation as
    screen_capture._extract_frame so results are directly comparable.
    """
    cmd = [
        "ffmpeg",
        "-ss", f"{timestamp:.3f}",   # seek before input → fast keyframe seek
        "-i", source,
        "-vframes", "1",
        "-q:v", "2",
        "-vf", "scale=1280:-1",
        "-y",
        str(out_path),
    ]
    log.debug(f"ffmpeg: {timestamp:.3f}s → {out_path.name}")
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=45)
        if not out_path.exists() or out_path.stat().st_size == 0:
            log.warning(f"  ffmpeg produced empty file at {out_path.name}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")[:200]
        log.warning(f"  ffmpeg error at {timestamp:.3f}s: {stderr}")
        return False
    except subprocess.TimeoutExpired:
        log.warning(f"  ffmpeg timed out at {timestamp:.3f}s")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_frame(
    frame_path: Path,
    bottom_fraction: float,
    pixel_threshold: float,
    min_chars: int,
) -> dict:
    """
    Run both prescreen methods on a frame and return a result dict.
    Matches the logic in screen_capture._prescreen_frame.
    """
    img = Image.open(frame_path)
    w, h = img.size
    crop = img.crop((0, int(h * (1 - bottom_fraction)), w, h))

    # Tesseract
    tess_chars: int | None = None
    tess_pass: bool | None = None
    if _TESSERACT_OK:
        try:
            text = pytesseract.image_to_string(crop, config="--psm 6").strip()
            tess_chars = len(text)
            tess_pass = tess_chars >= min_chars
        except Exception as e:
            log.warning(f"  Tesseract failed on {frame_path.name}: {e}")

    # Pixel heuristic
    gray = crop.convert("L")
    arr = np.array(gray)
    pixel_std = float(arr.std())
    pixel_pass = pixel_std > pixel_threshold

    return {
        "tess_chars": tess_chars,
        "tess_pass": tess_pass,
        "pixel_std": pixel_std,
        "pixel_pass": pixel_pass,
    }


# ──────────────────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
body { background:#111; color:#eee; font-family:monospace; margin:0; padding:16px; }
h1 { font-size:1.1em; color:#aaa; margin-bottom:8px; }
.stats { background:#1e1e1e; padding:10px 16px; border-radius:6px; margin-bottom:20px;
         font-size:0.9em; display:flex; gap:24px; flex-wrap:wrap; }
.stat { display:flex; flex-direction:column; }
.stat-label { color:#888; font-size:0.8em; }
.stat-value { font-size:1.2em; font-weight:bold; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:12px; }
.card { border-radius:6px; overflow:hidden; background:#1a1a1a; border:2px solid #444; }
.card.both-pass { border-color:#2ea043; }
.card.one-pass  { border-color:#d29922; }
.card img { width:100%; display:block; }
.card .placeholder { width:100%; height:140px; background:#2a2a2a; display:flex;
                     align-items:center; justify-content:center; color:#666;
                     font-size:0.85em; }
.card .info { padding:8px; }
.card .ts { font-weight:bold; font-size:0.95em; margin-bottom:6px; }
.badges { display:flex; flex-direction:column; gap:4px; }
.badge { display:inline-block; padding:2px 8px; border-radius:3px; font-size:0.8em; }
.badge.pass { background:#2ea043; color:#fff; }
.badge.fail { background:#c0392b; color:#fff; }
.badge.na   { background:#555;    color:#ccc; }
"""


def _badge(label: str, passed: bool | None, detail: str) -> str:
    if passed is None:
        cls = "na"
    elif passed:
        cls = "pass"
    else:
        cls = "fail"
    return f'<span class="badge {cls}">{label}: {detail}</span>'


def write_html_report(
    rows: list[dict],
    output_dir: Path,
    video_name: str,
    start: float,
    end: float,
    interval: float,
    pixel_threshold: float,
    min_chars: int,
) -> Path:
    total = len(rows)
    extract_fails = sum(1 for r in rows if not r["extraction_ok"])
    tess_passes = sum(1 for r in rows if r["tess_pass"] is True)
    pixel_passes = sum(1 for r in rows if r["pixel_pass"] is True)
    both_pass = sum(1 for r in rows if r["tess_pass"] is True and r["pixel_pass"] is True)
    neither = sum(1 for r in rows if r["extraction_ok"] and not r["tess_pass"] and not r["pixel_pass"])

    stats_html = "\n".join([
        f'<div class="stat"><span class="stat-label">Total frames</span><span class="stat-value">{total}</span></div>',
        f'<div class="stat"><span class="stat-label">Extract failures</span><span class="stat-value" style="color:{"#c0392b" if extract_fails else "#eee"}">{extract_fails}</span></div>',
        f'<div class="stat"><span class="stat-label">Tesseract passes</span><span class="stat-value" style="color:#2ea043">{tess_passes}</span></div>' if _TESSERACT_OK else "",
        f'<div class="stat"><span class="stat-label">Pixel passes</span><span class="stat-value" style="color:#2ea043">{pixel_passes}</span></div>',
        f'<div class="stat"><span class="stat-label">Both pass</span><span class="stat-value" style="color:#2ea043">{both_pass}</span></div>',
        f'<div class="stat"><span class="stat-label">Neither passes</span><span class="stat-value" style="color:{"#d29922" if neither else "#eee"}">{neither}</span></div>',
    ])

    cards_html_parts = []
    for r in rows:
        ts = r["ts"]
        fname = r["frame_filename"]
        extraction_ok = r["extraction_ok"]

        if extraction_ok:
            img_html = f'<img src="frames/{fname}" loading="lazy" alt="frame at {ts:.2f}s">'
        else:
            img_html = '<div class="placeholder">EXTRACT FAILED</div>'

        tess_badge = _badge(
            "T",
            r["tess_pass"],
            f'{r["tess_chars"]} chars' if r["tess_chars"] is not None else "N/A",
        )
        pixel_badge = _badge(
            "P",
            r["pixel_pass"],
            f'std={r["pixel_std"]:.1f}' if r["pixel_std"] is not None else "N/A",
        )

        if not extraction_ok:
            card_cls = "card"
        elif r["tess_pass"] is True and r["pixel_pass"] is True:
            card_cls = "card both-pass"
        elif r["tess_pass"] is True or r["pixel_pass"] is True:
            card_cls = "card one-pass"
        else:
            card_cls = "card"

        cards_html_parts.append(f"""
<div class="{card_cls}">
  {img_html}
  <div class="info">
    <div class="ts">{ts:.2f}s</div>
    <div class="badges">
      {tess_badge}
      {pixel_badge}
    </div>
  </div>
</div>""")

    cards_html = "\n".join(cards_html_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prescreen Audit — {video_name}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Prescreen Audit &mdash; {video_name} &mdash; {start:.1f}s to {end:.1f}s &mdash; interval={interval}s &mdash; pixel_threshold={pixel_threshold} &mdash; min_chars={min_chars}</h1>
<div class="stats">
{stats_html}
</div>
<div class="grid">
{cards_html}
</div>
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prescreen audit: exhaustively scan a video segment and visualise frame scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Local video file path")
    parser.add_argument("--start", "-s", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--end", "-e", type=float, required=True, help="End time in seconds")
    parser.add_argument("--interval", "-i", type=float, default=0.5, help="Scan interval in seconds (default: 0.5)")
    parser.add_argument("--output", "-o", default="output/prescreen_audit", help="Output directory (default: output/prescreen_audit)")
    parser.add_argument("--bottom-fraction", type=float, default=0.25, help="Bottom strip fraction for OCR/heuristic (default: 0.25)")
    parser.add_argument("--pixel-threshold", type=float, default=40.0, help="Pixel std threshold (default: 40.0)")
    parser.add_argument("--min-chars", type=int, default=8, help="Tesseract min chars to pass (default: 8)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not _TESSERACT_OK:
        log.warning("Tesseract not available — running in pixel-only mode")

    output_dir = Path(args.output)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Build timestamp list
    timestamps: list[float] = []
    ts = args.start
    while ts < args.end:
        timestamps.append(round(ts, 6))
        ts += args.interval

    video_name = Path(args.video).name
    log.info(f"Video  : {video_name}")
    log.info(f"Range  : {args.start}s – {args.end}s  ({len(timestamps)} frames at {args.interval}s interval)")
    log.info(f"Output : {output_dir}")
    if not _TESSERACT_OK:
        log.info("Tesseract: unavailable (pixel-only mode)")

    rows: list[dict] = []

    for i, ts in enumerate(timestamps):
        frame_name = f"frame_{ts:.3f}.png"
        frame_path = frames_dir / frame_name

        log.info(f"[{i+1}/{len(timestamps)}]  t={ts:.3f}s  → {frame_name}")

        ok = extract_frame(args.video, ts, frame_path)

        if not ok:
            rows.append({
                "ts": ts,
                "frame_filename": frame_name,
                "extraction_ok": False,
                "tess_chars": None,
                "tess_pass": None,
                "pixel_std": None,
                "pixel_pass": None,
            })
            continue

        try:
            scores = score_frame(frame_path, args.bottom_fraction, args.pixel_threshold, args.min_chars)
        except Exception as e:
            log.warning(f"  Scoring failed for {frame_name}: {e}")
            scores = {"tess_chars": None, "tess_pass": None, "pixel_std": None, "pixel_pass": None}

        row = {"ts": ts, "frame_filename": frame_name, "extraction_ok": True, **scores}
        rows.append(row)

        tess_info = f"T={scores['tess_chars']}chars pass={scores['tess_pass']}" if _TESSERACT_OK else "T=N/A"
        pixel_info = f"P_std={scores['pixel_std']:.1f} pass={scores['pixel_pass']}"
        log.info(f"  {tess_info}  |  {pixel_info}")

    # Write CSV
    csv_path = output_dir / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ts", "frame_filename", "extraction_ok", "tess_chars", "tess_pass", "pixel_std", "pixel_pass"],
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"CSV report written: {csv_path}")

    # Write HTML
    html_path = write_html_report(
        rows=rows,
        output_dir=output_dir,
        video_name=video_name,
        start=args.start,
        end=args.end,
        interval=args.interval,
        pixel_threshold=args.pixel_threshold,
        min_chars=args.min_chars,
    )

    # Summary
    total = len(rows)
    extract_fails = sum(1 for r in rows if not r["extraction_ok"])
    tess_passes = sum(1 for r in rows if r["tess_pass"] is True)
    pixel_passes = sum(1 for r in rows if r["pixel_pass"] is True)
    both_pass = sum(1 for r in rows if r["tess_pass"] is True and r["pixel_pass"] is True)
    neither = sum(1 for r in rows if r["extraction_ok"] and not r["tess_pass"] and not r["pixel_pass"])

    print()
    print("── Prescreen Audit Summary ────────────────────────────────────")
    print(f"  Total frames scanned : {total}")
    print(f"  Extraction failures  : {extract_fails}")
    if _TESSERACT_OK:
        print(f"  Tesseract passes     : {tess_passes}  ({100*tess_passes/max(total-extract_fails,1):.0f}%)")
    else:
        print("  Tesseract            : unavailable")
    print(f"  Pixel passes         : {pixel_passes}  ({100*pixel_passes/max(total-extract_fails,1):.0f}%)")
    print(f"  Both methods pass    : {both_pass}")
    print(f"  Neither method passes: {neither}")
    print()
    print(f"  HTML report : {html_path.resolve()}")
    print(f"  CSV data    : {csv_path.resolve()}")
    print()
    print(f"Open the report:\n  open {html_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
