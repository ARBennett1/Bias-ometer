#!/usr/bin/env python3
"""
caption_tuner.py  —  Bias-ometer Caption Configuration Tool
===========================================================
Interactive browser-based tool for configuring, tuning, and testing
the caption region settings used by the OCR pipeline.

Run from the Bias-ometer project root:
    python caption_tuner.py [--port 7654]

Then open http://localhost:7654 in your browser.

Dependencies: ffmpeg, ffprobe, pytesseract, Pillow, numpy, fastapi, uvicorn
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import logging
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Project imports (run from project root) ───────────────────────────────────
try:
    from caption_ocr import (
        CaptureMode,
        crop_region,
        pre_screen_passes,
        ocr_crop,
        extract_caption_result,
        TESSERACT_AVAILABLE,
    )
    from sources import CaptionRegion, PreScreen, SourceConfig
    OCR_AVAILABLE = TESSERACT_AVAILABLE
except ImportError as e:
    print(f"[caption_tuner] WARNING: Could not import project modules: {e}")
    print("  Run this tool from the Bias-ometer project root directory.")
    OCR_AVAILABLE = False

# spaCy must be imported with the project root temporarily removed from sys.path
# to prevent the project's catalogue.py shadowing the `catalogue` pip package
# that spaCy's dependency chain (srsly → confection → thinc) depends on.
_nlp = None
try:
    _project_root = str(Path(__file__).parent.resolve())
    _saved_path = [p for p in sys.path if Path(p).resolve() == Path(_project_root)]
    for _p in _saved_path:
        sys.path.remove(_p)
    try:
        import spacy as _spacy
        _nlp = _spacy.load("en_core_web_sm")
    finally:
        sys.path[:0] = _saved_path  # restore at front so project modules still work
except (ImportError, OSError):
    pass

log = logging.getLogger("caption_tuner")

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
_video_file: Optional[Path] = None
_video_duration: Optional[float] = None
_scan_jobs: dict[str, dict] = {}          # scan_id → {status, current, total, results, error}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()}")
    return float(r.stdout.strip())


def _extract_frame_b64(path: Path, ts: float) -> tuple[str, int, int]:
    """Extract one frame at `ts` seconds. Returns (base64_png, width, height)."""
    ts = max(0.0, ts)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        out_path = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", str(path),
            "-vframes", "1",
            "-f", "image2",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[:200]}")
        from PIL import Image
        img = Image.open(out_path)
        w, h = img.size
        data = out_path.read_bytes()
        b64 = base64.b64encode(data).decode()
        return b64, w, h
    finally:
        out_path.unlink(missing_ok=True)


def _frame_bgr_from_b64(b64: str) -> np.ndarray:
    from PIL import Image
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil)[:, :, ::-1]


def _crop_to_b64(crop_bgr: np.ndarray) -> str:
    from PIL import Image
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    rgb = crop_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_source_config(body: dict) -> SourceConfig:
    """Construct a SourceConfig from request body fields."""
    region = CaptionRegion(
        x=float(body["x"]),
        y=float(body["y"]),
        w=float(body["w"]),
        h=float(body["h"]),
    )
    use_prescreen = body.get("use_prescreen", True)
    pre_screen = None
    if use_prescreen:
        pre_screen = PreScreen(
            bg_colour=(int(body["bg_r"]), int(body["bg_g"]), int(body["bg_b"])),
            tolerance=int(body.get("tolerance", 30)),
        )
    return SourceConfig(
        source_id=str(body.get("source_id", "custom")),
        caption_region=region,
        name_line_index=int(body.get("name_line_index", 0)),
        pre_screen=pre_screen,
        notes=str(body.get("notes", "")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Caption Tuner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load video ────────────────────────────────────────────────────────────────
class LoadRequest(BaseModel):
    path: str

@app.post("/load")
def load_video(req: LoadRequest):
    global _video_file, _video_duration
    p = Path(req.path.strip()).expanduser()
    if not p.exists():
        raise HTTPException(400, f"File not found: {p}")
    if not p.is_file():
        raise HTTPException(400, f"Not a file: {p}")
    try:
        dur = _get_duration(p)
    except Exception as exc:
        raise HTTPException(400, f"Could not read duration: {exc}")
    _video_file = p
    _video_duration = dur
    return {"ok": True, "duration": dur, "name": p.name}


# ── Extract frame ─────────────────────────────────────────────────────────────
@app.get("/frame")
def get_frame(ts: float = 0.0):
    if _video_file is None:
        raise HTTPException(400, "No video loaded. POST /load first.")
    try:
        b64, w, h = _extract_frame_b64(_video_file, ts)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"frame_b64": b64, "width": w, "height": h, "ts": ts}


# ── Test OCR on a frame ───────────────────────────────────────────────────────
class OcrRequest(BaseModel):
    ts: float
    x: float
    y: float
    w: float
    h: float
    bg_r: int = 255
    bg_g: int = 255
    bg_b: int = 255
    tolerance: int = 30
    use_prescreen: bool = True
    name_line_index: int = 0
    capture_mode: str = "all_captions"
    source_id: str = "custom"
    notes: str = ""

@app.post("/ocr")
def test_ocr(req: OcrRequest):
    if _video_file is None:
        raise HTTPException(400, "No video loaded.")
    if not OCR_AVAILABLE:
        raise HTTPException(503, "Tesseract not available. Install: brew install tesseract && pip install pytesseract")
    try:
        b64, fw, fh = _extract_frame_b64(_video_file, req.ts)
        frame_bgr = _frame_bgr_from_b64(b64)
    except Exception as exc:
        raise HTTPException(500, f"Frame extraction failed: {exc}")

    cfg = _build_source_config(req.model_dump())
    crop = crop_region(frame_bgr, cfg.caption_region)
    crop_b64 = _crop_to_b64(crop)

    prescreen_passed = pre_screen_passes(crop, cfg.pre_screen)

    ocr_lines: list[str] = []
    result: Optional[dict] = None
    if prescreen_passed and crop.size > 0:
        try:
            ocr_lines = ocr_crop(crop)
            if ocr_lines:
                result = extract_caption_result(
                    ocr_lines,
                    cfg.name_line_index,
                    CaptureMode(req.capture_mode),
                    _nlp,
                )
        except Exception as exc:
            log.warning(f"OCR error: {exc}")

    # Sample actual pixel at region origin for colour feedback
    h_frame, w_frame = frame_bgr.shape[:2]
    px = int(req.x * w_frame)
    py = int(req.y * h_frame)
    px = min(px + 10, w_frame - 1)
    py = min(py + 10, h_frame - 1)
    sample_bgr = frame_bgr[py, px].tolist()
    sample_rgb = [sample_bgr[2], sample_bgr[1], sample_bgr[0]]

    return {
        "prescreen_passed": prescreen_passed,
        "prescreen_enabled": req.use_prescreen,
        "ocr_lines": ocr_lines,
        "name": result.get("name") if result else None,
        "raw_lines": result.get("raw_lines") if result else ocr_lines,
        "crop_b64": crop_b64,
        "sample_rgb": sample_rgb,
        "frame_b64": b64,
    }


# ── Scan ──────────────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    x: float
    y: float
    w: float
    h: float
    bg_r: int = 255
    bg_g: int = 255
    bg_b: int = 255
    tolerance: int = 30
    use_prescreen: bool = True
    name_line_index: int = 0
    capture_mode: str = "all_captions"
    source_id: str = "custom"
    notes: str = ""
    scan_start: float = 0.0
    scan_end: Optional[float] = None
    interval: float = 2.0


def _run_scan(scan_id: str, video_path: Path, cfg: SourceConfig,
              capture_mode: str, timestamps: list[float]) -> None:
    job = _scan_jobs[scan_id]
    job["total"] = len(timestamps)
    results = []

    # Deduplication state
    last_text: Optional[str] = None
    consecutive_misses: int = 0
    unique_results: list = []

    def _normalise(lines: list[str]) -> str:
        return " ".join(lines).lower().strip()

    for i, ts in enumerate(timestamps):
        if job.get("cancelled"):
            break
        job["current"] = i
        try:
            b64, fw, fh = _extract_frame_b64(video_path, ts)
            frame_bgr = _frame_bgr_from_b64(b64)
            crop = crop_region(frame_bgr, cfg.caption_region)
            crop_b64 = _crop_to_b64(crop)
            prescreen_passed = pre_screen_passes(crop, cfg.pre_screen)
            ocr_lines: list[str] = []
            name = None
            if OCR_AVAILABLE and prescreen_passed and crop.size > 0:
                try:
                    ocr_lines = ocr_crop(crop)
                    if ocr_lines:
                        r = extract_caption_result(
                            ocr_lines,
                            cfg.name_line_index,
                            CaptureMode(capture_mode),
                            _nlp,
                        )
                        if r:
                            name = r.get("name")
                except Exception:
                    pass

            if ocr_lines:
                norm = _normalise(ocr_lines)
                if last_text is not None:
                    ratio = difflib.SequenceMatcher(None, norm, last_text).ratio()
                    if ratio >= 0.90:
                        is_duplicate = True
                        consecutive_misses = 0
                    else:
                        is_duplicate = False
                        last_text = norm
                        consecutive_misses = 0
                else:
                    is_duplicate = False
                    last_text = norm
                    consecutive_misses = 0
            else:
                is_duplicate = False
                consecutive_misses += 1
                if consecutive_misses >= 2:
                    last_text = None

            entry = {
                "ts": ts,
                "frame_b64": b64,
                "crop_b64": crop_b64,
                "prescreen_passed": prescreen_passed,
                "ocr_lines": ocr_lines,
                "name": name,
                "is_duplicate": is_duplicate,
            }
            results.append(entry)
            if not is_duplicate and ocr_lines:
                unique_results.append(entry)
        except Exception as exc:
            results.append({
                "ts": ts,
                "error": str(exc),
                "prescreen_passed": False,
                "ocr_lines": [],
                "name": None,
                "frame_b64": "",
                "crop_b64": "",
                "is_duplicate": False,
            })
        job["results"] = results  # live update
        job["unique_captions"] = unique_results
    job["current"] = len(timestamps)
    job["status"] = "done"


@app.post("/scan/start")
def scan_start(req: ScanRequest):
    if _video_file is None:
        raise HTTPException(400, "No video loaded.")

    scan_end = req.scan_end if req.scan_end is not None else _video_duration
    if scan_end is None:
        raise HTTPException(400, "Unknown video duration — load the video first.")

    interval = max(0.5, req.interval)
    ts = req.scan_start
    timestamps = []
    while ts <= scan_end:
        timestamps.append(round(ts, 3))
        ts += interval
    if len(timestamps) > 600:
        raise HTTPException(400, f"Scan would produce {len(timestamps)} frames — reduce range or increase interval (max 600).")

    cfg = _build_source_config(req.model_dump())
    scan_id = str(uuid.uuid4())[:8]
    _scan_jobs[scan_id] = {
        "status": "running",
        "current": 0,
        "total": len(timestamps),
        "results": [],
        "unique_captions": [],
        "error": None,
        "cancelled": False,
    }
    t = threading.Thread(
        target=_run_scan,
        args=(scan_id, _video_file, cfg, req.capture_mode, timestamps),
        daemon=True,
    )
    t.start()
    return {"scan_id": scan_id, "total": len(timestamps)}


@app.get("/scan/poll")
def scan_poll(scan_id: str):
    job = _scan_jobs.get(scan_id)
    if job is None:
        raise HTTPException(404, "Unknown scan_id")
    return {
        "status": job["status"],
        "current": job["current"],
        "total": job["total"],
        "results": job["results"],
        "unique_captions": job.get("unique_captions", []),
        "error": job.get("error"),
    }


@app.post("/scan/cancel")
def scan_cancel(scan_id: str):
    job = _scan_jobs.get(scan_id)
    if job:
        job["cancelled"] = True
    return {"ok": True}


# ── UI ────────────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caption Tuner — Bias-ometer</title>
<style>
:root {
  --bg: #0d0f18;
  --surface: #161925;
  --surface2: #1e2235;
  --border: #2a2f45;
  --accent: #4f8ef7;
  --accent2: #7c6af7;
  --text: #dde2f0;
  --text-dim: #7a82a0;
  --pass: #3ecf6e;
  --fail: #f05060;
  --warn: #f0b040;
  --radius: 6px;
  --mono: 'JetBrains Mono', 'Fira Code', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; font-size: 13px; }
body { display: flex; flex-direction: column; }

/* ── Top bar ── */
.topbar {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 16px; background: var(--surface); border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.topbar h1 { font-size: 14px; font-weight: 600; color: var(--accent); white-space: nowrap; }
.topbar input[type=text] { flex: 1; min-width: 0; }
.badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-ocr { background: #1e3a26; color: var(--pass); }
.badge-no-ocr { background: #3a1e1e; color: var(--fail); }

/* ── Inputs ── */
input[type=text], input[type=number] {
  background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius);
  color: var(--text); padding: 5px 8px; font-size: 13px; outline: none;
  transition: border-color 0.15s;
}
input[type=text]:focus, input[type=number]:focus { border-color: var(--accent); }
input[type=number] { width: 72px; }
input[type=color] {
  width: 36px; height: 28px; padding: 2px; border-radius: var(--radius);
  border: 1px solid var(--border); background: var(--surface2); cursor: pointer;
}
input[type=range] { accent-color: var(--accent); width: 120px; }
select {
  background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius);
  color: var(--text); padding: 5px 8px; font-size: 13px; outline: none; cursor: pointer;
}
label { display: flex; align-items: center; gap: 6px; color: var(--text-dim); font-size: 12px; }
label span { color: var(--text); font-size: 12px; }
input[type=radio], input[type=checkbox] { accent-color: var(--accent); }

/* ── Buttons ── */
button {
  padding: 5px 12px; border-radius: var(--radius); border: none; cursor: pointer;
  font-size: 12px; font-weight: 600; transition: opacity 0.15s, transform 0.05s;
}
button:hover { opacity: 0.9; }
button:active { transform: scale(0.97); }
.btn-primary { background: var(--accent); color: #fff; }
.btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-success { background: #2a6e46; color: #fff; }
.btn-danger { background: #6e2a2a; color: #fff; }
.btn-scan { background: var(--accent2); color: #fff; padding: 6px 18px; font-size: 13px; }
button:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Main layout ── */
.main-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 0;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

/* ── Frame panel ── */
.frame-panel {
  display: flex; flex-direction: column; overflow: hidden;
  border-right: 1px solid var(--border);
}
.frame-wrap {
  flex: 1; overflow: auto; display: flex; align-items: flex-start; justify-content: flex-start;
  padding: 12px; background: #090b12;
  position: relative;
}
.frame-container {
  position: relative; display: inline-block; line-height: 0; cursor: crosshair;
}
#frameImg {
  display: block; max-width: 100%; max-height: 60vh;
  image-rendering: auto;
  user-select: none; -webkit-user-drag: none;
}
#regionOverlay {
  position: absolute; border: 2px solid var(--accent);
  background: rgba(79,142,247,0.12);
  pointer-events: none;
  box-shadow: 0 0 0 1px rgba(79,142,247,0.3);
}
#dragRect {
  position: absolute; border: 2px dashed var(--warn);
  background: rgba(240,176,64,0.08); pointer-events: none; display: none;
}
#pixelSample {
  position: absolute; width: 16px; height: 16px; border-radius: 50%;
  border: 2px solid #fff; pointer-events: none; display: none;
  box-shadow: 0 0 4px rgba(0,0,0,0.8);
}
.frame-placeholder {
  display: flex; align-items: center; justify-content: center;
  width: 640px; height: 360px; background: var(--surface2);
  border: 2px dashed var(--border); border-radius: var(--radius);
  color: var(--text-dim); font-size: 14px;
}

/* ── Jog bar ── */
.jog-bar {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 12px; background: var(--surface); border-top: 1px solid var(--border); flex-shrink: 0;
  flex-wrap: wrap;
}
.jog-bar .ts-label {
  font-family: var(--mono); font-size: 13px; color: var(--accent); min-width: 80px;
}
.jog-sep { color: var(--border); }
.pick-mode-active { background: #3a2e1a !important; color: var(--warn) !important; border: 1px solid var(--warn) !important; }

/* ── Config panel ── */
.config-panel {
  display: flex; flex-direction: column; gap: 0; overflow-y: auto; background: var(--surface);
}
.config-section {
  padding: 12px 14px; border-bottom: 1px solid var(--border);
}
.config-section h3 {
  font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px;
}
.coord-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
}
.coord-grid label { flex-direction: column; gap: 3px; }
.coord-grid label span { font-weight: 600; color: var(--text); }
.coord-grid input { width: 100%; }
.color-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.swatch {
  width: 36px; height: 28px; border-radius: var(--radius); border: 1px solid var(--border);
  cursor: pointer; flex-shrink: 0;
}
.color-row input[type=number] { width: 54px; }
.radio-row { display: flex; gap: 12px; }
.radio-row label { flex-direction: row; color: var(--text); }

/* ── OCR result ── */
.ocr-box {
  background: var(--surface2); border-radius: var(--radius);
  border: 1px solid var(--border); padding: 10px; margin-top: 8px;
}
.ocr-status-row {
  display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
}
.status-pill {
  padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700;
}
.pill-pass { background: #1a3d28; color: var(--pass); }
.pill-fail { background: #3d1a1a; color: var(--fail); }
.pill-skip { background: #2a2a1a; color: var(--warn); }
.ocr-lines { font-family: var(--mono); font-size: 11px; line-height: 1.6; color: var(--text-dim); }
.ocr-name { font-size: 13px; font-weight: 700; color: var(--pass); margin-top: 6px; }
.crop-preview { margin-top: 8px; }
.crop-preview img { max-width: 100%; border-radius: 3px; border: 1px solid var(--border); display: block; }
.sample-swatch-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 11px; color: var(--text-dim); }
.sample-swatch { width: 18px; height: 18px; border-radius: 3px; border: 1px solid var(--border); }

/* ── Export ── */
.export-area {
  margin-top: 8px; font-family: var(--mono); font-size: 11px;
  background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 10px; white-space: pre; overflow-x: auto; color: var(--text-dim);
  display: none;
}

/* ── Scan bar ── */
.scan-bar {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 10px 16px; background: var(--surface); border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.scan-bar label { color: var(--text-dim); }
.scan-status { font-family: var(--mono); font-size: 12px; color: var(--accent); }
.progress-bar {
  height: 4px; background: var(--border); border-radius: 2px; flex: 1; min-width: 100px;
}
.progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.2s; }

/* ── Scan results grid ── */
.scan-results-wrap {
  flex-shrink: 0; max-height: 320px; overflow-y: auto;
  background: var(--bg); border-top: 1px solid var(--border);
}
.scan-results-header {
  padding: 8px 14px; font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-dim); background: var(--surface);
  border-bottom: 1px solid var(--border); position: sticky; top: 0;
}
.scan-grid {
  display: flex; flex-wrap: wrap; gap: 8px; padding: 12px;
}
.scan-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  width: 156px; cursor: pointer; transition: border-color 0.15s; overflow: hidden;
}
.scan-card:hover { border-color: var(--accent); }
.scan-card.pass { border-color: var(--pass); }
.scan-card.fail-card { border-color: var(--border); }
.scan-card img { width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; }
.scan-card-body { padding: 6px 8px; }
.scan-card-ts { font-family: var(--mono); font-size: 11px; color: var(--text-dim); }
.scan-card-name { font-size: 12px; font-weight: 600; color: var(--pass); margin-top: 2px; word-break: break-word; }
.scan-card-lines { font-family: var(--mono); font-size: 10px; color: var(--text-dim); margin-top: 2px; line-height: 1.4; }
.scan-card-fail { font-size: 11px; color: var(--fail); margin-top: 2px; }

/* ── Captions Found panel ── */
.captions-panel {
  flex-shrink: 0; max-height: 260px; overflow-y: auto;
  background: var(--bg); border-top: 2px solid var(--accent2);
}
.captions-panel-header {
  padding: 8px 14px; font-size: 11px; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent2); background: var(--surface);
  border-bottom: 1px solid var(--border); position: sticky; top: 0;
}
.captions-list {
  display: flex; flex-direction: column; gap: 0;
}
.caption-row {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 14px; border-bottom: 1px solid var(--border);
  cursor: pointer; transition: background 0.12s;
}
.caption-row:hover { background: var(--surface2); }
.caption-ts {
  font-family: var(--mono); font-size: 12px; font-weight: 700;
  color: var(--accent2); white-space: nowrap; min-width: 56px;
}
.caption-crop {
  height: 36px; border-radius: 3px; border: 1px solid var(--border);
  object-fit: cover; flex-shrink: 0;
}
.caption-text {
  flex: 1; display: flex; flex-direction: column; gap: 2px;
}
.caption-name {
  font-size: 13px; font-weight: 600; color: var(--pass);
}
.caption-lines {
  font-family: var(--mono); font-size: 11px; color: var(--text-dim);
  line-height: 1.4;
}
.caption-empty {
  padding: 16px 14px; font-size: 12px; color: var(--text-dim);
  font-style: italic;
}

/* ── Lightbox ── */
.lightbox {
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85);
  z-index: 1000; align-items: center; justify-content: center; flex-direction: column; gap: 12px;
}
.lightbox.open { display: flex; }
.lightbox img { max-width: 90vw; max-height: 75vh; border-radius: var(--radius); }
.lightbox-meta { color: var(--text); font-size: 13px; text-align: center; }
.lightbox-close {
  position: absolute; top: 16px; right: 20px;
  background: var(--surface2); color: var(--text); border: 1px solid var(--border);
  font-size: 16px; padding: 6px 14px;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <h1>⚙ Caption Tuner</h1>
  <input type="text" id="videoPath" placeholder="/path/to/video.mp4" style="flex:1">
  <button class="btn-primary" onclick="loadVideo()">Load</button>
  <span id="durationInfo" style="color:var(--text-dim);font-family:var(--mono);font-size:12px;white-space:nowrap"></span>
  <span id="ocrBadge" class="badge badge-no-ocr">OCR ✗</span>
</div>

<!-- Main layout -->
<div class="main-layout" style="flex:1;min-height:0;overflow:hidden">

  <!-- Frame panel -->
  <div class="frame-panel">
    <div class="frame-wrap" id="frameWrap">
      <div class="frame-container" id="frameContainer"
           onmousedown="dragStart(event)"
           onmousemove="dragMove(event)"
           onmouseup="dragEnd(event)"
           onmouseleave="dragEnd(event)"
           onclick="frameClick(event)">
        <div class="frame-placeholder" id="framePlaceholder">Load a video file to begin</div>
        <img id="frameImg" src="" alt="" draggable="false" style="display:none"
             onload="imgLoaded()">
        <div id="regionOverlay" style="display:none"></div>
        <div id="dragRect"></div>
        <div id="pixelSample"></div>
        <canvas id="colorCanvas" style="display:none"></canvas>
      </div>
    </div>

    <!-- Jog bar -->
    <div class="jog-bar">
      <button class="btn-secondary" onclick="jog(-60)">−60s</button>
      <button class="btn-secondary" onclick="jog(-10)">−10s</button>
      <button class="btn-secondary" onclick="jog(-1)">−1s</button>
      <span class="ts-label" id="tsLabel">0.0s</span>
      <button class="btn-secondary" onclick="jog(1)">+1s</button>
      <button class="btn-secondary" onclick="jog(10)">+10s</button>
      <button class="btn-secondary" onclick="jog(60)">+60s</button>
      <span class="jog-sep">|</span>
      <label><span>Jump to</span>
        <input type="number" id="jumpTs" value="0" min="0" step="1" style="width:72px">s
      </label>
      <button class="btn-secondary" onclick="jumpTo()">Go</button>
      <span class="jog-sep">|</span>
      <button id="pickBtn" class="btn-secondary" onclick="togglePickMode()">🎨 Pick BG Colour</button>
      <span id="pickStatus" style="font-size:11px;color:var(--warn)"></span>
    </div>
  </div>

  <!-- Config panel -->
  <div class="config-panel">

    <!-- Region -->
    <div class="config-section">
      <h3>Caption Region</h3>
      <div style="margin-bottom:6px;font-size:11px;color:var(--text-dim)">
        Drag on frame to set • or edit values below
      </div>
      <div class="coord-grid">
        <label><span>X</span><input type="number" id="cfgX" value="0.05" step="0.005" min="0" max="0.999" oninput="updateOverlay()"></label>
        <label><span>Y</span><input type="number" id="cfgY" value="0.78" step="0.005" min="0" max="0.999" oninput="updateOverlay()"></label>
        <label><span>W</span><input type="number" id="cfgW" value="0.45" step="0.005" min="0.005" max="1" oninput="updateOverlay()"></label>
        <label><span>H</span><input type="number" id="cfgH" value="0.12" step="0.005" min="0.005" max="1" oninput="updateOverlay()"></label>
      </div>
      <div style="margin-top:8px;font-size:11px;color:var(--text-dim)">
        Load a preset:
        <select id="presetSelect" onchange="loadPreset()" style="margin-left:4px">
          <option value="">— select —</option>
          <option value="bbc_politics_live">bbc_politics_live</option>
          <option value="default">default</option>
        </select>
      </div>
    </div>

    <!-- Prescreen -->
    <div class="config-section">
      <h3>Background Pre-screen</h3>
      <div style="margin-bottom:8px">
        <label><input type="checkbox" id="usePrescreen" checked onchange="updatePrescreen()">
          <span>Enable colour pre-screen</span></label>
      </div>
      <div id="prescreenControls">
        <div class="color-row" style="margin-bottom:8px">
          <div class="swatch" id="bgSwatch" style="background:#ffffff" onclick="document.getElementById('bgColorPicker').click()"></div>
          <input type="color" id="bgColorPicker" value="#ffffff" style="display:none" oninput="colorPickerChanged()">
          <label><span>R</span><input type="number" id="bgR" value="255" min="0" max="255" oninput="rgbInputChanged()"></label>
          <label><span>G</span><input type="number" id="bgG" value="255" min="0" max="255" oninput="rgbInputChanged()"></label>
          <label><span>B</span><input type="number" id="bgB" value="255" min="0" max="255" oninput="rgbInputChanged()"></label>
        </div>
        <label><span>Tolerance</span>
          <input type="number" id="tolerance" value="30" min="0" max="255" style="width:60px">
        </label>
      </div>
    </div>

    <!-- OCR settings -->
    <div class="config-section">
      <h3>OCR Settings</h3>
      <div style="display:flex;flex-direction:column;gap:8px">
        <label><span>Source ID</span>
          <input type="text" id="sourceId" value="custom" style="flex:1">
        </label>
        <label><span>Name line index</span>
          <input type="number" id="nameLineIdx" value="0" min="0" max="9" style="width:60px">
        </label>
        <div>
          <div style="margin-bottom:5px;color:var(--text-dim);font-size:12px">Capture mode</div>
          <div class="radio-row">
            <label><input type="radio" name="captureMode" value="all_captions" checked> all_captions</label>
            <label><input type="radio" name="captureMode" value="names_only"> names_only</label>
          </div>
        </div>
        <label><span>Notes</span>
          <input type="text" id="cfgNotes" value="" placeholder="optional" style="flex:1">
        </label>
      </div>
    </div>

    <!-- Test OCR -->
    <div class="config-section">
      <h3>Test OCR</h3>
      <button class="btn-primary" style="width:100%" onclick="testOcr()">🔍 Test on Current Frame</button>
      <div id="ocrResult" style="display:none">
        <div class="ocr-box">
          <div class="ocr-status-row">
            <span id="ocrStatusPill" class="status-pill">—</span>
            <span id="ocrStatusMsg" style="font-size:11px;color:var(--text-dim)"></span>
          </div>
          <div id="ocrSampleRow" class="sample-swatch-row" style="display:none">
            <div id="ocrSampleSwatch" class="sample-swatch"></div>
            <span id="ocrSampleText"></span>
            <button class="btn-secondary" style="font-size:10px;padding:2px 6px" onclick="useSampledColor()">Use</button>
          </div>
          <div class="ocr-lines" id="ocrLines"></div>
          <div class="ocr-name" id="ocrName"></div>
          <div class="crop-preview" id="cropPreview"></div>
        </div>
      </div>
    </div>

    <!-- Export -->
    <div class="config-section">
      <h3>Export Config</h3>
      <button class="btn-secondary" style="width:100%" onclick="exportConfig()">📋 Generate sources.py snippet</button>
      <pre class="export-area" id="exportArea"></pre>
    </div>

  </div>
</div>

<!-- Scan bar -->
<div class="scan-bar">
  <strong style="color:var(--text-dim);font-size:11px;letter-spacing:.06em;text-transform:uppercase">Scan</strong>
  <label><span>From</span> <input type="number" id="scanStart" value="0" min="0" step="1" style="width:72px">s</label>
  <label><span>To</span> <input type="number" id="scanEnd" value="" placeholder="end" step="1" style="width:72px">s</label>
  <label><span>Every</span> <input type="number" id="scanInterval" value="2" min="0.5" step="0.5" style="width:60px">s</label>
  <button class="btn-scan" id="scanBtn" onclick="startScan()">▶ Scan File</button>
  <button class="btn-danger" id="cancelBtn" style="display:none" onclick="cancelScan()">■ Cancel</button>
  <div class="progress-bar" id="progressBar" style="display:none"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
  <span class="scan-status" id="scanStatus"></span>
</div>

<!-- Captions Found panel -->
<div class="captions-panel" id="captionsPanel" style="display:none">
  <div class="captions-panel-header" id="captionsPanelHeader">
    Captions Found
  </div>
  <div class="captions-list" id="captionsList"></div>
</div>

<!-- Scan results -->
<div class="scan-results-wrap" id="scanResultsWrap" style="display:none">
  <div class="scan-results-header" id="scanResultsHeader">Scan Results</div>
  <div class="scan-grid" id="scanGrid"></div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <button class="lightbox-close" onclick="closeLightbox()">✕ Close</button>
  <img id="lightboxImg" src="" alt="">
  <div class="lightbox-meta" id="lightboxMeta"></div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let currentTs = 0;
let duration = null;
let pickMode = false;
let sampledRgb = null;
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let scanId = null;
let pollTimer = null;
let frameW = 0, frameH = 0;

// ── Preset registry (mirrors sources.py) ──────────────────────────────────
const PRESETS = {
  bbc_politics_live: { x:0.085, y:0.783, w:0.914, h:0.134, bg_r:214, bg_g:218, bg_b:227, tol:20, name_line:0 },
  default:           { x:0.02, y:0.75, w:0.55, h:0.18, bg_r:255, bg_g:255, bg_b:255, tol:30, name_line:0 },
};

// ── Load video ─────────────────────────────────────────────────────────────
async function loadVideo() {
  const path = document.getElementById('videoPath').value.trim();
  if (!path) return;
  try {
    const res = await fetch('/load', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({path})
    });
    const data = await res.json();
    if (!res.ok) { alert(data.detail || 'Load failed'); return; }
    duration = data.duration;
    document.getElementById('durationInfo').textContent =
      `${data.name}  ·  ${fmtDur(duration)}`;
    document.getElementById('scanEnd').value = Math.floor(duration);
    await loadFrame(0);
  } catch(e) { alert('Error: ' + e.message); }
}

// ── Frame loading ──────────────────────────────────────────────────────────
async function loadFrame(ts) {
  if (ts !== undefined) currentTs = Math.max(0, ts);
  if (duration !== null) currentTs = Math.min(currentTs, duration);
  document.getElementById('tsLabel').textContent = currentTs.toFixed(1) + 's';
  document.getElementById('jumpTs').value = Math.round(currentTs);

  try {
    const res = await fetch(`/frame?ts=${currentTs}`);
    if (!res.ok) return;
    const data = await res.json();
    frameW = data.width; frameH = data.height;

    const img = document.getElementById('frameImg');
    img.src = 'data:image/png;base64,' + data.frame_b64;
    img.style.display = 'block';
    document.getElementById('framePlaceholder').style.display = 'none';

    // Draw to hidden canvas for pixel sampling
    const canvas = document.getElementById('colorCanvas');
    canvas.width = frameW; canvas.height = frameH;
    const ctx = canvas.getContext('2d');
    img.onload = () => { ctx.drawImage(img, 0, 0); updateOverlay(); };
  } catch(e) { console.error(e); }
}

function imgLoaded() { updateOverlay(); }

function jog(delta) { loadFrame(currentTs + delta); }
function jumpTo() { loadFrame(parseFloat(document.getElementById('jumpTs').value) || 0); }

// ── Region overlay ─────────────────────────────────────────────────────────
function updateOverlay() {
  const img = document.getElementById('frameImg');
  if (!img.naturalWidth) return;
  const ov = document.getElementById('regionOverlay');
  const rect = img.getBoundingClientRect();
  const containerRect = img.parentElement.getBoundingClientRect();
  const iw = img.offsetWidth, ih = img.offsetHeight;

  const x = parseFloat(document.getElementById('cfgX').value) || 0;
  const y = parseFloat(document.getElementById('cfgY').value) || 0;
  const w = parseFloat(document.getElementById('cfgW').value) || 0;
  const h = parseFloat(document.getElementById('cfgH').value) || 0;

  ov.style.display = 'block';
  ov.style.left   = (x * iw) + 'px';
  ov.style.top    = (y * ih) + 'px';
  ov.style.width  = (w * iw) + 'px';
  ov.style.height = (h * ih) + 'px';
}

// ── Drag to select region ──────────────────────────────────────────────────
function getRelPos(e) {
  const img = document.getElementById('frameImg');
  const rect = img.getBoundingClientRect();
  const iw = img.offsetWidth, ih = img.offsetHeight;
  return {
    x: Math.max(0, Math.min(1, (e.clientX - rect.left) / iw)),
    y: Math.max(0, Math.min(1, (e.clientY - rect.top) / ih)),
    px: e.clientX - rect.left,
    py: e.clientY - rect.top,
  };
}

function dragStart(e) {
  if (pickMode) return;
  if (!document.getElementById('frameImg').src) return;
  isDragging = true;
  const pos = getRelPos(e);
  dragStartX = pos.x; dragStartY = pos.y;
  const dr = document.getElementById('dragRect');
  dr.style.display = 'block';
  dr.style.left = pos.px + 'px'; dr.style.top = pos.py + 'px';
  dr.style.width = '0'; dr.style.height = '0';
  document.getElementById('regionOverlay').style.display = 'none';
}

function dragMove(e) {
  if (!isDragging) return;
  const img = document.getElementById('frameImg');
  const rect = img.getBoundingClientRect();
  const iw = img.offsetWidth, ih = img.offsetHeight;
  const curX = Math.max(0, Math.min(iw, e.clientX - rect.left));
  const curY = Math.max(0, Math.min(ih, e.clientY - rect.top));
  const sx = dragStartX * iw, sy = dragStartY * ih;
  const dr = document.getElementById('dragRect');
  const left = Math.min(sx, curX), top = Math.min(sy, curY);
  dr.style.left = left + 'px'; dr.style.top = top + 'px';
  dr.style.width = Math.abs(curX - sx) + 'px'; dr.style.height = Math.abs(curY - sy) + 'px';
}

function dragEnd(e) {
  if (!isDragging) return;
  isDragging = false;
  const img = document.getElementById('frameImg');
  const rect = img.getBoundingClientRect();
  const iw = img.offsetWidth, ih = img.offsetHeight;
  const endX = Math.max(0, Math.min(1, (e.clientX - rect.left) / iw));
  const endY = Math.max(0, Math.min(1, (e.clientY - rect.top) / ih));
  const x1 = Math.min(dragStartX, endX), x2 = Math.max(dragStartX, endX);
  const y1 = Math.min(dragStartY, endY), y2 = Math.max(dragStartY, endY);
  const w = x2 - x1, h = y2 - y1;
  if (w > 0.005 && h > 0.005) {
    setCoords(x1, y1, w, h);
  }
  document.getElementById('dragRect').style.display = 'none';
  updateOverlay();
}

function setCoords(x, y, w, h) {
  const fix = v => Math.round(v * 1000) / 1000;
  document.getElementById('cfgX').value = fix(x);
  document.getElementById('cfgY').value = fix(y);
  document.getElementById('cfgW').value = fix(Math.min(w, 1 - x));
  document.getElementById('cfgH').value = fix(Math.min(h, 1 - y));
}

// ── Pick colour from frame ─────────────────────────────────────────────────
function togglePickMode() {
  pickMode = !pickMode;
  const btn = document.getElementById('pickBtn');
  const status = document.getElementById('pickStatus');
  if (pickMode) {
    btn.textContent = '✕ Cancel Pick';
    btn.classList.add('pick-mode-active');
    status.textContent = '← Click a pixel on the frame to sample its colour';
    document.getElementById('frameContainer').style.cursor = 'crosshair';
  } else {
    btn.textContent = '🎨 Pick BG Colour';
    btn.classList.remove('pick-mode-active');
    status.textContent = '';
    document.getElementById('frameContainer').style.cursor = 'crosshair';
  }
}

function frameClick(e) {
  if (!pickMode) return;
  const img = document.getElementById('frameImg');
  if (!img.src || img.style.display === 'none') return;

  const rect = img.getBoundingClientRect();
  const scaleX = frameW / img.offsetWidth;
  const scaleY = frameH / img.offsetHeight;
  const px = Math.round((e.clientX - rect.left) * scaleX);
  const py = Math.round((e.clientY - rect.top) * scaleY);

  const canvas = document.getElementById('colorCanvas');
  const ctx = canvas.getContext('2d');
  const pixel = ctx.getImageData(px, py, 1, 1).data;
  const r = pixel[0], g = pixel[1], b = pixel[2];

  sampledRgb = [r, g, b];
  applyRgb(r, g, b);
  togglePickMode();
}

function applyRgb(r, g, b) {
  document.getElementById('bgR').value = r;
  document.getElementById('bgG').value = g;
  document.getElementById('bgB').value = b;
  const hex = '#' + [r,g,b].map(v => v.toString(16).padStart(2,'0')).join('');
  document.getElementById('bgColorPicker').value = hex;
  document.getElementById('bgSwatch').style.background = hex;
}

function colorPickerChanged() {
  const hex = document.getElementById('bgColorPicker').value;
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  applyRgb(r, g, b);
}

function rgbInputChanged() {
  const r = parseInt(document.getElementById('bgR').value)||0;
  const g = parseInt(document.getElementById('bgG').value)||0;
  const b = parseInt(document.getElementById('bgB').value)||0;
  const hex = '#' + [r,g,b].map(v => Math.max(0,Math.min(255,v)).toString(16).padStart(2,'0')).join('');
  document.getElementById('bgColorPicker').value = hex;
  document.getElementById('bgSwatch').style.background = hex;
}

function updatePrescreen() {
  document.getElementById('prescreenControls').style.opacity =
    document.getElementById('usePrescreen').checked ? '1' : '0.4';
}

// ── Presets ────────────────────────────────────────────────────────────────
function loadPreset() {
  const key = document.getElementById('presetSelect').value;
  if (!key || !PRESETS[key]) return;
  const p = PRESETS[key];
  setCoords(p.x, p.y, p.w, p.h);
  applyRgb(p.bg_r, p.bg_g, p.bg_b);
  document.getElementById('tolerance').value = p.tol;
  document.getElementById('nameLineIdx').value = p.name_line;
  document.getElementById('sourceId').value = key;
  document.getElementById('usePrescreen').checked = true;
  updatePrescreen();
  updateOverlay();
}

// ── OCR test ───────────────────────────────────────────────────────────────
async function testOcr() {
  const body = buildConfigBody();
  body.ts = currentTs;
  try {
    const res = await fetch('/ocr', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) { alert(data.detail || 'OCR failed'); return; }

    document.getElementById('ocrResult').style.display = 'block';
    const pill = document.getElementById('ocrStatusPill');
    const msg  = document.getElementById('ocrStatusMsg');

    if (!body.use_prescreen) {
      pill.className = 'status-pill pill-skip';
      pill.textContent = 'PRE-SCREEN OFF';
      msg.textContent = 'No colour check applied';
    } else if (data.prescreen_passed) {
      pill.className = 'status-pill pill-pass';
      pill.textContent = '✓ PRE-SCREEN PASS';
      msg.textContent = 'Region colour matched — OCR attempted';
    } else {
      pill.className = 'status-pill pill-fail';
      pill.textContent = '✗ PRE-SCREEN FAIL';
      msg.textContent = 'Region colour did not match — OCR skipped';
    }

    // Sampled pixel
    const sampleRow = document.getElementById('ocrSampleRow');
    if (data.sample_rgb) {
      const [sr, sg, sb] = data.sample_rgb;
      sampledRgb = [sr, sg, sb];
      const hex = '#' + [sr,sg,sb].map(v => v.toString(16).padStart(2,'0')).join('');
      document.getElementById('ocrSampleSwatch').style.background = hex;
      document.getElementById('ocrSampleText').textContent =
        `Sampled pixel at region origin: rgb(${sr},${sg},${sb})`;
      sampleRow.style.display = 'flex';
    } else {
      sampleRow.style.display = 'none';
    }

    document.getElementById('ocrLines').innerHTML = data.ocr_lines && data.ocr_lines.length
      ? 'Lines: ' + data.ocr_lines.map(l => `<em>${escHtml(l)}</em>`).join(' | ')
      : '<span style="color:var(--text-dim)">No OCR lines found</span>';

    const nameEl = document.getElementById('ocrName');
    nameEl.textContent = data.name ? '→ Name: ' + data.name : '';

    const cropEl = document.getElementById('cropPreview');
    if (data.crop_b64) {
      cropEl.innerHTML = '<img src="data:image/png;base64,' + data.crop_b64 + '" alt="Crop">';
    } else {
      cropEl.innerHTML = '';
    }
  } catch(e) { alert('OCR request failed: ' + e.message); }
}

function useSampledColor() {
  if (sampledRgb) applyRgb(...sampledRgb);
}

// ── Config builder ─────────────────────────────────────────────────────────
function buildConfigBody() {
  const mode = document.querySelector('input[name=captureMode]:checked');
  return {
    x: parseFloat(document.getElementById('cfgX').value),
    y: parseFloat(document.getElementById('cfgY').value),
    w: parseFloat(document.getElementById('cfgW').value),
    h: parseFloat(document.getElementById('cfgH').value),
    bg_r: parseInt(document.getElementById('bgR').value)||255,
    bg_g: parseInt(document.getElementById('bgG').value)||255,
    bg_b: parseInt(document.getElementById('bgB').value)||255,
    tolerance: parseInt(document.getElementById('tolerance').value)||30,
    use_prescreen: document.getElementById('usePrescreen').checked,
    name_line_index: parseInt(document.getElementById('nameLineIdx').value)||0,
    capture_mode: mode ? mode.value : 'all_captions',
    source_id: document.getElementById('sourceId').value || 'custom',
    notes: document.getElementById('cfgNotes').value || '',
  };
}

// ── Export ─────────────────────────────────────────────────────────────────
function exportConfig() {
  const c = buildConfigBody();
  const ps = c.use_prescreen
    ? `pre_screen=PreScreen(bg_colour=(${c.bg_r}, ${c.bg_g}, ${c.bg_b}), tolerance=${c.tolerance}),`
    : 'pre_screen=None,';
  const snippet = `    "${c.source_id}": SourceConfig(
        source_id="${c.source_id}",
        caption_region=CaptionRegion(
            x=${c.x.toFixed(3)}, y=${c.y.toFixed(3)},
            w=${c.w.toFixed(3)}, h=${c.h.toFixed(3)},
        ),
        name_line_index=${c.name_line_index},
        ${ps}
        notes="${c.notes}",
    ),`;
  const el = document.getElementById('exportArea');
  el.textContent = snippet;
  el.style.display = 'block';

  // Copy to clipboard
  navigator.clipboard.writeText(snippet).then(() => {
    const btn = el.previousElementSibling;
    btn.textContent = '✓ Copied!';
    setTimeout(() => { btn.textContent = '📋 Generate sources.py snippet'; }, 2000);
  }).catch(() => {});
}

// ── Scan ───────────────────────────────────────────────────────────────────
async function startScan() {
  const body = buildConfigBody();
  body.scan_start = parseFloat(document.getElementById('scanStart').value)||0;
  const endVal = document.getElementById('scanEnd').value;
  body.scan_end = endVal ? parseFloat(endVal) : null;
  body.interval = parseFloat(document.getElementById('scanInterval').value)||2;

  document.getElementById('scanBtn').disabled = true;
  document.getElementById('cancelBtn').style.display = 'inline-block';
  document.getElementById('progressBar').style.display = 'flex';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('scanStatus').textContent = 'Starting…';
  document.getElementById('scanGrid').innerHTML = '';
  document.getElementById('captionsList').innerHTML = '';
  document.getElementById('captionsPanel').style.display = 'none';
  document.getElementById('scanResultsWrap').style.display = 'block';

  try {
    const res = await fetch('/scan/start', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) { alert(data.detail || 'Scan failed'); resetScanUI(); return; }
    scanId = data.scan_id;
    pollTimer = setInterval(pollScan, 400);
  } catch(e) { alert('Scan error: ' + e.message); resetScanUI(); }
}

async function pollScan() {
  if (!scanId) return;
  try {
    const res = await fetch('/scan/poll?scan_id=' + scanId);
    const data = await res.json();
    if (!res.ok) { clearInterval(pollTimer); resetScanUI(); return; }

    const pct = data.total > 0 ? (data.current / data.total * 100) : 0;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('scanStatus').textContent =
      `${data.current} / ${data.total} frames`;
    const uniqueCount = (data.unique_captions || []).length;
    document.getElementById('scanResultsHeader').textContent =
      `Scan Results  ·  ${data.results.length} frames  ·  ` +
      `${data.results.filter(r=>r.prescreen_passed).length} passed pre-screen  ·  ` +
      `${data.results.filter(r=>r.name).length} names found  ·  ` +
      `${uniqueCount} unique`;

    renderScanResults(data.results);
    renderUniqueCaptions(data.unique_captions || []);

    if (data.status === 'done') {
      clearInterval(pollTimer); pollTimer = null;
      document.getElementById('scanStatus').textContent =
        `Done — ${data.results.length} frames scanned`;
      resetScanUI();
    }
  } catch(e) { clearInterval(pollTimer); resetScanUI(); }
}

function renderScanResults(results) {
  const grid = document.getElementById('scanGrid');
  // Append only new cards
  const existing = grid.children.length;
  for (let i = existing; i < results.length; i++) {
    const r = results[i];
    const card = document.createElement('div');
    card.className = 'scan-card ' + (r.name ? 'pass' : (r.error ? 'fail-card' : ''));
    card.onclick = () => openLightbox(r);

    let imgHtml = r.frame_b64
      ? `<img src="data:image/png;base64,${r.frame_b64}" loading="lazy" alt="">`
      : `<div style="aspect-ratio:16/9;background:var(--surface2);display:flex;align-items:center;justify-content:center;color:var(--fail);font-size:10px">Failed</div>`;

    let bodyHtml = `<div class="scan-card-ts">${r.ts.toFixed(1)}s</div>`;
    if (r.error) {
      bodyHtml += `<div class="scan-card-fail" title="${escHtml(r.error)}">⚠ Error</div>`;
    } else {
      const ps = r.prescreen_passed
        ? '<span style="color:var(--pass);font-size:10px">✓ PS</span>'
        : '<span style="color:var(--text-dim);font-size:10px">✗ PS</span>';
      bodyHtml += ps;
      if (r.name) bodyHtml += `<div class="scan-card-name">${escHtml(r.name)}</div>`;
      if (r.ocr_lines && r.ocr_lines.length) {
        bodyHtml += `<div class="scan-card-lines">${r.ocr_lines.slice(0,2).map(escHtml).join('<br>')}</div>`;
      }
    }
    card.innerHTML = imgHtml + `<div class="scan-card-body">${bodyHtml}</div>`;
    grid.appendChild(card);
  }
}

function renderUniqueCaptions(captions) {
  const panel = document.getElementById('captionsPanel');
  const header = document.getElementById('captionsPanelHeader');
  const list = document.getElementById('captionsList');

  if (!captions || captions.length === 0) {
    panel.style.display = 'none';
    return;
  }

  panel.style.display = 'block';
  header.textContent = `Captions Found  ·  ${captions.length} unique`;

  // Only append newly arrived rows (live update during scan)
  const existing = list.children.length;
  for (let i = existing; i < captions.length; i++) {
    const c = captions[i];
    const row = document.createElement('div');
    row.className = 'caption-row';
    row.onclick = () => {
      // Seek video to this timestamp and open lightbox
      currentTs = c.ts;
      document.getElementById('jumpTs').value = c.ts.toFixed(2);
      openLightbox(c);
    };

    const ts = fmtTs(c.ts);
    const nameHtml = c.name
      ? `<div class="caption-name">${escHtml(c.name)}</div>` : '';
    const linesText = (c.ocr_lines || [])
      .filter(l => !c.name || l !== c.name)   // avoid repeating name
      .map(l => escHtml(l)).join(' · ');
    const linesHtml = linesText
      ? `<div class="caption-lines">${linesText}</div>` : '';
    const cropHtml = c.crop_b64
      ? `<img class="caption-crop"
             src="data:image/png;base64,${c.crop_b64}" alt="crop">`
      : '';

    row.innerHTML = `
      <span class="caption-ts">${ts}</span>
      ${cropHtml}
      <div class="caption-text">
        ${nameHtml}
        ${linesHtml}
      </div>`;
    list.appendChild(row);
  }
}

async function cancelScan() {
  if (scanId) await fetch('/scan/cancel?scan_id=' + scanId, {method:'POST'});
  clearInterval(pollTimer); pollTimer = null;
  resetScanUI();
  document.getElementById('scanStatus').textContent = 'Cancelled';
}

function resetScanUI() {
  document.getElementById('scanBtn').disabled = false;
  document.getElementById('cancelBtn').style.display = 'none';
}

// ── Lightbox ───────────────────────────────────────────────────────────────
function openLightbox(r) {
  if (!r.frame_b64) return;
  document.getElementById('lightboxImg').src = 'data:image/png;base64,' + r.frame_b64;
  let meta = `${r.ts.toFixed(1)}s  ·  Pre-screen: ${r.prescreen_passed ? '✓ pass' : '✗ fail'}`;
  if (r.name) meta += `  ·  Name: ${r.name}`;
  if (r.ocr_lines && r.ocr_lines.length) meta += `  ·  Lines: ${r.ocr_lines.join(' | ')}`;
  document.getElementById('lightboxMeta').textContent = meta;
  document.getElementById('lightbox').classList.add('open');
}

function closeLightbox() { document.getElementById('lightbox').classList.remove('open'); }

// ── Utilities ──────────────────────────────────────────────────────────────
function fmtDur(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60);
  return h > 0
    ? `${h}h ${String(m).padStart(2,'0')}m ${String(sec).padStart(2,'0')}s`
    : `${m}m ${String(sec).padStart(2,'0')}s`;
}

function fmtTs(s) {
  const m = Math.floor(s/60), sec = Math.floor(s%60);
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Init ───────────────────────────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    document.getElementById('ocrBadge').className = 'badge ' +
      (data.ocr_available ? 'badge-ocr' : 'badge-no-ocr');
    document.getElementById('ocrBadge').textContent =
      data.ocr_available ? 'OCR ✓' : 'OCR ✗';
    if (!data.ocr_available) {
      document.getElementById('ocrBadge').title =
        'Tesseract not installed — install with: brew install tesseract && pip install pytesseract';
    }
  } catch(e) {}

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft')  jog(e.shiftKey ? -10 : -1);
    if (e.key === 'ArrowRight') jog(e.shiftKey ? 10  :  1);
    if (e.key === 'Escape')     closeLightbox();
    if (e.key === 't' || e.key === 'T') testOcr();
  });
})();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML

@app.get("/status")
def status():
    return {
        "ocr_available": OCR_AVAILABLE,
        "video_loaded": _video_file is not None,
        "duration": _video_duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bias-ometer Caption Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port", type=int, default=7654, help="Port to listen on (default: 7654)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n  Caption Tuner — Bias-ometer")
    print(f"  ─────────────────────────────────────────")
    print(f"  URL    : http://{args.host}:{args.port}")
    print(f"  OCR    : {'✓ available' if OCR_AVAILABLE else '✗ unavailable (pip install pytesseract)'}")
    print(f"  spaCy  : {'✓ loaded' if _nlp else '✗ unavailable (pip install spacy)'}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
