"""
screen_capture.py
========================================================================
Extracts video frames at speaker-change timestamps and uses Claude Vision
to read on-screen text (lower-thirds, chyrons, name cards) for automatic
speaker identification.

Two complementary approaches are combined:
  1. Screenshot  — save a frame as a visual audit trail (always done)
  2. OCR / Vision — send the frame to Claude Vision to extract name,
                    title, and organisation from lower-third graphics

Works with:
  • Local video files (.mp4, .mkv, .mov, .avi, …)
  • YouTube URLs  (direct CDN stream obtained via yt-dlp; no full download)

Requirements:
  pip install anthropic yt-dlp          (yt-dlp already required by project)
  brew install ffmpeg                   (already required by project)

Environment:
  ANTHROPIC_API_KEY  — loaded automatically from .env via main.py / dotenv

Typical usage (called from main.py after diarization):
  from screen_capture import ScreenCapture

  sc = ScreenCapture(output_dir=out_dir)
  captures = sc.capture_new_speakers(
      video_source="https://www.youtube.com/watch?v=...",
      result=diarization_result,
      session_id=session_id,
  )
  for sid, cap in captures.items():
      print(sid, cap.suggested_name, cap.confidence)

Stand-alone test (no diarization needed):
  python screen_capture.py "https://www.youtube.com/watch?v=..." --timestamp 42
  python screen_capture.py /path/to/video.mp4 --timestamp 10.5
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ─── How far into a speaker turn to grab the frame ────────────────────────────
# Offset from turn start (seconds). A small pause lets cut transitions settle.
_FRAME_OFFSET_SECS: float = 1.5

# ─── How many frames to sample per speaker (picks the one with most text) ─────
_CANDIDATE_FRAMES: int = 3
_CANDIDATE_SPREAD_SECS: float = 4.0   # spread candidates over this window


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    """
    Result of a single frame extraction + Claude Vision analysis.

    Attributes
    ----------
    speaker_id : str
        Ephemeral pyannote ID (e.g. "SPEAKER_00").
    timestamp : float
        Position in the video (seconds) where the frame was captured.
    frame_path : Optional[Path]
        Saved PNG file, or None if ffmpeg extraction failed.
    raw_text : Optional[str]
        All visible text found on screen by Claude Vision.
    suggested_name : Optional[str]
        Best-guess full name extracted from lower-third / chyron.
    suggested_title : Optional[str]
        Job title or role (e.g. "Political Correspondent").
    suggested_org : Optional[str]
        Organisation or affiliation (e.g. "Reuters").
    confidence : Optional[str]
        Vision model self-rating: "high" | "medium" | "low".
    vision_used : bool
        Whether Claude Vision was actually called for this frame.
    error : Optional[str]
        Human-readable error message if something went wrong.
    """
    speaker_id: str
    timestamp: float
    frame_path: Optional[Path]
    raw_text: Optional[str]
    suggested_name: Optional[str]
    suggested_title: Optional[str]
    suggested_org: Optional[str]
    confidence: Optional[str]
    vision_used: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frame_path"] = str(self.frame_path) if self.frame_path else None
        return d

    @property
    def identified(self) -> bool:
        """True if Vision found a name with at least medium confidence."""
        return bool(
            self.suggested_name
            and self.confidence in ("high", "medium")
        )

    def summary(self) -> str:
        """One-line human-readable summary for CLI output."""
        name  = self.suggested_name  or "?"
        title = f" · {self.suggested_title}" if self.suggested_title else ""
        org   = f" ({self.suggested_org})" if self.suggested_org else ""
        conf  = f" [{self.confidence}]" if self.confidence else ""
        err   = f"  ⚠ {self.error}" if self.error else ""
        return f"{self.speaker_id:<14}  {name}{title}{org}{conf}{err}"


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class ScreenCapture:
    """
    Orchestrates frame extraction and Claude Vision speaker identification.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for saved frames.  Frames are written to:
            <output_dir>/<session_id>/frames/<speaker_id>_<timestamp>.png
    use_vision : bool
        If True (default), send each frame to Claude Vision for OCR.
        Set False to only save screenshots without API calls.
    anthropic_api_key : Optional[str]
        Anthropic API key.  If None, reads ANTHROPIC_API_KEY from the
        environment (loaded from .env by main.py).
    multi_frame : bool
        If True, capture _CANDIDATE_FRAMES frames per speaker and pick the
        one that Vision identifies most confidently.  Slower but more
        reliable for content that doesn't always show a lower-third.
    cookies_from_browser : Optional[str]
        Browser name for yt-dlp cookie auth (e.g. "safari", "chrome").
        Only needed for YouTube sources.
    cookies_file : Optional[str | Path]
        Path to a Netscape-format cookies.txt for yt-dlp.
    """

    def __init__(
        self,
        output_dir: str | Path = "output",
        use_vision: bool = True,
        anthropic_api_key: Optional[str] = None,
        multi_frame: bool = True,
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str | Path] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_vision = use_vision
        self.multi_frame = multi_frame
        self.cookies_from_browser = cookies_from_browser
        self.cookies_file = Path(cookies_file) if cookies_file else None

        self._api_key = (
            anthropic_api_key
            or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        )
        if use_vision and not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or set use_vision=False."
            )

        # Cache resolved direct video URLs (YouTube CDN links, etc.)
        self._url_cache: dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def capture_speaker(
        self,
        video_source: str,
        speaker_id: str,
        timestamp: float,
        session_id: str,
    ) -> FrameResult:
        """
        Extract a frame at `timestamp` for one speaker and optionally
        analyse it with Claude Vision.

        Parameters
        ----------
        video_source : str
            Local file path or YouTube URL.
        speaker_id : str
            Ephemeral speaker ID (e.g. "SPEAKER_00").
        timestamp : float
            Seconds into the video.
        session_id : str
            Used to organise output files into per-session sub-directories.

        Returns
        -------
        FrameResult
        """
        frame_dir = self._frame_dir(session_id)
        playback_url = self._resolve_source(video_source)

        if self.multi_frame:
            return self._capture_best_frame(
                playback_url, speaker_id, timestamp, frame_dir
            )
        else:
            frame_path = frame_dir / f"{speaker_id}_{timestamp:.2f}.png"
            ok = self._extract_frame(playback_url, timestamp, frame_path)
            if not ok:
                return FrameResult(
                    speaker_id=speaker_id, timestamp=timestamp,
                    frame_path=None, raw_text=None,
                    suggested_name=None, suggested_title=None,
                    suggested_org=None, confidence=None,
                    error="ffmpeg frame extraction failed",
                )
            if self.use_vision:
                return self._analyse_frame(speaker_id, timestamp, frame_path)
            return FrameResult(
                speaker_id=speaker_id, timestamp=timestamp,
                frame_path=frame_path, raw_text=None,
                suggested_name=None, suggested_title=None,
                suggested_org=None, confidence=None,
            )

    def capture_new_speakers(
        self,
        video_source: str,
        result,           # diarizer.DiarizationResult
        session_id: str,
    ) -> dict[str, FrameResult]:
        """
        Capture the first frame for every unique speaker in a
        DiarizationResult and return {speaker_id: FrameResult}.

        Only the *first* turn for each speaker is processed; subsequent
        turns are skipped.  The frame is sampled slightly after the turn
        start to avoid capture during scene transitions.

        Parameters
        ----------
        video_source : str
            Local file path or YouTube URL.
        result : DiarizationResult
            Output from NewsDiarizer.process().
        session_id : str
            Used to organise output files.

        Returns
        -------
        dict[str, FrameResult]
        """
        seen: set[str] = set()
        captures: dict[str, FrameResult] = {}

        for turn in result.turns:
            sid = turn.speaker_id
            if sid in seen:
                continue
            seen.add(sid)

            turn_duration = turn.end - turn.start
            # Aim _FRAME_OFFSET_SECS in; clamp to 80% of the turn length
            ts = turn.start + min(_FRAME_OFFSET_SECS, turn_duration * 0.8)

            log.info(
                f"Screen capture: {sid} at {ts:.1f}s "
                f"(turn {turn.start:.1f}–{turn.end:.1f}s)"
            )
            captures[sid] = self.capture_speaker(
                video_source, sid, ts, session_id
            )

        return captures

    # ──────────────────────────────────────────────────────────────────────────
    # Source resolution  (YouTube → direct CDN URL)
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_source(self, source: str) -> str:
        """
        If `source` looks like a YouTube URL, use yt-dlp to obtain the
        direct CDN stream URL that ffmpeg can seek into without a full
        download.  Local file paths are returned unchanged.
        """
        if not _is_youtube(source):
            return source   # local file path — ffmpeg handles it directly

        if source in self._url_cache:
            log.debug(f"Using cached stream URL for: {source}")
            return self._url_cache[source]

        log.info("Resolving YouTube stream URL for frame extraction …")
        url = self._get_youtube_stream_url(source)
        self._url_cache[source] = url
        log.info(f"  Stream URL resolved ({url[:60]}…)")
        return url

    def _get_youtube_stream_url(self, youtube_url: str) -> str:
        """
        Call yt-dlp to get a direct CDN video URL.

        Prefers mp4 so ffmpeg can seek cheaply with a byte-range request.
        Falls back to the best available format.
        """
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError(
                "yt-dlp is required for YouTube sources. Run: pip install yt-dlp"
            )

        opts: dict = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        if self.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self.cookies_from_browser,)
        elif self.cookies_file and self.cookies_file.exists():
            opts["cookiefile"] = str(self.cookies_file)

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)

        # Prefer the top-level URL; fall back to the first format entry
        url = info.get("url") or ""
        if not url and info.get("formats"):
            url = info["formats"][-1].get("url", "")

        if not url:
            raise RuntimeError(
                "yt-dlp could not resolve a direct stream URL for this video.\n"
                "Try passing --cookies-from-browser safari (or chrome)."
            )
        return url

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-frame selection
    # ──────────────────────────────────────────────────────────────────────────

    def _capture_best_frame(
        self,
        playback_url: str,
        speaker_id: str,
        base_timestamp: float,
        frame_dir: Path,
    ) -> FrameResult:
        """
        Extract _CANDIDATE_FRAMES evenly spaced around `base_timestamp`,
        run Vision on each, and return the result with the highest-confidence
        name identification.  Falls back to the first frame if none identify.
        """
        spread = _CANDIDATE_SPREAD_SECS
        n = _CANDIDATE_FRAMES
        offsets = [spread * i / (n - 1) for i in range(n)] if n > 1 else [0.0]

        candidates: list[FrameResult] = []
        for i, offset in enumerate(offsets):
            ts = base_timestamp + offset
            frame_path = frame_dir / f"{speaker_id}_{ts:.2f}_c{i}.png"
            ok = self._extract_frame(playback_url, ts, frame_path)
            if not ok:
                continue

            if self.use_vision:
                result = self._analyse_frame(speaker_id, ts, frame_path)
            else:
                result = FrameResult(
                    speaker_id=speaker_id, timestamp=ts,
                    frame_path=frame_path, raw_text=None,
                    suggested_name=None, suggested_title=None,
                    suggested_org=None, confidence=None,
                )
            candidates.append(result)

        if not candidates:
            return FrameResult(
                speaker_id=speaker_id, timestamp=base_timestamp,
                frame_path=None, raw_text=None,
                suggested_name=None, suggested_title=None,
                suggested_org=None, confidence=None,
                error="All frame extractions failed",
            )

        # Pick the best: high > medium > low > None, breaking ties by most text
        _conf_rank = {"high": 3, "medium": 2, "low": 1}
        best = max(
            candidates,
            key=lambda r: (
                _conf_rank.get(r.confidence or "", 0),
                len(r.raw_text or ""),
            ),
        )

        # Tidy up — delete the candidate frames we didn't pick
        for r in candidates:
            if r is not best and r.frame_path and r.frame_path.exists():
                try:
                    # Rename winning candidate to a clean filename
                    pass
                except OSError:
                    pass

        # Rename winning frame to a clean, predictable name
        if best.frame_path and best.frame_path.exists():
            clean_name = frame_dir / f"{speaker_id}_{best.timestamp:.2f}.png"
            best.frame_path.rename(clean_name)
            best.frame_path = clean_name

        # Delete losers
        for r in candidates:
            if r is not best and r.frame_path and r.frame_path.exists():
                r.frame_path.unlink(missing_ok=True)

        return best

    # ──────────────────────────────────────────────────────────────────────────
    # Frame extraction  (ffmpeg)
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_frame(
        self, source: str, timestamp: float, out_path: Path
    ) -> bool:
        """
        Use ffmpeg to grab a single frame from `source` at `timestamp`.

        Works with local files and direct HTTPS stream URLs.  The `-ss`
        flag is placed *before* `-i` so ffmpeg uses fast keyframe seeking
        (O(1)) rather than decoding every frame up to the timestamp.
        """
        cmd = [
            "ffmpeg",
            "-ss", f"{timestamp:.3f}",  # seek before input → fast
            "-i", source,
            "-vframes", "1",            # grab exactly one frame
            "-q:v", "2",               # near-lossless quality for PNG
            "-vf", "scale=1280:-1",    # normalise width; keep aspect ratio
            "-y",                       # overwrite output silently
            str(out_path),
        ]

        log.debug(f"ffmpeg: seeking to {timestamp:.1f}s in {source[:60]}")
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=45,
            )
            if not out_path.exists() or out_path.stat().st_size == 0:
                log.warning(f"  ffmpeg produced an empty file at {out_path}")
                return False
            log.debug(f"  Frame saved: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")
            return True

        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace")[:300]
            log.warning(f"  ffmpeg error: {stderr}")
            return False
        except subprocess.TimeoutExpired:
            log.warning("  ffmpeg timed out after 45 s")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Claude Vision analysis
    # ──────────────────────────────────────────────────────────────────────────

    _VISION_PROMPT = """\
You are analysing a video frame to identify the speaker shown on screen.

Look carefully for any on-screen text that identifies a person, including:
  • Lower-third graphics (the name/title bar often shown at the bottom of
    news broadcasts and interviews)
  • Name tags, chyrons, or super-imposed captions
  • Watermarks or corner bugs that include a presenter's name
  • On-screen titles, placards, or introduction cards

Extract exactly what you see. Do not guess or invent names.

Respond ONLY with a JSON object in this exact format — no preamble, no markdown
fences, no extra keys:
{
  "raw_text": "<every piece of text visible anywhere on screen, or null>",
  "suggested_name": "<full name of the speaker shown in on-screen text, or null>",
  "suggested_title": "<their job title or role shown on screen, or null>",
  "suggested_org": "<their organisation or affiliation shown on screen, or null>",
  "confidence": "<high|medium|low — how certain are you that the name is correct>"
}

Rules:
  • Set confidence "high" only when a clear lower-third or name card is present.
  • Set confidence "medium" when text is present but partially obscured or ambiguous.
  • Set confidence "low" when guessing from context alone.
  • If no identifying text is visible, return null for name/title/org and omit confidence.
"""

    def _analyse_frame(
        self, speaker_id: str, timestamp: float, frame_path: Path
    ) -> FrameResult:
        """
        Send `frame_path` to Claude Vision and parse the structured response.
        """
        import anthropic

        try:
            image_bytes = frame_path.read_bytes()
        except OSError as e:
            return FrameResult(
                speaker_id=speaker_id, timestamp=timestamp,
                frame_path=frame_path, raw_text=None,
                suggested_name=None, suggested_title=None,
                suggested_org=None, confidence=None,
                vision_used=False, error=f"Cannot read frame file: {e}",
            )

        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": self._VISION_PROMPT,
                            },
                        ],
                    }
                ],
            )

            raw_response = response.content[0].text.strip()
            data = _parse_vision_json(raw_response)

            name  = data.get("suggested_name")  or None
            title = data.get("suggested_title") or None
            org   = data.get("suggested_org")   or None
            conf  = data.get("confidence")       or None

            log.info(
                f"  Vision → {name or '(no name)'}"
                + (f" · {title}" if title else "")
                + (f" [{conf}]" if conf else "")
            )

            return FrameResult(
                speaker_id=speaker_id,
                timestamp=timestamp,
                frame_path=frame_path,
                raw_text=data.get("raw_text") or None,
                suggested_name=name,
                suggested_title=title,
                suggested_org=org,
                confidence=conf,
                vision_used=True,
            )

        except anthropic.APIError as e:
            log.warning(f"  Anthropic API error: {e}")
            return FrameResult(
                speaker_id=speaker_id, timestamp=timestamp,
                frame_path=frame_path, raw_text=None,
                suggested_name=None, suggested_title=None,
                suggested_org=None, confidence=None,
                vision_used=True, error=f"API error: {e}",
            )
        except (ValueError, KeyError) as e:
            log.warning(f"  Vision response parse error: {e}")
            return FrameResult(
                speaker_id=speaker_id, timestamp=timestamp,
                frame_path=frame_path, raw_text=None,
                suggested_name=None, suggested_title=None,
                suggested_org=None, confidence=None,
                vision_used=True, error=f"Parse error: {e}",
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _frame_dir(self, session_id: str) -> Path:
        d = self.output_dir / session_id / "frames"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_youtube(source: str) -> bool:
    """Return True if the source looks like a YouTube URL."""
    return any(
        host in source
        for host in ("youtube.com", "youtu.be", "yt.be")
    )


def _parse_vision_json(text: str) -> dict:
    """
    Robustly parse the JSON blob returned by Claude Vision.

    Handles:
    • Plain JSON (ideal)
    • JSON wrapped in ```json … ``` fences
    • Leading/trailing whitespace
    """
    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        # parts[1] is the content between the first pair of fences
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Vision model: {e}\nRaw: {text[:200]}")


def print_capture_summary(captures: dict[str, FrameResult]) -> None:
    """
    Pretty-print a table of capture results to stdout.
    Designed to slot into the existing main.py summary blocks.
    """
    if not captures:
        return

    w = 72
    print("\n── Speaker identification (screen capture + Vision) " + "─" * 20)
    identified = 0
    for sid in sorted(captures):
        cap = captures[sid]
        print(f"  {cap.summary()}")
        if cap.frame_path:
            print(f"  {'':14}  📷 {cap.frame_path}")
        if cap.identified:
            identified += 1

    total = len(captures)
    print(f"\n  Identified {identified}/{total} speakers automatically.")
    if identified < total:
        unidentified = [
            sid for sid, cap in captures.items()
            if not cap.identified
        ]
        print(
            "  Run  python main.py link  to map remaining speakers:\n"
            + "\n".join(f"    {sid}" for sid in unidentified)
        )
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Stand-alone CLI  (for testing without full diarization pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    """
    Quick test harness.

    Usage:
        python screen_capture.py <video_source> --timestamp <seconds>
        python screen_capture.py <video_source> --timestamp 30 --no-vision
        python screen_capture.py <video_source> --timestamp 15 \\
            --cookies-from-browser safari
    """
    import argparse
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Extract a frame and optionally identify on-screen speaker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        help="Local video file path or YouTube URL",
    )
    parser.add_argument(
        "--timestamp", "-t",
        type=float,
        default=10.0,
        help="Seconds into the video to capture (default: 10)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output/test_captures",
        help="Directory for output frames (default: output/test_captures)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip Claude Vision; only save the screenshot",
    )
    parser.add_argument(
        "--no-multi-frame",
        action="store_true",
        help="Capture a single frame instead of sampling multiple candidates",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Browser for yt-dlp cookie auth (safari | chrome | firefox …)",
    )
    parser.add_argument(
        "--cookies",
        metavar="FILE",
        help="Path to Netscape cookies.txt for yt-dlp",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    sc = ScreenCapture(
        output_dir=args.output_dir,
        use_vision=not args.no_vision,
        multi_frame=not args.no_multi_frame,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies,
    )

    print(f"\nCapturing frame at {args.timestamp:.1f}s from: {args.source}")
    cap = sc.capture_speaker(
        video_source=args.source,
        speaker_id="TEST_SPEAKER",
        timestamp=args.timestamp,
        session_id="test",
    )

    print("\n── Result " + "─" * 60)
    print(f"  Frame saved  : {cap.frame_path or '(failed)'}")
    print(f"  Vision used  : {cap.vision_used}")
    print(f"  Raw text     : {(cap.raw_text or '').strip()[:120] or '(none)'}")
    print(f"  Name         : {cap.suggested_name  or '—'}")
    print(f"  Title        : {cap.suggested_title or '—'}")
    print(f"  Organisation : {cap.suggested_org   or '—'}")
    print(f"  Confidence   : {cap.confidence      or '—'}")
    if cap.error:
        print(f"  Error        : {cap.error}")
    print()

    if cap.frame_path:
        print(f"Open the frame with:\n  open {cap.frame_path}\n")


if __name__ == "__main__":
    _cli()
