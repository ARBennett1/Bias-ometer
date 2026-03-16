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
from typing import Any, Optional, cast

log = logging.getLogger(__name__)

# ─── Scan parameters ─────────────────────────────────────────────────────────
_SCAN_INTERVAL_SECS: float = 2.0       # extract one frame every N seconds
_SCAN_WINDOW_SECS: float = 60.0        # scan at most this many seconds of a turn
_MAX_SCAN_FRAMES_REMOTE: int = 20      # cap for YouTube CDN sources
_PRESCREEN_MIN_CHARS: int = 8          # min Tesseract char count to pass prescreen
_BOTTOM_STRIP_FRACTION: float = 0.25   # bottom fraction of frame to crop for OCR
_PIXEL_STD_THRESHOLD: float = 40.0    # pixel std threshold for contrast heuristic


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
    prescreen_method : Optional[str]
        Method used to pre-screen frames for text before Vision:
        'tesseract' | 'pixel_heuristic' | 'none'.
        None means pre-screening was not attempted (e.g. fallback frame path).
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
    prescreen_method: Optional[str] = None

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
    scan_window_secs : float
        Seconds of each speaker's first turn to scan for lower-third text.
        Default: 60.0.
    max_scan_frames_remote : int
        Maximum frames to extract per speaker for YouTube CDN sources.
        Default: 20.
    text_prescreen : bool
        If True (default), pre-screen frames for lower-third text using
        Tesseract OCR (local sources) or a pixel-contrast heuristic
        (YouTube CDN sources) before committing to a Vision API call.
        Set False to send all extracted frames to Vision (up to 3).
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
        scan_window_secs: float = _SCAN_WINDOW_SECS,
        max_scan_frames_remote: int = _MAX_SCAN_FRAMES_REMOTE,
        text_prescreen: bool = True,
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str | Path] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_vision = use_vision
        self.scan_window_secs = scan_window_secs
        self.max_scan_frames_remote = max_scan_frames_remote
        self.text_prescreen = text_prescreen
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

        # Attempt to import pytesseract for OCR pre-screening
        self._tesseract_available = False
        if self.text_prescreen:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
                log.info("Tesseract available — will use OCR pre-screening for local sources")
            except Exception:
                log.info("Tesseract not available — will use pixel heuristic for pre-screening")

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
        is_remote = _is_youtube(video_source)
        return self._scan_turn_for_best_frame(
            playback_url=playback_url,
            speaker_id=speaker_id,
            turn_start=timestamp,
            turn_end=None,
            frame_dir=frame_dir,
            is_remote=is_remote,
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
        first_turns: dict[str, tuple] = {}  # speaker_id → (turn_start, turn_end)

        # Collect first turn per speaker
        for turn in result.turns:
            sid = turn.speaker_id
            if sid not in seen:
                seen.add(sid)
                first_turns[sid] = (turn.start, turn.end)

        # Detect local audio (not YouTube, not a video file extension)
        _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        is_local_audio = (
            not _is_youtube(video_source)
            and Path(video_source).suffix.lower() in _AUDIO_EXTS
        )

        is_remote = _is_youtube(video_source)
        playback_url = self._resolve_source(video_source)

        for sid, (turn_start, turn_end) in first_turns.items():
            if is_local_audio:
                waveform_path = self._generate_waveform_image(
                    audio_path=Path(video_source),
                    speaker_id=sid,
                    turn_start=turn_start,
                    turn_end=turn_end,
                    session_id=session_id,
                )
                captures[sid] = FrameResult(
                    speaker_id=sid,
                    timestamp=turn_start,
                    frame_path=waveform_path,
                    raw_text=None,
                    suggested_name=None,
                    suggested_title=None,
                    suggested_org=None,
                    confidence=None,
                    vision_used=False,
                )
            else:
                log.info(
                    f"Screen capture: {sid} — scanning "
                    f"{turn_start:.1f}s–{min(turn_end, turn_start + self.scan_window_secs):.1f}s"
                )
                captures[sid] = self._scan_turn_for_best_frame(
                    playback_url=playback_url,
                    speaker_id=sid,
                    turn_start=turn_start,
                    turn_end=turn_end,
                    frame_dir=self._frame_dir(session_id),
                    is_remote=is_remote,
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

        opts: dict[str, object] = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        if self.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self.cookies_from_browser,)
        elif self.cookies_file and self.cookies_file.exists():
            opts["cookiefile"] = str(self.cookies_file)

        with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
            info = cast(dict[str, Any], ydl.extract_info(youtube_url, download=False))

        # Prefer the top-level URL; fall back to the last format entry.
        url = info.get("url")
        if not isinstance(url, str):
            url = ""

        formats = info.get("formats")
        if not url and isinstance(formats, list) and formats:
            last_fmt = formats[-1]
            if isinstance(last_fmt, dict):
                candidate = last_fmt.get("url")
                if isinstance(candidate, str):
                    url = candidate

        if not url:
            raise RuntimeError(
                "yt-dlp could not resolve a direct stream URL for this video.\n"
                "Try passing --cookies-from-browser safari (or chrome)."
            )
        return url

    # ──────────────────────────────────────────────────────────────────────────
    # Text-aware frame scanning
    # ──────────────────────────────────────────────────────────────────────────

    def _scan_turn_for_best_frame(
        self,
        playback_url: str,
        speaker_id: str,
        turn_start: float,
        turn_end: Optional[float],
        frame_dir: Path,
        is_remote: bool,
    ) -> FrameResult:
        """
        Two-pass text-aware scan across a speaker turn.

        Pass 1: extract frames at regular intervals across the turn window,
        pre-screen each for lower-third text.
        Pass 2: send the top-3 text-rich frames to Claude Vision.

        Returns the FrameResult with the highest-confidence name identification,
        or the first successfully extracted frame if Vision finds nothing.
        """
        # 1. Build candidate timestamp list
        if turn_end is not None:
            scan_end = min(turn_end, turn_start + self.scan_window_secs)
        else:
            scan_end = turn_start + self.scan_window_secs

        timestamps: list[float] = []
        ts = turn_start
        while ts < scan_end:
            timestamps.append(ts)
            ts += _SCAN_INTERVAL_SECS

        # 2. Apply remote frame cap
        if is_remote and len(timestamps) > self.max_scan_frames_remote:
            log.debug(f"  Remote source: capping scan at {self.max_scan_frames_remote} frames")
            timestamps = timestamps[:self.max_scan_frames_remote]

        # 3. Log scan plan
        log.info(f"  Scanning {len(timestamps)} frames for {speaker_id} ({turn_start:.1f}s – scan end)")

        # 4. Extract and pre-screen all frames
        prescreened: list[tuple[float, Path, int, str]] = []  # (ts, path, char_count, method)
        first_extracted: Optional[tuple[float, Path]] = None

        for scan_ts in timestamps:
            temp_path = frame_dir / f"{speaker_id}_scan_{scan_ts:.2f}.png"
            ok = self._extract_frame(playback_url, scan_ts, temp_path)
            if not ok:
                continue

            if first_extracted is None:
                first_extracted = (scan_ts, temp_path)

            passed, char_count, method = self._prescreen_frame(temp_path, is_remote)
            log.debug(f"    Prescreen {scan_ts:.2f}s: passed={passed} chars={char_count} method={method}")

            if passed:
                prescreened.append((scan_ts, temp_path, char_count, method))

        # 5. Fallback if no frames passed prescreening
        if not prescreened and first_extracted is not None:
            log.info(f"  No text-rich frames found for {speaker_id} — falling back to first extracted frame")
            fallback_ts, fallback_path = first_extracted
            prescreened = [(fallback_ts, fallback_path, 0, "none")]

        # 10. Handle total failure (no frames extracted at all)
        if not prescreened:
            return FrameResult(
                speaker_id=speaker_id, timestamp=turn_start,
                frame_path=None, raw_text=None,
                suggested_name=None, suggested_title=None,
                suggested_org=None, confidence=None,
                prescreen_method=None,
                error="All frame extractions failed during scan",
            )

        # 6. Sort by char_count descending, take top 3
        prescreened.sort(key=lambda x: x[2], reverse=True)
        candidates_to_try = prescreened[:3]

        # 7. Call Vision on candidates
        _conf_rank = {"high": 3, "medium": 2, "low": 1}
        vision_results: list[FrameResult] = []
        for cand_ts, cand_path, cand_chars, cand_method in candidates_to_try:
            if self.use_vision:
                result = self._analyse_frame(speaker_id, cand_ts, cand_path)
            else:
                result = FrameResult(
                    speaker_id=speaker_id, timestamp=cand_ts,
                    frame_path=cand_path, raw_text=None,
                    suggested_name=None, suggested_title=None,
                    suggested_org=None, confidence=None,
                )
            result.prescreen_method = cand_method
            vision_results.append(result)

        # 8. Pick the best result
        best = max(
            vision_results,
            key=lambda r: (
                _conf_rank.get(r.confidence or "", 0),
                len(r.raw_text or ""),
            ),
        )

        # 9. Clean up: rename winning frame, delete others
        winning_path = frame_dir / f"{speaker_id}_{best.timestamp:.2f}.png"
        if best.frame_path and best.frame_path.exists():
            best.frame_path.rename(winning_path)
            best.frame_path = winning_path

        # Delete all scan frames that are not the winner
        all_scan_paths = {path for _, path, _, _ in prescreened}
        for scan_path in all_scan_paths:
            if scan_path != winning_path and scan_path.exists():
                scan_path.unlink(missing_ok=True)

        # Also clean up any extracted frames that failed prescreening
        for scan_ts in timestamps:
            leftover = frame_dir / f"{speaker_id}_scan_{scan_ts:.2f}.png"
            if leftover != winning_path and leftover.exists():
                leftover.unlink(missing_ok=True)

        # 11. Log the outcome
        log.info(
            f"  Best frame for {speaker_id}: {best.timestamp:.1f}s  "
            f"confidence={best.confidence}  prescreen={best.prescreen_method}"
        )
        return best

    def _prescreen_frame(
        self, frame_path: Path, is_remote: bool
    ) -> tuple[bool, int, str]:
        """
        Pre-screen a frame for the presence of lower-third text.
        Returns (passed, char_count, method).
        """
        if not self.text_prescreen:
            return (True, 0, "none")

        # Path A — Tesseract OCR (local sources only)
        if self._tesseract_available and not is_remote:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(frame_path)
                w, h = img.size
                crop = img.crop((0, int(h * (1 - _BOTTOM_STRIP_FRACTION)), w, h))
                text = pytesseract.image_to_string(crop, config="--psm 6")
                text = text.strip()
                char_count = len(text)
                return (char_count >= _PRESCREEN_MIN_CHARS, char_count, "tesseract")
            except Exception as e:
                log.debug(f"    Tesseract prescreen failed ({e}), falling through to pixel heuristic")

        # Path B — Pixel contrast heuristic
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(frame_path)
            w, h = img.size
            crop = img.crop((0, int(h * (1 - _BOTTOM_STRIP_FRACTION)), w, h))
            gray = crop.convert("L")
            arr = np.array(gray)
            std = float(arr.std())
            return (std > _PIXEL_STD_THRESHOLD, int(std), "pixel_heuristic")
        except Exception as e:
            log.debug(f"    Pixel heuristic prescreen failed ({e}), passing frame through")
            return (True, 0, "none")

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

            text_parts: list[str] = []
            for block in response.content:
                if getattr(block, "type", None) != "text":
                    continue
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str) and block_text.strip():
                    text_parts.append(block_text.strip())

            raw_response = "\n".join(text_parts).strip()
            if not raw_response:
                raise ValueError("Claude response contained no text blocks")
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

    def _generate_waveform_image(
        self,
        audio_path: Path,
        speaker_id: str,
        turn_start: float,
        turn_end: float,
        session_id: str,
    ) -> Optional[Path]:
        """
        Extract the speaker's first-turn audio segment and save a waveform
        PNG to the same frames/ directory used for video captures.
        Returns the path on success, None on any error.
        """
        try:
            import librosa
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            frame_dir = self._frame_dir(session_id)
            out_path = frame_dir / f"{speaker_id}_waveform.png"

            y, _ = librosa.load(
                str(audio_path),
                sr=16000,
                offset=turn_start,
                duration=turn_end - turn_start,
            )

            fig, ax = plt.subplots(figsize=(4, 1.5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            ax.plot(np.linspace(0, turn_end - turn_start, len(y)), y, color="#00d4aa", linewidth=0.6)
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(str(out_path), dpi=80, bbox_inches="tight", facecolor="#1a1a2e")
            plt.close(fig)

            log.info(f"  Waveform saved: {out_path.name}")
            return out_path

        except Exception as exc:
            log.warning(f"  Waveform generation failed for {speaker_id}: {exc}")
            return None

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

def save_captures(
    captures: dict[str, FrameResult],
    session_id: str,
    output_dir: Path,
    source_url: str = "",
) -> Path:
    """
    Persist Vision results to  <output_dir>/<session_id>/captures.json
    so they can be linked back to frames and the diarization transcript.
    """
    out = {
        "session_id":  session_id,
        "source_url":  source_url,
        "captured_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "speakers": {
            sid: {
                "speaker_id":      cap.speaker_id,
                "timestamp":       cap.timestamp,
                "frame_path":      str(cap.frame_path) if cap.frame_path else None,
                "suggested_name":  cap.suggested_name,
                "suggested_title": cap.suggested_title,
                "suggested_org":   cap.suggested_org,
                "confidence":      cap.confidence,
                "raw_text":        cap.raw_text,
                "vision_used":     cap.vision_used,
                "identified":      cap.identified,
                "error":           cap.error,
                "prescreen_method": cap.prescreen_method,
            }
            for sid, cap in captures.items()
        },
    }

    captures_file = output_dir / session_id / "captures.json"
    captures_file.parent.mkdir(parents=True, exist_ok=True)
    captures_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"Captures saved → {captures_file}")
    return captures_file

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
        "--no-text-prescreen",
        action="store_true",
        help="Disable Tesseract/pixel pre-screening; send all extracted frames to Vision",
    )
    parser.add_argument(
        "--scan-window",
        type=float,
        default=60.0,
        help="Seconds of the turn to scan for lower-third text (default: 60)",
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
        scan_window_secs=args.scan_window,
        text_prescreen=not args.no_text_prescreen,
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
