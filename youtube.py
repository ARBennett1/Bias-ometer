"""
youtube.py
========================================================================
Downloads the best available audio track from a YouTube URL using yt-dlp
and converts it to a 16 kHz mono WAV ready for pyannote / Whisper.

Requires:
  pip install yt-dlp
  brew install ffmpeg   # yt-dlp uses ffmpeg for post-processing

YouTube 403 / SABR streaming errors
────────────────────────────────────
YouTube increasingly blocks unauthenticated yt-dlp requests with HTTP
403. The fix is to pass your browser's cookies so the request looks
authenticated.

Recommended approach — export cookies from your browser once:

    # Export from Safari (must be logged in to YouTube in that browser)
    yt-dlp --cookies-from-browser safari -x --audio-format wav \
    "https://www.youtube.com/watch?v=..." -o /tmp/test.wav

    # Or Chrome / Firefox / Edge
    yt-dlp --cookies-from-browser chrome ...

Then pass --cookies-from-browser to this application:

    python main.py process-youtube "https://..." \
        --cookies-from-browser safari

Or export a Netscape-format cookie file once and reuse it:

    yt-dlp --cookies-from-browser safari --cookies cookies.txt \
    --skip-download \
    "https://www.youtube.com/watch?v=..."

    python main.py process-youtube "https://..." --cookies cookies.txt

Usage (programmatic):
    from youtube import YouTubeSource
    yt = YouTubeSource(cookies_from_browser="safari")
    meta, wav = yt.fetch("https://www.youtube.com/watch?v=...")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

log = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """
    Structured metadata about a YouTube video.
    
    Fields are populated from yt-dlp's extracted info dict, which is
    quite comprehensive. We select a subset of the most relevant fields
    here.

    Attributes
    ----------
    video_id:
        Unique YouTube video ID (e.g. "dQw4w9WgXcQ").
    title:
        Video title.
    channel:
        Channel name (e.g. "Rick Astley").
    channel_url:
        URL of the channel.
    upload_date:
        Upload date in YYYYMMDD format (e.g. "20091025").
    duration_seconds:
        Video duration in seconds (e.g. 212.0).
    url:
        Original video URL.
    description:
        Video description text.
    thumbnail:
        URL of the video's thumbnail image.
    """
    video_id: str
    title: str
    channel: str
    channel_url: str
    upload_date: str           # YYYYMMDD
    duration_seconds: float
    url: str
    description: str = ""
    thumbnail: str = ""

class YouTubeSource:
    """
    Wrapper around yt-dlp that fetches metadata, downloads the best
    audio stream, and re-encodes it to 16 kHz mono WAV.

    Parameters
    ----------
    audio_dir: str | Path | None
        Directory for cached WAV files.
    cookies_from_browser: Optional[str]
        Use cookies from an installed browser to authenticate with
        YouTube. Accepts: "safari", "chrome", "firefox", "edge",
        "chromium", "brave", "opera", "vivaldi".
        Resolves most HTTP 403 / SABR streaming errors.
    cookies_file: Optional[str | Path]
        Path to a Netscape-format cookies.txt file.
        Alternative to cookies_from_browser.
    """

    def __init__(
        self,
        audio_dir: str | Path | None = None,
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str | Path] = None,
    ):
        self.audio_dir = Path(audio_dir) if audio_dir else Path(__file__).parent / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_from_browser = cookies_from_browser
        self.cookies_file = Path(cookies_file) if cookies_file else None
        self._check_yt_dlp()

    def download_video(self, url: str, out_path: Path) -> Path:
        """
        Download the best quality video+audio stream as MP4 to *out_path*.
        Re-uses the existing file if it already exists (cached).

        Parameters
        ----------
        url: str
            YouTube video URL.
        out_path: Path
            Destination path (should end in .mp4).

        Returns
        -------
        Path
            Path to the downloaded MP4 file.
        """
        import subprocess
        import shutil

        if out_path.exists():
            log.info(f"  Cached video found: {out_path.name} — skipping download")
            return out_path

        if not shutil.which("yt-dlp"):
            raise RuntimeError("yt-dlp binary not found on PATH.\nRun: pip install yt-dlp")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "yt-dlp",
            "--extractor-args", "youtube:player_client=android_vr",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--output", str(out_path),
        ]

        if self.cookies_from_browser:
            cmd += ["--cookies-from-browser", self.cookies_from_browser]
        elif self.cookies_file:
            cmd += ["--cookies", str(self.cookies_file)]

        cmd.append(url)
        log.debug(f"yt-dlp video command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, text=True, capture_output=False)
        except subprocess.CalledProcessError as exc:
            out_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"yt-dlp exited with code {exc.returncode}.\n"
                "Run the command above manually to see the full error output."
            ) from exc

        log.info(f"  Video saved: {out_path}")
        return out_path

    def fetch(
        self,
        url: str,
        filename_hint: Optional[str] = None,
    ) -> tuple[VideoMetadata, Path]:
        """
        Download audio from *url* and return (metadata, path_to_wav).
        Re-uses a cached WAV if the same video has already been
        downloaded.

        Parameters
        ----------
        url: str
            YouTube video URL.
        filename_hint: Optional[str]
            Optional hint for the output filename (without extension).
            Defaults to a safe version of the video ID.
        """
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

        log.info(f"Fetching metadata: {url}")
        meta = self._get_metadata(url)
        log.info(f"  Title   : {meta.title}")
        log.info(f"  Channel : {meta.channel}")
        log.info(f"  Length  : {meta.duration_seconds:.0f}s")

        stem = filename_hint or _safe_stem(meta.video_id)
        out_path = self.audio_dir / f"{stem}.wav"

        if out_path.exists():
            log.info(f"  Cached audio found: {out_path.name} — skipping download")
            return meta, out_path

        log.info("  Downloading audio …")
        self._download_audio(url, out_path)
        log.info(f"  Saved: {out_path}")
        return meta, out_path

    def fetch_metadata_only(self, url: str) -> VideoMetadata:
        """
        Return video metadata without downloading any audio.

        Parameters
        ----------
        url: str
            YouTube video URL.
        """
        return self._get_metadata(url)

    def _base_opts(self) -> dict:
        """
        Build the yt-dlp options dict that handles authentication.
        cookies_from_browser takes precedence over cookies_file.
        """
        opts: dict = {}
        if self.cookies_from_browser:
            opts["cookiesfrombrowser"] = (self.cookies_from_browser,)
            log.debug(f"  Using cookies from browser: {self.cookies_from_browser}")
        elif self.cookies_file:
            if not self.cookies_file.exists():
                raise FileNotFoundError(
                    f"Cookies file not found: {self.cookies_file}\n"
                    "Export one with:\n"
                    "  yt-dlp --cookies-from-browser safari --cookies cookies.txt "
                    "--skip-download 'https://www.youtube.com/watch?v=...'"
                )
            opts["cookiefile"] = str(self.cookies_file)
            log.debug(f"  Using cookies file: {self.cookies_file}")
        return opts

    def _get_metadata(self, url: str) -> VideoMetadata:
        """
        Fetch video metadata without downloading any audio.

        Parameters
        ----------
        url: str
            YouTube video URL.
        """
        import yt_dlp

        opts: dict[str, object] = {
            **self._base_opts(),
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
            # android_vr client bypasses the JS n-challenge without needing a JS runtime
            "extractor_args": {"youtube": {"player_client": ["android_vr"]}},
        }

        with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
            info = cast(dict[str, Any], ydl.extract_info(url, download=False))

        def _as_text(value: Any, default: str = "") -> str:
            return value if isinstance(value, str) else default

        return VideoMetadata(
            video_id=_as_text(info.get("id"), "unknown"),
            title=_as_text(info.get("title"), ""),
            channel=_as_text(info.get("uploader") or info.get("channel"), ""),
            channel_url=_as_text(info.get("uploader_url") or info.get("channel_url"), ""),
            upload_date=_as_text(info.get("upload_date"), ""),
            duration_seconds=float(info.get("duration") or 0),
            url=url,
            description=_as_text(info.get("description"), ""),
            thumbnail=_as_text(info.get("thumbnail"), ""),
        )

    def _download_audio(self, url: str, out_path: Path) -> None:
        """
        Download best audio and re-encode to 16 kHz mono WAV.

        Parameters
        ----------
        url: str
            YouTube video URL.
        out_path: Path
            Path to save the downloaded audio.
        """
        import subprocess
        import shutil

        if not shutil.which("yt-dlp"):
            raise RuntimeError(
                "yt-dlp binary not found on PATH.\n"
                "Run: pip install yt-dlp"
            )

        tmp_base = self.audio_dir / (out_path.stem + ".tmp")

        cmd = [
            "yt-dlp",
            # android_vr client bypasses the JS n-challenge without needing a JS runtime
            "--extractor-args", "youtube:player_client=android_vr",
            # Format: prefer direct HTTPS stream (format 18), fall back to best audio
            "--format", "bestaudio/best",
            # Output template — yt-dlp will append the codec extension
            "--output", str(tmp_base) + ".%(ext)s",
            # Extract audio and convert to WAV at 16 kHz mono
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        ]

        # Cookie authentication
        if self.cookies_from_browser:
            cmd += ["--cookies-from-browser", self.cookies_from_browser]
        elif self.cookies_file:
            cmd += ["--cookies", str(self.cookies_file)]

        cmd.append(url)

        log.debug(f"yt-dlp command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=False,   # let stdout/stderr stream to terminal
            )
        except subprocess.CalledProcessError as exc:
            # Clean up any partial files
            for f in self.audio_dir.glob(f"{tmp_base.name}*"):
                f.unlink(missing_ok=True)
            raise RuntimeError(
                f"yt-dlp exited with code {exc.returncode}.\n"
                "Run the command above manually to see the full error output."
            ) from exc

        # Locate the output file yt-dlp wrote and rename to final path
        candidates = sorted(self.audio_dir.glob(f"{tmp_base.name}*.wav"))
        if not candidates:
            raise RuntimeError(
                "yt-dlp reported success but no WAV file was found.\n"
                f"Searched: {self.audio_dir / (tmp_base.name + '*.wav')}"
            )
        candidates[0].rename(out_path)

    @staticmethod
    def _check_yt_dlp() -> None:
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            log.warning("yt-dlp is not installed. Run: pip install yt-dlp")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_stem(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", text)[:80]


def source_name_from_meta(meta: VideoMetadata) -> str:
    if meta.channel:
        return f"YouTube · {meta.channel}"
    return f"YouTube · {meta.video_id}"


def metadata_to_dict(meta: VideoMetadata) -> dict:
    """Return VideoMetadata as a plain dict suitable for JSON serialisation."""
    import dataclasses
    return dataclasses.asdict(meta)


def _raise_friendly(exc: Exception) -> None:
    """Re-raise yt-dlp errors with actionable advice."""
    msg = str(exc)
    if "403" in msg or "Forbidden" in msg or "SABR" in msg.upper():
        raise RuntimeError(
            "YouTube blocked the download (HTTP 403).\n\n"
            "Fix: pass your browser cookies so the request looks authenticated.\n\n"
            "Option 1 — browser cookies (simplest, must be logged in to YouTube):\n"
            "  python main.py process-youtube <url> --cookies-from-browser safari\n"
            "  python main.py process-youtube <url> --cookies-from-browser chrome\n\n"
            "Option 2 — export a cookies file once, reuse it:\n"
            "  yt-dlp --cookies-from-browser safari --cookies cookies.txt \\\n"
            "      --skip-download 'https://www.youtube.com/watch?v=...'\n"
            "  python main.py process-youtube <url> --cookies cookies.txt\n\n"
            "Option 3 — update yt-dlp (may fix without cookies for some videos):\n"
            "  pip install -U yt-dlp\n"
        ) from exc
    raise exc