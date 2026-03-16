"""
news_diarizer/main.py
─────────────────────
Command-line interface.

Commands
--------
  process           Diarise a local audio/video file and save JSON results (MP4 supported)
  process-youtube   Download audio from a YouTube URL and diarise it
  sessions          List all recorded sessions
  speakers          Query / list speakers in the catalogue
  add-speaker       Register a new named speaker
  update-speaker    Edit a speaker's metadata
  link              Bind an ephemeral SPEAKER_XX to a catalogue entry

Quick examples
--------------
  # Process a local audio file
  python main.py process audio/clip.wav --source "BBC Radio 4"

  # Process a local MP4 (audio extracted automatically; screen capture runs too)
  python main.py process output/mUV_p2IGLtc.mp4 --source "BBC News"

  # Process directly from YouTube
  python main.py process-youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

  # YouTube with speaker hints and a custom source label
  python main.py process-youtube "https://youtu.be/dQw4w9WgXcQ" \\
      --source "BBC News" --max-speakers 4

  # Just download without processing (inspect audio first)
  python main.py process-youtube "https://youtu.be/dQw4w9WgXcQ" --download-only

  # Process with speaker count hints (improves accuracy)
  python main.py process audio/panel.mp3 --source "ITV News" \\
      --min-speakers 2 --max-speakers 6

  # Skip Whisper / sentiment for a faster diarization-only run
  python main.py process audio/clip.wav --no-transcription --no-sentiment

  # Review the session list, then link speakers
  python main.py sessions
  python main.py add-speaker --name "Jane Smith" --affiliation "Reuters" \\
      --role "Political Correspondent"
  python main.py link --session clip_2024-03-11T14-00-00 \\
      --ephemeral SPEAKER_00 --catalogue SPK-0001

  # Query
  python main.py speakers --top 20
  python main.py speakers --search "Smith" --history
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (same directory as this file).
# Variables already present in the environment take precedence.
_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

if _ENV_FILE.exists():
    log.info(f"Loaded environment from {_ENV_FILE}")
else:
    log.warning(f".env file not found at {_ENV_FILE} — falling back to shell environment")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_audio_from_mp4(mp4_path: Path, wav_path: Path) -> None:
    """Extract a 16 kHz mono WAV from an MP4 file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr}")


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        sys.exit(
            "\nERROR: HF_TOKEN not set.\n\n"
            "Either add it to your .env file:\n"
            "  HF_TOKEN=hf_xxxxxxxxxxxx\n\n"
            "Or export it in your shell:\n"
            "  export HF_TOKEN=hf_xxxxxxxxxxxx\n\n"
            "You must also accept the pyannote model licences at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
        )
    return token


def _sentiment_label(score) -> str:
    if score is None:
        return "n/a"
    if score > 0.1:
        return f"\033[32m+{score:.2f} positive\033[0m"
    if score < -0.1:
        return f"\033[31m{score:.2f} negative\033[0m"
    return f"{score:.2f} neutral"


# ─── Sub-commands ─────────────────────────────────────────────────────────────

def cmd_process(args: argparse.Namespace) -> None:
    from diarizer import NewsDiarizer
    from catalogue import SpeakerCatalogue

    input_path = Path(args.audio_file)
    is_mp4 = input_path.suffix.lower() == ".mp4"

    # ── MP4: extract audio to a temporary WAV ─────────────────────────────
    _tmp_wav = None
    if is_mp4:
        _tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = Path(_tmp_wav.name)
        log.info(f"MP4 detected – extracting audio to {wav_path} …")
        _extract_audio_from_mp4(input_path, wav_path)
        audio_file = wav_path
    else:
        audio_file = input_path

    diarizer = NewsDiarizer(
        hf_token=_require_hf_token(),
        enable_transcription=not args.no_transcription,
        enable_sentiment=not args.no_sentiment,
        enable_ner=not args.no_ner,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        merge_gap_secs=args.merge_gap,
    )

    result = diarizer.process(
        audio_path=audio_file,
        source_name=args.source or input_path.stem,
        show_progress=True,
    )

    # Point source_file back at the original MP4, not the temp WAV
    if is_mp4:
        result.source_file = str(input_path)

    # Clean up temp WAV
    if _tmp_wav is not None:
        try:
            wav_path.unlink()
        except OSError:
            pass

    # Print summary ──────────────────────────────────────────────────────────
    w = 64
    print("\n" + "═" * w)
    print(f"  Source  : {result.source_name}")
    print(f"  File    : {Path(result.source_file).name}")
    print(f"  Length  : {result.total_duration:.1f}s")
    print(f"  Speakers: {result.num_speakers}")
    if args.merge_gap > 0:
        print(f"  Merged gap : {args.merge_gap}s")
    else:
        print(f"  Merged gap : disabled")
    print("─" * w)
    print(f"  {'Speaker':<14} {'Time':>8}  {'%Audio':>7}  {'Turns':>5}  Sentiment")
    print("─" * w)
    for sid, s in sorted(result.speaker_stats.items()):
        print(
            f"  {sid:<14} {s['total_speaking_time']:>7.1f}s"
            f"  {s['pct_of_audio']:>6.1f}%"
            f"  {s['turn_count']:>5}"
            f"  {_sentiment_label(s.get('avg_sentiment'))}"
        )
    if not args.no_ner and result.name_hints:
        print("─" * w)
        print("  NER hints:")
        for sid, names in sorted(result.name_hints.items()):
            if names:
                print(f"    {sid}  →  {', '.join(names)}")
    print("═" * w + "\n")

    # Save JSON ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{input_path.stem}_diarization.json"
    out_file.write_text(result.to_json())
    log.info(f"Results → {out_file}")

    # Record session ─────────────────────────────────────────────────────────
    cat = SpeakerCatalogue()
    session_id = cat.record_session(result)
    print(f"Session ID : {session_id}")
    print(
        "Next step  : review the JSON, identify speakers, then run:\n"
        f"  python main.py link --session {session_id} "
        "--ephemeral SPEAKER_00 --catalogue SPK-0001\n"
    )

    # ── Screen capture (MP4 only) ─────────────────────────────────────────
    if is_mp4 and not args.no_capture:
        from screen_capture import ScreenCapture, print_capture_summary, save_captures

        sc = ScreenCapture(
            output_dir=out_dir,
            use_vision=not args.no_vision,
            scan_window_secs=args.scan_window,
            max_scan_frames_remote=getattr(args, 'max_scan_frames', 20),
            text_prescreen=not args.no_text_prescreen,
        )
        captures = sc.capture_new_speakers(
            video_source=str(input_path),
            result=result,
            session_id=session_id,
        )
        print_capture_summary(captures)
        save_captures(captures, session_id, out_dir, source_url=str(input_path))


def cmd_process_youtube(args: argparse.Namespace) -> None:
    from youtube import YouTubeSource, source_name_from_meta, metadata_to_dict
    from diarizer import NewsDiarizer
    from catalogue import SpeakerCatalogue
    from screen_capture import ScreenCapture, print_capture_summary, save_captures

    # ── Step 1: fetch audio from YouTube ──────────────────────────────────
    yt = YouTubeSource(
        audio_dir=args.audio_dir,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies_file,
    )

    log.info(f"YouTube URL: {args.url}")
    meta, wav_path = yt.fetch(args.url)

    # ── Step 1b: optionally download full video ────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path: Path | None = None
    if args.save_video:
        video_path = out_dir / f"{wav_path.stem}.mp4"
        log.info("Downloading full video …")
        video_path = yt.download_video(args.url, video_path)

    # ── Step 1c: save metadata JSON ───────────────────────────────────────
    meta_file = out_dir / f"{wav_path.stem}_metadata.json"
    meta_file.write_text(
        json.dumps(metadata_to_dict(meta), indent=2, ensure_ascii=False)
    )
    log.info(f"Metadata   → {meta_file}")

    if args.download_only:
        print(f"\nAudio      : {wav_path}")
        if video_path:
            print(f"Video      : {video_path}")
        print(f"Metadata   : {meta_file}")
        print(f"Title      : {meta.title}")
        print(f"Channel    : {meta.channel}")
        print(f"Duration   : {meta.duration_seconds:.0f}s")
        print(f"Uploaded   : {meta.upload_date}")
        return

    # ── Step 2: diarise ───────────────────────────────────────────────────
    source_name = args.source or source_name_from_meta(meta)

    diarizer = NewsDiarizer(
        hf_token=_require_hf_token(),
        enable_transcription=not args.no_transcription,
        enable_sentiment=not args.no_sentiment,
        enable_ner=not args.no_ner,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        merge_gap_secs=args.merge_gap,
    )

    result = diarizer.process(
        audio_path=wav_path,
        source_name=source_name,
        show_progress=True,
    )

    # Attach YouTube metadata to the result's source_file for reference
    result.source_file = args.url  # store original URL rather than local path

    # ── Step 3: print summary ─────────────────────────────────────────────
    w = 64
    print("\n" + "═" * w)
    print(f"  Source  : {result.source_name}")
    print(f"  Title   : {meta.title}")
    print(f"  Channel : {meta.channel}")
    print(f"  URL     : {meta.url}")
    print(f"  Uploaded: {meta.upload_date}")
    print(f"  Length  : {result.total_duration:.1f}s")
    print(f"  Speakers: {result.num_speakers}")
    if args.merge_gap > 0:
        print(f"  Merged gap : {args.merge_gap}s")
    else:
        print(f"  Merged gap : disabled")
    if video_path:
        print(f"  Video   : {video_path}")
    print("─" * w)
    print(f"  {'Speaker':<14} {'Time':>8}  {'%Audio':>7}  {'Turns':>5}  Sentiment")
    print("─" * w)
    for sid, s in sorted(result.speaker_stats.items()):
        print(
            f"  {sid:<14} {s['total_speaking_time']:>7.1f}s"
            f"  {s['pct_of_audio']:>6.1f}%"
            f"  {s['turn_count']:>5}"
            f"  {_sentiment_label(s.get('avg_sentiment'))}"
        )
    if not args.no_ner and result.name_hints:
        print("─" * w)
        print("  NER hints:")
        for sid, names in sorted(result.name_hints.items()):
            if names:
                print(f"    {sid}  →  {', '.join(names)}")
    print("═" * w + "\n")

    # ── Step 4: save JSON ─────────────────────────────────────────────────
    out_file = out_dir / f"{wav_path.stem}_diarization.json"
    out_file.write_text(result.to_json())
    log.info(f"Results    → {out_file}")

    # ── Step 5: record session ────────────────────────────────────────────
    cat = SpeakerCatalogue()
    session_id = cat.record_session(result)
    print(f"Session ID : {session_id}")
    print(
        "Next step  : review the JSON, identify speakers, then run:\n"
        f"  python main.py link --session {session_id} "
        "--ephemeral SPEAKER_00 --catalogue SPK-0001\n"
    )

    # ── Step 6: screen capture + Vision ───────────────────────────────────
    if not args.no_capture:
        # Use local video file if already downloaded, otherwise stream from URL
        video_source = str(video_path) if video_path else args.url
        sc = ScreenCapture(
            output_dir=out_dir,
            use_vision=not args.no_vision,
            scan_window_secs=args.scan_window,
            max_scan_frames_remote=getattr(args, 'max_scan_frames', 20),
            text_prescreen=not args.no_text_prescreen,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
        )
        captures = sc.capture_new_speakers(
            video_source=video_source,
            result=result,
            session_id=session_id,
        )
        print_capture_summary(captures)
        save_captures(captures, session_id, out_dir, source_url=args.url)
        
def cmd_sessions(args: argparse.Namespace) -> None:
    from catalogue import SpeakerCatalogue
    rows = SpeakerCatalogue().list_sessions()
    if not rows:
        print("No sessions recorded yet.")
        return
    header = f"  {'Session ID':<48}  {'Source':<22}  {'Date':<12}  {'Dur':>7}  {'Spk':>4}"
    print("\n" + header)
    print("  " + "─" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['session_id']:<48}  {r['source_name'] or '':22}  "
            f"{r['processed_at'][:10]}  {r['total_duration']:>7.1f}s  {r['num_speakers']:>4}"
        )
    print()


def cmd_speakers(args: argparse.Namespace) -> None:
    from catalogue import SpeakerCatalogue
    cat = SpeakerCatalogue()

    if args.search or args.affiliation or args.role:
        profiles = cat.search_speakers(
            name=args.search, affiliation=args.affiliation, role=args.role
        )
        if not profiles:
            print("No speakers found.")
            return
        for p in profiles:
            print(f"\n  [{p.catalogue_id}]  {p.display_name or '(unnamed)'}")
            print(f"    Affiliation : {p.affiliation or '—'}")
            print(f"    Role        : {p.role or '—'}")
            print(f"    Appearances : {p.total_appearances}")
            print(f"    Total time  : {p.total_speaking_time:.1f}s")
            print(f"    Last seen   : {p.last_seen[:10]}")
            if args.history:
                for h in cat.get_appearances(p.catalogue_id):
                    print(
                        f"      · {h['appeared_at'][:10]}  "
                        f"{h['source_name'] or '':22}  "
                        f"{h['speaking_time']:>6.1f}s  {h['turn_count']} turns"
                    )
    else:
        rows = cat.top_speakers(limit=args.top)
        if not rows:
            print("Catalogue is empty.")
            return
        print(f"\n  {'ID':<10}  {'Name':<25}  {'Affiliation':<25}  {'Apps':>5}  {'Time':>9}")
        print("  " + "─" * 80)
        for r in rows:
            print(
                f"  {r['catalogue_id']:<10}  {r['display_name'] or '':25}  "
                f"{r['affiliation'] or '':25}  {r['total_appearances']:>5}  "
                f"{r['total_speaking_time']:>8.1f}s"
            )
    print()


def cmd_add_speaker(args: argparse.Namespace) -> None:
    from catalogue import SpeakerCatalogue
    cid = SpeakerCatalogue().add_speaker(
        display_name=args.name,
        affiliation=args.affiliation,
        role=args.role,
        notes=args.notes,
    )
    print(f"Speaker registered: {cid}")


def cmd_update_speaker(args: argparse.Namespace) -> None:
    from catalogue import SpeakerCatalogue
    SpeakerCatalogue().update_speaker(
        args.catalogue_id,
        display_name=args.name,
        affiliation=args.affiliation,
        role=args.role,
        notes=args.notes,
    )
    print(f"Updated: {args.catalogue_id}")


def cmd_link(args: argparse.Namespace) -> None:
    from catalogue import SpeakerCatalogue
    SpeakerCatalogue().link_appearance(
        catalogue_id=args.catalogue,
        session_id=args.session,
        ephemeral_id=args.ephemeral,
    )
    print(f"Linked {args.ephemeral} → {args.catalogue}  (session: {args.session})")


def cmd_review(args: argparse.Namespace) -> None:
    import socket
    import threading
    import time
    import webbrowser
    import urllib.request

    port = args.port

    def _port_open(p: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", p), timeout=0.5):
                return True
        except OSError:
            return False

    if not _port_open(port):
        try:
            import uvicorn
        except ImportError:
            sys.exit("uvicorn is not installed. Run: pip install uvicorn[standard]")

        log.info(f"Starting API server on port {port}…")
        t = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": "api:app",
                "host": "127.0.0.1",
                "port": port,
                "log_level": "warning",
            },
            daemon=True,
        )
        t.start()

        # Wait up to 3 s for the server to be ready
        deadline = time.time() + 3.0
        ready = False
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
                ready = True
                break
            except Exception:
                time.sleep(0.1)

        if not ready:
            log.warning("Server did not respond to health check in 3 s — opening browser anyway")
    else:
        log.info(f"Server already running on port {port}")

    url = f"http://localhost:{port}/review"
    if args.session_id:
        url += f"?session={args.session_id}"

    if not args.no_open:
        webbrowser.open(url)

    print(f"Review UI running at {url}  (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


# ─── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="news-diarizer",
        description="News source speaker diarization pipeline",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # process ────────────────────────────────────────────────────────────────
    pr = sub.add_parser("process", help="Diarise a local audio file")
    pr.add_argument("audio_file")
    pr.add_argument("--source", default="", help="Human label for the source")
    pr.add_argument("--output-dir", default="output")
    pr.add_argument("--num-speakers", type=int, default=None)
    pr.add_argument("--min-speakers", type=int, default=None)
    pr.add_argument("--max-speakers", type=int, default=None)
    pr.add_argument("--no-transcription", action="store_true")
    pr.add_argument("--no-sentiment", action="store_true")
    pr.add_argument("--no-capture", action="store_true",
                    help="Skip screenshot and Vision speaker identification (MP4 only)")
    pr.add_argument("--no-vision", action="store_true",
                    help="Take screenshots but skip the Claude Vision API call (MP4 only)")
    pr.add_argument("--no-ner", action="store_true",
                    help="Skip NER name extraction from transcripts")
    pr.add_argument("--scan-window", type=float, default=60.0,
                    help="Seconds of each speaker's first turn to scan for lower-third text (default: 60)")
    pr.add_argument("--no-text-prescreen", action="store_true",
                    help="Disable Tesseract/pixel pre-screening; send all extracted frames to Vision")
    pr.add_argument("--merge-gap", type=float, default=1.0,
                    help="Maximum silence gap in seconds between turns of the same speaker to merge (0 to disable, default: 1.0)")

    # process-youtube ────────────────────────────────────────────────────────
    yt = sub.add_parser("process-youtube", help="Download and diarise a YouTube video")
    yt.add_argument("url", help="YouTube video URL")
    yt.add_argument("--source", default="", help="Override source label (default: 'YouTube · <channel>')")
    yt.add_argument("--audio-dir", default="audio", help="Where to cache downloaded WAV files")
    yt.add_argument("--output-dir", default="output")
    yt.add_argument("--num-speakers", type=int, default=None)
    yt.add_argument("--min-speakers", type=int, default=None)
    yt.add_argument("--max-speakers", type=int, default=None)
    yt.add_argument("--no-transcription", action="store_true")
    yt.add_argument("--no-sentiment", action="store_true")
    yt.add_argument("--download-only", action="store_true",
                    help="Download audio (and video if --save-video) without running the diarization pipeline")
    yt.add_argument("--save-video", action="store_true",
                    help="Download and save the full MP4 video to the output directory for local reprocessing")
    yt.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help=(
            "Extract cookies from a browser to authenticate with YouTube. "
            "Fixes HTTP 403 errors. Accepted values: safari, chrome, firefox, "
            "edge, chromium, brave, opera, vivaldi. "
            "You must be logged in to YouTube in that browser."
        ),
    )
    yt.add_argument(
        "--cookies-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a Netscape-format cookies.txt file. "
            "Export one with: yt-dlp --cookies-from-browser safari "
            "--cookies cookies.txt --skip-download <url>"
        ),
    )
    yt.add_argument(
        "--no-capture",
        action="store_true",
        help="Skip screenshot and Vision speaker identification",
    )
    yt.add_argument(
        "--no-vision",
        action="store_true",
        help="Take screenshots but skip the Claude Vision API call",
    )
    yt.add_argument("--no-ner", action="store_true",
                    help="Skip NER name extraction from transcripts")
    yt.add_argument("--scan-window", type=float, default=60.0,
                    help="Seconds of each speaker's first turn to scan for lower-third text (default: 60)")
    yt.add_argument("--no-text-prescreen", action="store_true",
                    help="Disable Tesseract/pixel pre-screening; send all extracted frames to Vision")
    yt.add_argument("--max-scan-frames", type=int, default=20,
                    help="Maximum frames to extract per speaker for YouTube CDN sources (default: 20)")
    yt.add_argument("--merge-gap", type=float, default=1.0,
                    help="Maximum silence gap in seconds between turns of the same speaker to merge (0 to disable, default: 1.0)")
    # sessions ───────────────────────────────────────────────────────────────
    sub.add_parser("sessions", help="List recorded sessions")

    # speakers ───────────────────────────────────────────────────────────────
    sp = sub.add_parser("speakers", help="Query the speaker catalogue")
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--search", default=None, help="Filter by name substring")
    sp.add_argument("--affiliation", default=None)
    sp.add_argument("--role", default=None)
    sp.add_argument("--history", action="store_true", help="Show appearance history")

    # add-speaker ────────────────────────────────────────────────────────────
    ap = sub.add_parser("add-speaker", help="Register a new speaker")
    ap.add_argument("--name", default=None)
    ap.add_argument("--affiliation", default=None)
    ap.add_argument("--role", default=None)
    ap.add_argument("--notes", default=None)

    # update-speaker ─────────────────────────────────────────────────────────
    up = sub.add_parser("update-speaker", help="Edit a speaker's metadata")
    up.add_argument("catalogue_id")
    up.add_argument("--name", default=None)
    up.add_argument("--affiliation", default=None)
    up.add_argument("--role", default=None)
    up.add_argument("--notes", default=None)

    # link ───────────────────────────────────────────────────────────────────
    lk = sub.add_parser("link", help="Link an ephemeral ID to a catalogue entry")
    lk.add_argument("--session", required=True)
    lk.add_argument("--ephemeral", required=True, help="e.g. SPEAKER_00")
    lk.add_argument("--catalogue", required=True, help="e.g. SPK-0001")

    # review ─────────────────────────────────────────────────────────────────
    rv = sub.add_parser("review", help="Open the speaker review UI in a browser")
    rv.add_argument("session_id", nargs="?", default=None, help="Session ID to open (optional)")
    rv.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    rv.add_argument("--no-open", action="store_true", help="Start the server but do not open a browser tab")

    return p


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = build_parser().parse_args()
    {
        "process": cmd_process,
        "process-youtube": cmd_process_youtube,
        "sessions": cmd_sessions,
        "speakers": cmd_speakers,
        "add-speaker": cmd_add_speaker,
        "update-speaker": cmd_update_speaker,
        "link": cmd_link,
        "review": cmd_review,
    }[args.cmd](args)