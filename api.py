"""
api.py
========================================================================
FastAPI REST wrapper around NewsDiarizer + SpeakerCatalogue.

Start the server:
    pip install fastapi uvicorn[standard] python-multipart
    uvicorn api:app --reload --port 8000

The React frontend expects this at http://localhost:8000
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, cast

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Env / path setup ─────────────────────────────────────────────────────────

_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE, override=False)
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="News Diarizer API",
    description="Speaker diarization, transcription and sentiment for news audio.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output-files", StaticFiles(directory="output", check_dir=False), name="output-files")


# ── In-memory job store ────────────────────────────────────────────────────────

@dataclass
class Job:
    job_id: str
    status: str           # queued | running | complete | error
    source_type: str      # youtube | file
    source_ref: str       # URL or original filename
    source_name: str = ""
    created_at: str = ""
    completed_at: str = ""
    # Flat progress string (legacy / fallback)
    progress: str = ""
    # Rich progress fields polled by the frontend
    progress_pct: int = 0
    progress_stage: str = "queued"  # queued|download|models|diarize|transcribe|sentiment|saving|done
    progress_detail: str = ""       # e.g. "Turn 12 / 45" or "Downloaded 67%"
    # Results
    session_id: str = ""
    result: Optional[dict] = None
    error: str = ""
    # Config snapshot
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    enable_transcription: bool = True
    enable_sentiment: bool = True
    enable_ner: bool = True
    merge_gap_secs: float = 1.0
    broadcast_channel: str = ""
    broadcast_date: str = ""
    save_video: bool = False
    video_path: Optional[str] = None  # set after successful video download
    vision_scan_window: float = 60.0
    vision_max_scan_frames: int = 20
    vision_text_prescreen: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_hf_token() -> str:
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail=(
                "HF_TOKEN not set. Add it to your .env file:\n"
                "  HF_TOKEN=hf_xxxxxxxxxxxx\n"
                "Then restart the server."
            ),
        )
    return HF_TOKEN


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)


# ── Custom pyannote progress hook ─────────────────────────────────────────────

class _PyannoteProgressHook:
    """
    Drop-in replacement for pyannote's ProgressHook that fires a callback
    instead of printing to the terminal.

    Pyannote calls: hook(step_name, artifact, completed=n, total=m)
    We map the completed/total fraction onto a 0-100 pct and forward it.
    """

    def __init__(self, on_progress: Callable[[int, str], None]):
        self._on_progress = on_progress   # (pct_0_100, detail_str)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __call__(
        self,
        step_name: str,
        step_artifact: Any = None,
        *,
        completed: Optional[float] = None,
        total: Optional[float] = None,
        **kwargs,          # absorb any extra args pyannote passes (e.g. 'file')
    ) -> None:
        if completed is not None and total and total > 0:
            frac = min(completed / total, 1.0)
            detail = f"{step_name.replace('_', ' ').title()}  {int(completed)}/{int(total)}"
            self._on_progress(int(frac * 100), detail)


# ── Progress-aware diarizer wrapper ───────────────────────────────────────────

class _ProgressDiarizer:
    """
    Wraps NewsDiarizer and injects live progress callbacks at every pipeline
    stage. We hold an inner NewsDiarizer instance and re-implement process(),
    _transcribe(), and _run_sentiment() with per-item callbacks.

    Percentage bands are passed in so the caller can shift them (e.g. YouTube
    jobs reserve 0-30% for the download, so the pipeline starts at 30%).
    """

    def __init__(
        self,
        *,
        hf_token: str,
        enable_transcription: bool,
        enable_sentiment: bool,
        enable_ner: bool = True,
        num_speakers: Optional[int],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
        merge_gap_secs: float,
        on_progress: Callable[[int, str, str], None],  # (pct, stage, detail)
        model_band: tuple[int, int],
        diarize_band: tuple[int, int],
        transcribe_band: tuple[int, int],
        sentiment_band: tuple[int, int],
        ner_band: tuple[int, int],
        saving_band: tuple[int, int],
    ):
        self._on_progress = on_progress
        self._diarize_band = diarize_band
        self._transcribe_band = transcribe_band
        self._sentiment_band = sentiment_band
        self._ner_band = ner_band
        self._saving_band = saving_band

        # Load models ─────────────────────────────────────────────────────────
        self._emit(model_band[0], "models", "Loading pyannote…")
        from diarizer import NewsDiarizer
        self._inner = NewsDiarizer(
            hf_token=hf_token,
            enable_transcription=enable_transcription,
            enable_sentiment=enable_sentiment,
            enable_ner=enable_ner,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            merge_gap_secs=merge_gap_secs,
        )
        self._emit(model_band[1], "models", "Models ready")

    # ── Emit helper ──────────────────────────────────────────────────────────

    def _emit(self, pct: int, stage: str, detail: str) -> None:
        self._on_progress(pct, stage, detail)

    # ── Pyannote diarize hook ─────────────────────────────────────────────────

    def _diarize_hook(self, frac_pct: int, detail: str) -> None:
        lo, hi = self._diarize_band
        mapped = lo + int((frac_pct / 100) * (hi - lo))
        self._emit(mapped, "diarize", detail)

    # ── Public: process ───────────────────────────────────────────────────────

    def process(self, audio_path: Path, source_name: str) -> Any:
        import torchaudio
        from datetime import datetime, timezone
        from diarizer import Turn, DiarizationResult, _speaker_stats

        inner = self._inner
        audio_path = Path(audio_path)

        # Load audio ──────────────────────────────────────────────────────────
        self._emit(self._diarize_band[0], "diarize", "Loading audio file…")
        waveform, sample_rate = torchaudio.load(str(audio_path))
        total_duration = waveform.shape[-1] / sample_rate
        log.info(f"  {total_duration:.1f}s · {sample_rate} Hz")

        # Diarize ─────────────────────────────────────────────────────────────
        diar_kwargs: dict = {}
        if inner.num_speakers is not None:
            diar_kwargs["num_speakers"] = inner.num_speakers
        else:
            if inner.min_speakers is not None:
                diar_kwargs["min_speakers"] = inner.min_speakers
            if inner.max_speakers is not None:
                diar_kwargs["max_speakers"] = inner.max_speakers

        audio_in = {"waveform": waveform, "sample_rate": sample_rate}

        with _PyannoteProgressHook(self._diarize_hook) as hook:
            output = inner._diar(audio_in, hook=hook, **diar_kwargs)

        annotation = getattr(output, "speaker_diarization", output)
        turns: list[Turn] = [
            Turn(
                speaker_id=spk,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                duration=round(seg.end - seg.start, 3),
            )
            for seg, _, spk in annotation.itertracks(yield_label=True)
        ]
        unique_speakers = sorted({t.speaker_id for t in turns})
        self._emit(
            self._diarize_band[1], "diarize",
            f"{len(unique_speakers)} speakers · {len(turns)} turns",
        )
        log.info(f"  {len(unique_speakers)} speakers · {len(turns)} turns")

        # Transcribe ──────────────────────────────────────────────────────────
        if inner.enable_transcription and inner._whisper:
            turns = self._transcribe(waveform, sample_rate, turns)

        # Sentiment ───────────────────────────────────────────────────────────
        if inner.enable_sentiment and inner._sentiment_pipe:
            turns = self._sentiment(turns)

        # Merge contiguous same-speaker turns ─────────────────────────────────
        pre_merge_turns = list(turns)
        turns = inner._merge_turns(turns)

        # NER name hints ───────────────────────────────────────────────────────
        self._emit(self._ner_band[0], "ner", "Extracting name hints…")
        name_hints = inner._extract_name_hints(turns) if inner.enable_ner else {}
        self._emit(self._ner_band[1], "ner", "Name hints complete")

        # Build result ────────────────────────────────────────────────────────
        self._emit(self._saving_band[0], "saving", "Building result…")
        stats = _speaker_stats(turns, total_duration)
        return DiarizationResult(
            source_file=str(audio_path),
            source_name=source_name,
            processed_at=datetime.now(timezone.utc).isoformat(),
            total_duration=round(total_duration, 3),
            num_speakers=len(unique_speakers),
            turns=turns,
            speaker_stats=stats,
            original_turns=pre_merge_turns if inner.merge_gap_secs > 0.0 else [],
            merge_gap_secs=inner.merge_gap_secs,
            name_hints=name_hints,
        )

    # ── Transcription with per-turn progress ──────────────────────────────────

    def _transcribe(self, waveform, sample_rate, turns):
        import numpy as np
        import torchaudio

        lo, hi = self._transcribe_band
        whisper = self._inner._whisper
        if whisper is None:
            return turns

        total = len(turns)

        resampler = (
            torchaudio.transforms.Resample(sample_rate, 16_000)
            if sample_rate != 16_000 else None
        )

        for i, turn in enumerate(turns):
            s = int(turn.start * sample_rate)
            e = int(turn.end * sample_rate)
            seg = waveform[:, s:e]
            if resampler:
                seg = resampler(seg)

            audio_np = seg.mean(0).numpy().astype(np.float32)
            if len(audio_np) < 8_000:
                turn.transcript = ""
            else:
                out = whisper.transcribe(audio_np, fp16=False, language="en")
                text = out.get("text", "") if isinstance(out, dict) else ""
                turn.transcript = str(text).strip()

            pct = lo + int(((i + 1) / total) * (hi - lo))
            self._emit(pct, "transcribe", f"Turn {i + 1} / {total}")

        return turns

    # ── Sentiment with per-turn progress ─────────────────────────────────────

    def _sentiment(self, turns):
        lo, hi = self._sentiment_band
        pipe = self._inner._sentiment_pipe
        if pipe is None:
            return turns

        total = len(turns)

        for i, turn in enumerate(turns):
            text = turn.transcript or ""
            if not text:
                turn.sentiment, turn.sentiment_score = "neutral", 0.0
            else:
                try:
                    r = pipe(text)[0]
                    label, conf = r["label"].lower(), float(r["score"])
                    turn.sentiment = label
                    turn.sentiment_score = round(conf if label == "positive" else -conf, 4)
                except Exception:
                    turn.sentiment, turn.sentiment_score = "neutral", 0.0

            pct = lo + int(((i + 1) / total) * (hi - lo))
            self._emit(pct, "sentiment", f"Turn {i + 1} / {total}")

        return turns


# ── YouTube download with yt-dlp progress hook ────────────────────────────────

def _download_youtube_with_progress(
    url: str,
    job_id: str,
    download_band: tuple[int, int],
) -> tuple[Any, Path]:
    """
    Download YouTube audio via the yt-dlp Python API, using its progress_hooks
    to stream download percentage back into the job store.
    Returns (VideoMetadata, wav_path).
    """
    from youtube import VideoMetadata, _safe_stem

    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    audio_dir = Path("audio")
    audio_dir.mkdir(exist_ok=True)
    lo, hi = download_band

    # ── yt-dlp progress hook ─────────────────────────────────────────────────
    def _yt_hook(d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            downloaded = d.get("downloaded_bytes") or 0
            total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            speed = d.get("speed") or 0
            speed_str = f"{speed / 1024 / 1024:.1f} MB/s" if speed else ""
            if total_bytes:
                frac = min(downloaded / total_bytes, 1.0)
                pct = lo + int(frac * (hi - lo))
                detail = f"Downloaded {int(frac * 100)}%  {speed_str}".strip()
            else:
                pct = lo + (hi - lo) // 2
                detail = f"Downloading…  {speed_str}".strip()
            _update_job(job_id,
                        progress_pct=pct,
                        progress_stage="download",
                        progress_detail=detail,
                        progress=detail)
        elif status == "finished":
            _update_job(job_id,
                        progress_pct=hi,
                        progress_stage="download",
                        progress_detail="Converting to WAV…",
                        progress="Converting to WAV…")

    base_opts: dict = {"quiet": True, "no_warnings": True}

    # ── Fetch metadata ────────────────────────────────────────────────────────
    _update_job(job_id,
                progress_pct=lo,
                progress_stage="download",
                progress_detail="Fetching video metadata…",
                progress="Fetching metadata…")

    metadata_opts: dict[str, object] = {
        **base_opts,
        "skip_download": True,
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(cast(Any, metadata_opts)) as ydl:
        info = cast(dict[str, Any], ydl.extract_info(url, download=False))

    def _s(v, default=""):
        return v if isinstance(v, str) else default

    meta = VideoMetadata(
        video_id=_s(info.get("id"), "unknown"),
        title=_s(info.get("title")),
        channel=_s(info.get("uploader") or info.get("channel")),
        channel_url=_s(info.get("uploader_url") or info.get("channel_url")),
        upload_date=_s(info.get("upload_date")),
        duration_seconds=float(info.get("duration") or 0),
        url=url,
        description=_s(info.get("description")),
        thumbnail=_s(info.get("thumbnail")),
    )
    log.info(f"  Title   : {meta.title}")
    log.info(f"  Channel : {meta.channel}")
    log.info(f"  Length  : {meta.duration_seconds:.0f}s")

    # ── Check cache ───────────────────────────────────────────────────────────
    stem = _safe_stem(meta.video_id)
    out_path = audio_dir / f"{stem}.wav"
    if out_path.exists():
        log.info(f"  Cached audio: {out_path.name}")
        _update_job(job_id,
                    progress_pct=hi,
                    progress_stage="download",
                    progress_detail="Using cached audio",
                    progress="Using cached audio")
        return meta, out_path

    # ── Download + convert ────────────────────────────────────────────────────
    tmp_base = audio_dir / f"{stem}.tmp"
    dl_opts: dict[str, object] = {
        **base_opts,
        "format": "18/bestaudio",
        "outtmpl": str(tmp_base) + ".%(ext)s",
        "progress_hooks": [_yt_hook],
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"ffmpeg": ["-ar", "16000", "-ac", "1"]},
    }

    with yt_dlp.YoutubeDL(cast(Any, dl_opts)) as ydl:
        ydl.download([url])

    candidates = sorted(audio_dir.glob(f"{tmp_base.name}*.wav"))
    if not candidates:
        raise RuntimeError("yt-dlp completed but no WAV file was found.")
    candidates[0].rename(out_path)

    return meta, out_path


# ── Background workers ────────────────────────────────────────────────────────

def _make_progress_cb(job_id: str) -> Callable[[int, str, str], None]:
    def _cb(pct: int, stage: str, detail: str) -> None:
        _update_job(job_id,
                    progress_pct=pct,
                    progress_stage=stage,
                    progress_detail=detail,
                    progress=detail)
    return _cb


def _run_diarization(
    job_id: str,
    wav_path: Path,
    *,
    model_band: tuple[int, int] = (0, 5),
    diarize_band: tuple[int, int] = (5, 55),
    transcribe_band: tuple[int, int] = (55, 80),
    sentiment_band: tuple[int, int] = (80, 92),
    ner_band: tuple[int, int] = (92, 95),
    saving_band: tuple[int, int] = (95, 100),
) -> None:
    try:
        from catalogue import SpeakerCatalogue

        job = _jobs[job_id]
        _update_job(job_id, status="running")

        diarizer = _ProgressDiarizer(
            hf_token=HF_TOKEN,
            enable_transcription=job.enable_transcription,
            enable_sentiment=job.enable_sentiment,
            enable_ner=job.enable_ner,
            num_speakers=job.num_speakers,
            min_speakers=job.min_speakers,
            max_speakers=job.max_speakers,
            merge_gap_secs=job.merge_gap_secs,
            on_progress=_make_progress_cb(job_id),
            model_band=model_band,
            diarize_band=diarize_band,
            transcribe_band=transcribe_band,
            sentiment_band=sentiment_band,
            ner_band=ner_band,
            saving_band=saving_band,
        )

        result = diarizer.process(
            audio_path=wav_path,
            source_name=job.source_name or wav_path.stem,
        )

        # For YouTube jobs, source_file must be a usable video reference (URL or
        # local path) so the review screen-capture can extract frames later.
        if job.source_type == "youtube":
            if job.video_path and Path(job.video_path).exists():
                result.source_file = job.video_path
            else:
                result.source_file = job.source_ref  # original YouTube URL

        _update_job(job_id, progress_pct=saving_band[0], progress_stage="saving",
                    progress_detail="Writing to catalogue…", progress="Saving…")
        cat = SpeakerCatalogue()
        session_id = cat.record_session(result)

        # Write diarization JSON to output/ (mirrors what the CLI does)
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        json_path = out_dir / f"{session_id}_diarization.json"
        try:
            json_path.write_text(json.dumps(result.to_dict(), indent=2))
            log.info(f"Diarization JSON written: {json_path}")
        except Exception as je:
            log.warning(f"Could not write diarization JSON: {je}")

        job = _jobs[job_id]
        if job.broadcast_channel or job.broadcast_date:
            cat.update_session_meta(
                session_id,
                source_name=job.source_name or None,
                broadcast_date=job.broadcast_date or None,
                broadcast_channel=job.broadcast_channel or None,
            )

        _update_job(
            job_id,
            status="complete",
            progress_pct=100,
            progress_stage="done",
            progress_detail="",
            progress="Done",
            session_id=session_id,
            result=result.to_dict(),
            completed_at=_now(),
        )
        log.info(f"Job {job_id} complete — session {session_id}")

    except Exception as exc:
        log.exception(f"Job {job_id} failed")
        _update_job(job_id, status="error", error=str(exc), completed_at=_now())


def _youtube_worker(job_id: str, url: str) -> None:
    try:
        job = _jobs[job_id]
        _update_job(job_id, status="running")

        # Download occupies 0-30%; pipeline stages share the remaining 70%
        meta, wav_path = _download_youtube_with_progress(
            url, job_id, download_band=(0, 30),
        )

        if not job.source_name:
            channel = meta.channel or meta.video_id
            _update_job(job_id, source_name=f"YouTube · {channel}")

        if job.save_video:
            try:
                from youtube import _safe_stem
                video_dir = Path("videos")
                video_dir.mkdir(exist_ok=True)
                stem = _safe_stem(meta.video_id)
                video_path = video_dir / f"{stem}.mp4"
                if not video_path.exists():
                    _update_job(job_id, progress_detail="Saving video…", progress="Saving video…")
                    import yt_dlp
                    vdl_opts: dict = {
                        "quiet": True, "no_warnings": True,
                        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "outtmpl": str(video_path),
                        "merge_output_format": "mp4",
                    }
                    with yt_dlp.YoutubeDL(vdl_opts) as ydl:
                        ydl.download([url])
                    log.info(f"Video saved: {video_path}")
                if video_path.exists():
                    _update_job(job_id, video_path=str(video_path))
            except Exception as ve:
                log.warning(f"Video save failed (non-fatal): {ve}")

        _run_diarization(
            job_id, wav_path,
            model_band=(30, 35),
            diarize_band=(35, 65),
            transcribe_band=(65, 82),
            sentiment_band=(82, 93),
            ner_band=(93, 96),
            saving_band=(96, 100),
        )

    except Exception as exc:
        log.exception(f"YouTube job {job_id} failed")
        _update_job(job_id, status="error", error=str(exc), completed_at=_now())


# ── Jobs endpoints ────────────────────────────────────────────────────────────

@app.post("/jobs", summary="Submit a new processing job")
async def submit_job(
    background_tasks: BackgroundTasks,
    url: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    source_name: str = Form(""),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    enable_transcription: bool = Form(True),
    enable_sentiment: bool = Form(True),
    enable_ner: bool = Form(True),
    merge_gap_secs: float = Form(1.0),
    broadcast_channel: str = Form(""),
    broadcast_date: str = Form(""),
    save_video: bool = Form(False),
    vision_scan_window: float = Form(60.0),
    vision_max_scan_frames: int = Form(20),
    vision_text_prescreen: bool = Form(True),
):
    _require_hf_token()

    if not url and not audio_file:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'audio_file'.")

    job_id = str(uuid.uuid4())[:8]
    now = _now()

    if url:
        job = Job(
            job_id=job_id, status="queued", source_type="youtube",
            source_ref=url, source_name=source_name, created_at=now,
            num_speakers=num_speakers, min_speakers=min_speakers,
            max_speakers=max_speakers,
            enable_transcription=enable_transcription,
            enable_sentiment=enable_sentiment,
            enable_ner=enable_ner,
            merge_gap_secs=merge_gap_secs,
            broadcast_channel=broadcast_channel,
            broadcast_date=broadcast_date,
            save_video=save_video,
            vision_scan_window=vision_scan_window,
            vision_max_scan_frames=vision_max_scan_frames,
            vision_text_prescreen=vision_text_prescreen,
        )
        with _jobs_lock:
            _jobs[job_id] = job
        threading.Thread(target=_youtube_worker, args=(job_id, url), daemon=True).start()

    else:
        uploaded = audio_file
        if uploaded is None:
            raise HTTPException(status_code=400, detail="Provide 'audio_file' when 'url' is empty.")

        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        suffix = Path(uploaded.filename).suffix if uploaded.filename else ".wav"
        wav_path = upload_dir / f"{job_id}{suffix}"
        wav_path.write_bytes(await uploaded.read())

        label = source_name or Path(uploaded.filename or "").stem or job_id
        job = Job(
            job_id=job_id, status="queued", source_type="file",
            source_ref=uploaded.filename or "upload",
            source_name=label, created_at=now,
            num_speakers=num_speakers, min_speakers=min_speakers,
            max_speakers=max_speakers,
            enable_transcription=enable_transcription,
            enable_sentiment=enable_sentiment,
            enable_ner=enable_ner,
            merge_gap_secs=merge_gap_secs,
            broadcast_channel=broadcast_channel,
            broadcast_date=broadcast_date,
            save_video=save_video,
            vision_scan_window=vision_scan_window,
            vision_max_scan_frames=vision_max_scan_frames,
            vision_text_prescreen=vision_text_prescreen,
        )
        with _jobs_lock:
            _jobs[job_id] = job
        threading.Thread(target=_run_diarization, args=(job_id, wav_path), daemon=True).start()

    log.info(f"Job {job_id} queued ({job.source_type})")
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs", summary="List recent jobs")
def list_jobs():
    with _jobs_lock:
        return [j.to_dict() for j in reversed(list(_jobs.values()))]


@app.get("/jobs/{job_id}", summary="Poll job status")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.get("/sessions", summary="List recorded sessions")
def list_sessions():
    from catalogue import SpeakerCatalogue
    return SpeakerCatalogue().list_sessions()


@app.get("/sessions/{session_id}", summary="Get full session detail")
def get_session(session_id: str):
    import json, sqlite3
    from catalogue import SpeakerCatalogue

    cat = SpeakerCatalogue()
    rows = cat.list_sessions()
    match = next((r for r in rows if r["session_id"] == session_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Session not found")

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if row and row["result_json"]:
        match["result"] = json.loads(row["result_json"])

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        links = cx.execute(
            """SELECT a.ephemeral_id, a.catalogue_id, s.display_name, s.affiliation, s.role
               FROM appearances a
               LEFT JOIN speakers s ON a.catalogue_id = s.catalogue_id
               WHERE a.session_id = ?""",
            (session_id,),
        ).fetchall()
    match["links"] = [dict(r) for r in links]
    return match


@app.patch("/sessions/{session_id}", summary="Update session metadata")
def update_session(session_id: str, body: dict):
    from catalogue import SpeakerCatalogue
    cat = SpeakerCatalogue()
    rows = cat.list_sessions()
    if not any(r["session_id"] == session_id for r in rows):
        raise HTTPException(status_code=404, detail="Session not found")
    cat.update_session_meta(
        session_id,
        source_name=body.get("source_name"),
        broadcast_date=body.get("broadcast_date"),
        broadcast_channel=body.get("broadcast_channel"),
    )
    return {"ok": True}


# ── Speakers ───────────────────────────────────────────────────────────────────

@app.get("/speakers", summary="List or search the speaker catalogue")
def get_speakers(
    search: Optional[str] = None,
    affiliation: Optional[str] = None,
    role: Optional[str] = None,
    top: int = 50,
):
    from catalogue import SpeakerCatalogue
    cat = SpeakerCatalogue()
    if search or affiliation or role:
        profiles = cat.search_speakers(name=search, affiliation=affiliation, role=role)
        return [
            {"catalogue_id": p.catalogue_id, "display_name": p.display_name,
             "affiliation": p.affiliation, "role": p.role, "notes": p.notes,
             "first_seen": p.first_seen, "last_seen": p.last_seen,
             "total_appearances": p.total_appearances,
             "total_speaking_time": p.total_speaking_time}
            for p in profiles
        ]
    return cat.top_speakers(limit=top)


@app.post("/speakers", summary="Register a new speaker")
def add_speaker(
    name: Optional[str] = Form(None),
    affiliation: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    from catalogue import SpeakerCatalogue
    cid = SpeakerCatalogue().add_speaker(
        display_name=name, affiliation=affiliation, role=role, notes=notes,
    )
    return {"catalogue_id": cid}


@app.put("/speakers/{catalogue_id}", summary="Update speaker metadata")
def update_speaker(
    catalogue_id: str,
    name: Optional[str] = Form(None),
    affiliation: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    from catalogue import SpeakerCatalogue
    SpeakerCatalogue().update_speaker(
        catalogue_id, display_name=name, affiliation=affiliation,
        role=role, notes=notes,
    )
    return {"status": "updated", "catalogue_id": catalogue_id}


@app.get("/speakers/{catalogue_id}/appearances", summary="Get appearance history")
def get_appearances(catalogue_id: str):
    from catalogue import SpeakerCatalogue
    cat = SpeakerCatalogue()
    profile = cat.get_speaker(catalogue_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {
        "profile": {
            "catalogue_id": profile.catalogue_id,
            "display_name": profile.display_name,
            "affiliation": profile.affiliation,
            "role": profile.role,
            "total_appearances": profile.total_appearances,
            "total_speaking_time": profile.total_speaking_time,
            "first_seen": profile.first_seen,
            "last_seen": profile.last_seen,
        },
        "appearances": cat.get_appearances(catalogue_id),
    }


# ── Link ephemeral → catalogue ─────────────────────────────────────────────────

@app.post("/link", summary="Link an ephemeral speaker ID to a catalogue entry")
def link_speaker(
    session_id: str = Form(...),
    ephemeral_id: str = Form(...),
    catalogue_id: str = Form(...),
):
    from catalogue import SpeakerCatalogue
    cat = SpeakerCatalogue()
    if not cat.get_speaker(catalogue_id):
        raise HTTPException(
            status_code=404,
            detail=f"Catalogue ID {catalogue_id} not found. Create the speaker first.",
        )
    cat.link_appearance(
        catalogue_id=catalogue_id,
        session_id=session_id,
        ephemeral_id=ephemeral_id,
    )
    return {"status": "linked", "ephemeral_id": ephemeral_id, "catalogue_id": catalogue_id}


# ── Review UI page ─────────────────────────────────────────────────────────────

@app.get("/review", response_class=HTMLResponse, summary="Open the speaker review UI")
def review_page():
    return Path("review_ui.html").read_text()


# ── Review endpoints ───────────────────────────────────────────────────────────

@app.get("/review/{session_id}", summary="Get full review payload for a session")
def get_review(session_id: str):
    import sqlite3
    from catalogue import SpeakerCatalogue

    cat = SpeakerCatalogue()

    # Load session row
    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = {
        "session_id": row["session_id"],
        "source_name": row["source_name"],
        "source_file": row["source_file"],
        "processed_at": row["processed_at"],
        "total_duration": row["total_duration"],
        "num_speakers": row["num_speakers"],
    }

    result_json = row["result_json"]
    if not result_json:
        raise HTTPException(status_code=404, detail="Session has no result data")
    result = json.loads(result_json)
    turns_raw = result.get("turns", [])
    session_data["original_turns"] = result.get("original_turns", [])
    session_data["merge_gap_secs"] = result.get("merge_gap_secs", 1.0)
    name_hints: dict = result.get("name_hints", {})

    # Load appearance links
    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        links = cx.execute("""
            SELECT a.ephemeral_id, a.catalogue_id, s.display_name, s.affiliation, s.role
            FROM appearances a
            LEFT JOIN speakers s ON a.catalogue_id = s.catalogue_id
            WHERE a.session_id = ?
        """, (session_id,)).fetchall()
    link_map: dict[str, dict] = {}
    for lnk in links:
        link_map[lnk["ephemeral_id"]] = {
            "catalogue_id": lnk["catalogue_id"],
            "display_name": lnk["display_name"],
            "affiliation": lnk["affiliation"],
            "role": lnk["role"],
        }

    # Load turn overrides
    overrides = cat.get_turn_overrides(session_id)

    # Load captures.json if exists
    captures_path = Path("output") / session_id / "captures.json"
    captures: dict = {}
    if captures_path.exists():
        try:
            captures = json.loads(captures_path.read_text()).get("speakers", {})
        except Exception:
            captures = {}

    # Build speaker objects grouped by effective speaker
    all_ephemeral_ids: list[str] = sorted({t["speaker_id"] for t in turns_raw})

    # Compute per-speaker turn lists with effective speaker applied
    speaker_turns: dict[str, list] = {sid: [] for sid in all_ephemeral_ids}
    for i, t in enumerate(turns_raw):
        orig = t["speaker_id"]
        if i in overrides:
            effective = overrides[i]["assigned_speaker"]
        else:
            effective = orig
        is_deleted = (effective == "__DELETED__")
        turn_obj = {
            "index": i,
            "start": t.get("start", 0),
            "end": t.get("end", 0),
            "duration": t.get("duration", 0),
            "transcript": t.get("transcript", ""),
            "sentiment": t.get("sentiment", "neutral"),
            "sentiment_score": t.get("sentiment_score", 0),
            "original_speaker": orig,
            "effective_speaker": effective,
            "overridden": i in overrides,
            "deleted": is_deleted,
            "override_notes": overrides[i]["notes"] if i in overrides else "",
            "frame_url": f"/review/{session_id}/turns/{i}/frame",
        }
        if is_deleted:
            # Keep deleted turns in the original speaker's bucket (for display),
            # but flag them so stats can exclude them.
            speaker_turns[orig].append(turn_obj)
        else:
            if effective not in speaker_turns:
                speaker_turns[effective] = []
            speaker_turns[effective].append(turn_obj)

    total_duration = row["total_duration"] or 1.0

    speakers_out = []
    for eph_id in all_ephemeral_ids:
        lnk = link_map.get(eph_id, {})
        cap = captures.get(eph_id, {})

        # Find frame URL
        frame_url = None
        frames_dir = Path("output") / session_id / "frames"
        if frames_dir.exists():
            matches = list(frames_dir.glob(f"{eph_id}_*.png"))
            if matches:
                frame_file = sorted(matches)[0]
                frame_url = f"/output-files/{session_id}/frames/{frame_file.name}"

        turns_for_speaker = speaker_turns.get(eph_id, [])
        active_turns = [t for t in turns_for_speaker if not t.get("deleted")]
        speaking_time = sum(t["duration"] for t in active_turns)

        speakers_out.append({
            "ephemeral_id": eph_id,
            "catalogue_id": lnk.get("catalogue_id"),
            "display_name": lnk.get("display_name"),
            "affiliation": lnk.get("affiliation"),
            "role": lnk.get("role"),
            "suggested_name": cap.get("suggested_name"),
            "suggested_title": cap.get("suggested_title"),
            "suggested_org": cap.get("suggested_org"),
            "confidence": cap.get("confidence"),
            "raw_text": cap.get("raw_text"),
            "capture_timestamp": cap.get("timestamp"),
            "frame_url": frame_url,
            "total_speaking_time": round(speaking_time, 3),
            "turn_count": len(active_turns),
            "turns": turns_for_speaker,
            "name_hints": name_hints.get(eph_id, []),
        })

    # Sort by total speaking time descending
    speakers_out.sort(key=lambda s: s["total_speaking_time"], reverse=True)

    return {
        "session": session_data,
        "speakers": speakers_out,
        "all_ephemeral_ids": all_ephemeral_ids,
    }


@app.post("/review/{session_id}/speakers/{ephemeral_id}/apply-hint", summary="Auto-link a speaker via an NER name hint")
def apply_hint(session_id: str, ephemeral_id: str, name: str = Form(...)):
    from catalogue import SpeakerCatalogue
    cat = SpeakerCatalogue()

    # Verify session exists
    import sqlite3
    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT session_id FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    profiles = cat.search_speakers(name=name)
    matches = [p for p in profiles if (p.display_name or "").lower() == name.lower()]

    if len(matches) == 0:
        return {"status": "no_match"}
    if len(matches) > 1:
        return {"status": "ambiguous", "matches": [p.display_name for p in matches]}

    matched = matches[0]
    cat.link_appearance(
        catalogue_id=matched.catalogue_id,
        session_id=session_id,
        ephemeral_id=ephemeral_id,
    )
    return {"status": "linked", "catalogue_id": matched.catalogue_id}


@app.post("/review/{session_id}/turns/{turn_index}/assign", summary="Reassign a single turn to a different speaker")
def assign_turn(
    session_id: str,
    turn_index: int,
    assigned_speaker: str = Form(...),
    notes: str = Form(""),
):
    import sqlite3, json
    from catalogue import SpeakerCatalogue

    cat = SpeakerCatalogue()

    # Verify session exists and get original speaker
    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    result = json.loads(row["result_json"])
    turns = result.get("turns", [])
    if turn_index < 0 or turn_index >= len(turns):
        raise HTTPException(status_code=400, detail=f"turn_index {turn_index} out of range")

    original_speaker = turns[turn_index]["speaker_id"]
    cat.save_turn_override(
        session_id=session_id,
        turn_index=turn_index,
        original_speaker=original_speaker,
        assigned_speaker=assigned_speaker,
        notes=notes,
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "turn_index": turn_index,
        "assigned_speaker": assigned_speaker,
    }


@app.delete("/review/{session_id}/turns/{turn_index}", summary="Mark a turn as deleted")
def delete_turn(session_id: str, turn_index: int):
    import sqlite3, json
    from catalogue import SpeakerCatalogue

    cat = SpeakerCatalogue()
    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    turns = json.loads(row["result_json"]).get("turns", [])
    if turn_index < 0 or turn_index >= len(turns):
        raise HTTPException(status_code=400, detail=f"turn_index {turn_index} out of range")

    original_speaker = turns[turn_index]["speaker_id"]
    cat.save_turn_override(
        session_id=session_id,
        turn_index=turn_index,
        original_speaker=original_speaker,
        assigned_speaker="__DELETED__",
        notes="deleted by reviewer",
    )
    return {"status": "deleted", "session_id": session_id, "turn_index": turn_index}


@app.post("/review/{session_id}/turns/{turn_index}/restore", summary="Restore a deleted or overridden turn to its original speaker")
def restore_turn(session_id: str, turn_index: int):
    import sqlite3
    from catalogue import SpeakerCatalogue

    cat = SpeakerCatalogue()
    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT session_id FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    cat.delete_turn_override(session_id=session_id, turn_index=turn_index)
    return {"status": "restored", "session_id": session_id, "turn_index": turn_index}


@app.post("/review/{session_id}/speakers/{ephemeral_id}/merge", summary="Bulk-reassign all turns from one speaker to another")
def merge_speaker(
    session_id: str,
    ephemeral_id: str,
    target_speaker: str = Form(...),
    notes: str = Form(""),
):
    import sqlite3, json
    from catalogue import SpeakerCatalogue

    cat = SpeakerCatalogue()

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    result = json.loads(row["result_json"])
    turns = result.get("turns", [])

    merged = 0
    for i, turn in enumerate(turns):
        if turn["speaker_id"] == ephemeral_id:
            cat.save_turn_override(
                session_id=session_id,
                turn_index=i,
                original_speaker=ephemeral_id,
                assigned_speaker=target_speaker,
                notes=notes,
            )
            merged += 1

    return {
        "status": "ok",
        "merged_turns": merged,
        "from": ephemeral_id,
        "into": target_speaker,
    }


@app.post("/review/{session_id}/remerge", summary="Preview a new merge-gap setting without saving")
def preview_remerge(session_id: str, merge_gap_secs: float = Form(...)):
    import sqlite3
    from diarizer import Turn, merge_turns

    cat_mod = __import__("catalogue", fromlist=["SpeakerCatalogue"])
    cat = cat_mod.SpeakerCatalogue()

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    result = json.loads(row["result_json"])
    # Use original (pre-merge) turns when available, else fall back to current turns
    raw = result.get("original_turns") or result.get("turns", [])

    _TURN_FIELDS = set(Turn.__dataclass_fields__.keys())
    turns = [Turn(**{k: v for k, v in t.items() if k in _TURN_FIELDS}) for t in raw]
    # Ensure merged_count defaults
    for t in turns:
        if t.merged_count == 0:
            t.merged_count = 1

    merged = merge_turns(turns, merge_gap_secs)
    return {
        "original_count": len(turns),
        "merged_count": len(merged),
        "merge_gap_secs": merge_gap_secs,
        "turns": [
            {
                "speaker_id": t.speaker_id,
                "start": t.start,
                "end": t.end,
                "duration": t.duration,
                "transcript": t.transcript,
                "sentiment": t.sentiment,
                "sentiment_score": t.sentiment_score,
                "merged_count": t.merged_count,
            }
            for t in merged
        ],
    }


@app.post("/review/{session_id}/apply-remerge", summary="Apply a new merge-gap and save to the session (clears overrides)")
def apply_remerge(session_id: str, merge_gap_secs: float = Form(...)):
    import sqlite3
    from diarizer import Turn, merge_turns, _speaker_stats

    cat_mod = __import__("catalogue", fromlist=["SpeakerCatalogue"])
    cat = cat_mod.SpeakerCatalogue()

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json, total_duration FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    result = json.loads(row["result_json"])
    total_duration = row["total_duration"] or 1.0

    raw = result.get("original_turns") or result.get("turns", [])
    _TURN_FIELDS = set(Turn.__dataclass_fields__.keys())
    turns = [Turn(**{k: v for k, v in t.items() if k in _TURN_FIELDS}) for t in raw]
    for t in turns:
        if t.merged_count == 0:
            t.merged_count = 1

    merged = merge_turns(turns, merge_gap_secs)
    new_stats = _speaker_stats(merged, total_duration)

    # Update result_json with new turns, stats, and merge settings
    from dataclasses import asdict
    result["turns"] = [asdict(t) for t in merged]
    result["speaker_stats"] = new_stats
    result["original_turns"] = [asdict(t) for t in turns]
    result["merge_gap_secs"] = merge_gap_secs

    new_json = json.dumps(result, indent=2)

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.execute(
            "UPDATE sessions SET result_json=? WHERE session_id=?",
            (new_json, session_id),
        )
        # Clear overrides — they reference old turn indices
        cx.execute("DELETE FROM turn_overrides WHERE session_id=?", (session_id,))

    log.info(f"Re-merged session {session_id}: {len(raw)} → {len(merged)} turns (gap={merge_gap_secs}s)")
    return {
        "status": "ok",
        "session_id": session_id,
        "original_count": len(raw),
        "merged_count": len(merged),
        "merge_gap_secs": merge_gap_secs,
    }


@app.get("/review/{session_id}/audio/{turn_index}", summary="Stream audio clip for a single turn")
def get_turn_audio(
    session_id: str,
    turn_index: int,
    start: Optional[float] = None,
    end: Optional[float] = None,
):
    import sqlite3, json, subprocess, tempfile
    from catalogue import SpeakerCatalogue

    cat = SpeakerCatalogue()

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json, source_file FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    source_file = row["source_file"] or ""
    # Check if it's a local file
    from screen_capture import _is_youtube
    if _is_youtube(source_file) or not Path(source_file).exists():
        raise HTTPException(status_code=404, detail="audio not available for this source")

    result = json.loads(row["result_json"])
    turns = result.get("turns", [])
    if turn_index < 0 or turn_index >= len(turns):
        raise HTTPException(status_code=400, detail=f"turn_index {turn_index} out of range")

    turn = turns[turn_index]
    # Allow caller to override start/end (used when playing a compressed span)
    start = start if start is not None else turn.get("start", 0)
    end = end if end is not None else turn.get("end", 0)
    duration = end - start

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
    tmp.close()
    wav_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", source_file,
        "-ac", "1",
        "-ar", "16000",
        wav_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail="ffmpeg failed to extract audio segment")

    return FileResponse(wav_path, media_type="audio/wav")


@app.get("/review/{session_id}/turns/{turn_index}/frame", summary="Get or on-demand capture a frame for a specific turn")
def get_turn_frame(session_id: str, turn_index: int):
    import sqlite3
    from catalogue import SpeakerCatalogue
    from screen_capture import ScreenCapture, _is_youtube

    cat = SpeakerCatalogue()

    with sqlite3.connect(str(cat.db_path)) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT result_json, source_file FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    result = json.loads(row["result_json"])
    turns = result.get("turns", [])

    if turn_index < 0 or turn_index >= len(turns):
        raise HTTPException(status_code=400, detail=f"turn_index {turn_index} out of range")

    turn = turns[turn_index]
    speaker_id = turn.get("speaker_id", "SPEAKER_00")
    start = turn.get("start", 0.0)
    end = turn.get("end", 0.0)
    duration = end - start
    ts = start + min(1.5, duration * 0.8)

    frames_dir = Path("output") / session_id / "frames"
    frame_path = frames_dir / f"turn_{turn_index}_{speaker_id}_{ts:.2f}.png"

    if not frame_path.exists():
        source_file = row["source_file"] or ""
        if not source_file:
            raise HTTPException(status_code=404, detail="No video source available")

        _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        if not _is_youtube(source_file) and Path(source_file).suffix.lower() in _AUDIO_EXTS:
            raise HTTPException(status_code=404, detail="Audio-only source: no video frames available")

        if not _is_youtube(source_file) and not Path(source_file).exists():
            raise HTTPException(status_code=404, detail="Source file not found on disk")

        try:
            sc = ScreenCapture(output_dir="output", use_vision=False)
            frames_dir.mkdir(parents=True, exist_ok=True)
            playback_url = sc._resolve_source(source_file)
            ok = sc._extract_frame(playback_url, ts, frame_path)
        except Exception as exc:
            log.warning(f"Frame capture failed for turn {turn_index}: {exc}")
            raise HTTPException(status_code=500, detail=f"Frame capture failed: {exc}")

        if not ok or not frame_path.exists():
            raise HTTPException(status_code=500, detail="ffmpeg produced no output")

    return FileResponse(str(frame_path), media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "hf_token_configured": bool(HF_TOKEN),
        "active_jobs": sum(1 for j in _jobs.values() if j.status == "running"),
    }
