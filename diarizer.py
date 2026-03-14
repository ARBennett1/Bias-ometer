"""
diarizer.py
========================================================================
Core pipeline:
  1. Load audio  (torchaudio - handles wav / mp3 / m4a / flac)
  2. Speaker diarization  (pyannote.audio 3.x)
  3. Transcription per turn  (OpenAI Whisper - optional)
  4. Sentiment per turn  (DistilBERT SST-2 - optional)

Designed for batch, offline processing on Apple Silicon (Mac Studio M2).
pyannote is routed to CPU for stability whilst Whisper uses MPS when
available.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

log = logging.getLogger(__name__)


# ======================================================================
# Data models
# ======================================================================
@dataclass
class Turn:
    """
    One continuous segment of speech by a single speaker.

    Attributes
    ----------
    speaker_id: str
        Unique identifier for the speaker (e.g. "SPEAKER_001").
    start: float
        Start time of the turn in seconds.
    end: float
        End time of the turn in seconds.
    duration: float
        Duration of the turn in seconds.
    transcript: Optional[str]
        Optional text transcript of the turn (if transcription enabled).
    sentiment: Optional[str]
        Optional sentiment label ("positive", "neutral", "negative").
    sentiment_score: Optional[float]
        Optional sentiment score (-1.0 to +1.0, where positive is more
        positive sentiment).
    """
    speaker_id: str
    start: float
    end: float              
    duration: float
    transcript: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None

@dataclass
class DiarizationResult:
    """
    Complete result for one audio file.
    
    Attributes
    ----------
    source_file: str
        Path to the original audio file.
    source_name: str
        Human-friendly name for the source (e.g. "BBC Radio 4").
    processed_at: str
        ISO timestamp of when the processing was done.
    total_duration: float
        Total duration of the audio in seconds.
    num_speakers: int
        Total number of unique speakers identified.
    turns: list[Turn]
        List of all speaker turns with metadata.
    speaker_stats: dict
        Aggregated stats per speaker, keyed by speaker_id, with values:
        {
            "total_speaking_time": float,
            "turn_count": int,
            "pct_of_audio": float,         
            "avg_turn_duration": float,    
            "avg_sentiment": Optional[float], 
        }
    """

    source_file: str
    source_name: str
    processed_at: str
    total_duration: float
    num_speakers: int
    turns: list[Turn] = field(default_factory=list)
    speaker_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ======================================================================
# Pipeline
# ======================================================================

class NewsDiarizer:
    """
    Wraps pyannote diarization along with optional Whisper transcription
    and optional DistilBERT sentiment analysis into a single .process()
    call.
    """

    def __init__(
        self,
        hf_token: str,
        enable_transcription: bool = True,
        enable_sentiment: bool = True,
        device: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        hf_token:
            HuggingFace access token (pyannote model gate).
        enable_transcription:
            Transcribe each turn with Whisper.
        enable_sentiment:
            Run sentiment on each transcript.
        device:
            "mps" | "cuda" | "cpu". Auto-detected when None.
        num_speakers:
            Fix speaker count (overrides min/max hints).
        min_speakers:
            Lower bound hint for speaker count.
        max_speakers:
            Upper bound hint for speaker count.
        """

        self.enable_transcription = enable_transcription
        self.enable_sentiment = enable_sentiment
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # ==============================================================
        # Device selection
        # ==============================================================
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                log.info("Apple Silicon MPS backend detected")
            elif torch.cuda.is_available():
                device = "cuda"
                log.info("CUDA backend detected")
            else:
                device = "cpu"
                log.info("Using CPU backend")
        self.device = torch.device(device)

        # ==============================================================
        # pyannote diarization (CPU most stable on MPS builds)
        # ==============================================================
        log.info("Loading pyannote/speaker-diarization-3.1 …")
        diar = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        if diar is None:
            raise RuntimeError("Failed to load pyannote diarization pipeline")
        self._diar: Pipeline = diar
        _diar_device = torch.device("cpu") if device == "mps" else self.device
        self._diar.to(_diar_device)
        log.info(f"  pyannote on: {_diar_device}")

        # ==============================================================
        # Whisper (loaded lazily)
        # ==============================================================
        self._whisper: Any | None = None
        if enable_transcription:
            self._load_whisper()

        # ==============================================================
        # Sentiment (loaded lazily)
        # ==============================================================
        self._sentiment_pipe: Any | None = None
        if enable_sentiment:
            self._load_sentiment()

    # ==================================================================
    # Model loaders
    # ==================================================================
    def _load_whisper(self) -> None:
        try:
            import whisper
            log.info("Loading Whisper base.en …")
            self._whisper = cast(
                Any, whisper.load_model("base.en", device=self.device)
            )

        except ImportError:
            log.warning("openai-whisper not installed - transcription disabled. "
                        "pip install openai-whisper")
            self.enable_transcription = False

    def _load_sentiment(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            log.info("Loading DistilBERT sentiment model …")
            pipeline_fn = cast(Any, hf_pipeline)
            self._sentiment_pipe = pipeline_fn(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device.type == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
        except ImportError:
            log.warning("transformers not installed – sentiment disabled. "
                        "pip install transformers")
            self.enable_sentiment = False

    # ==================================================================
    # Public API 
    # ==================================================================
    def process(
        self,
        audio_path: str | Path,
        source_name: str = "unknown",
        show_progress: bool = True,
    ) -> DiarizationResult:
        """
        Run the full pipeline on one audio file.

        Parameters
        ----------
        audio_path:
            Path to audio file (wav / mp3 / m4a / flac …).
        source_name:
            Human label for the source, e.g. "BBC Radio 4".
        show_progress:
            Show pyannote progress bar in the terminal.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info(f"Processing: {audio_path.name}")

        # ==============================================================
        # 1. Load audio
        # ==============================================================
        waveform, sample_rate = torchaudio.load(str(audio_path))
        total_duration = waveform.shape[-1] / sample_rate
        log.info(f"  {total_duration:.1f}s · {sample_rate} Hz · {waveform.shape[0]} ch")

        # ==============================================================
        # 2. Diarize
        # ==============================================================
        diar_kwargs: dict = {}
        if self.num_speakers is not None:
            diar_kwargs["num_speakers"] = self.num_speakers
        else:
            if self.min_speakers is not None:
                diar_kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                diar_kwargs["max_speakers"] = self.max_speakers

        audio_in = {"waveform": waveform, "sample_rate": sample_rate}
        if show_progress:
            with ProgressHook() as hook:
                output = self._diar(audio_in, hook=hook, **diar_kwargs)
        else:
            output = self._diar(audio_in, **diar_kwargs)

        # pyannote >= 3.3 wraps the result in a DiarizeOutput dataclass;
        # older versions returned an Annotation directly.
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
        log.info(f"  {len(unique_speakers)} speakers · {len(turns)} turns")

        # ==============================================================
        # 3. Transcribe
        # ==============================================================
        if self.enable_transcription and self._whisper:
            turns = self._transcribe(waveform, sample_rate, turns)

        # ==============================================================
        # 4. Sentiment
        # ==============================================================
        if self.enable_sentiment and self._sentiment_pipe:
            turns = self._run_sentiment(turns)

        # ==============================================================
        # 5. Per-speaker summary
        # ==============================================================
        stats = _speaker_stats(turns, total_duration)

        return DiarizationResult(
            source_file=str(audio_path),
            source_name=source_name,
            processed_at=datetime.now(timezone.utc).isoformat(),
            total_duration=round(total_duration, 3),
            num_speakers=len(unique_speakers),
            turns=turns,
            speaker_stats=stats,
        )

    # ==================================================================
    # Internal steps
    # ==================================================================
    def _transcribe(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        turns: list[Turn],
    ) -> list[Turn]:
        import numpy as np

        whisper_model = self._whisper
        if whisper_model is None:
            return turns

        log.info("  Transcribing turns …")
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
            if len(audio_np) < 8_000:       # skip < ~0.5 s clips
                turn.transcript = ""
                continue

            out = whisper_model.transcribe(audio_np, fp16=False, language="en")
            text_out = out.get("text", "") if isinstance(out, dict) else ""
            turn.transcript = str(text_out).strip()

            if (i + 1) % 20 == 0:
                log.info(f"    {i + 1}/{len(turns)} turns done")

        log.info("  Transcription complete")
        return turns

    def _run_sentiment(self, turns: list[Turn]) -> list[Turn]:
        sentiment_pipe = self._sentiment_pipe
        if sentiment_pipe is None:
            return turns

        log.info("  Running sentiment …")
        for turn in turns:
            text = turn.transcript or ""
            if not text:
                turn.sentiment, turn.sentiment_score = "neutral", 0.0
                continue
            try:
                r = sentiment_pipe(text)[0]
                label, conf = r["label"].lower(), float(r["score"])
                turn.sentiment = label
                turn.sentiment_score = round(conf if label == "positive" else -conf, 4)
            except Exception as exc:
                log.warning(f"Sentiment error: {exc}")
                turn.sentiment, turn.sentiment_score = "neutral", 0.0
        log.info("  Sentiment complete")
        return turns

# ======================================================================
# Helper to aggregate per-speaker stats from the list of turns.
# ======================================================================
def _speaker_stats(turns: list[Turn], total_duration: float) -> dict:
    """Aggregate speaking-time and sentiment per speaker."""
    stats: dict[str, dict] = {}
    for t in turns:
        s = stats.setdefault(t.speaker_id, {
            "total_speaking_time": 0.0,
            "turn_count": 0,
            "pct_of_audio": 0.0,
            "avg_turn_duration": 0.0,
            "_scores": [],
            "avg_sentiment": None,
        })
        s["total_speaking_time"] += t.duration
        s["turn_count"] += 1
        if t.sentiment_score is not None:
            s["_scores"].append(t.sentiment_score)

    for s in stats.values():
        s["total_speaking_time"] = round(s["total_speaking_time"], 3)
        s["pct_of_audio"] = (
            round(100 * s["total_speaking_time"] / total_duration, 2)
            if total_duration else 0.0
        )
        s["avg_turn_duration"] = (
            round(s["total_speaking_time"] / s["turn_count"], 3)
            if s["turn_count"] else 0.0
        )
        scores = s.pop("_scores")
        s["avg_sentiment"] = round(sum(scores) / len(scores), 4) if scores else None

    return stats
