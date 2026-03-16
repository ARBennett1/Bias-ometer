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
import re
from collections import Counter
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
    merged_count: int
        Number of original pyannote turns collapsed into this one by the
        contiguous-speaker merge step (1 means no merging occurred).
    """
    speaker_id: str
    start: float
    end: float
    duration: float
    transcript: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    merged_count: int = 1

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
    name_hints: dict[str, list[str]]
        Populated by the NER step when enable_ner=True and transcription is
        enabled. Maps each speaker_id to an ordered list of candidate name
        strings, ranked by frequency of occurrence descending (up to 5 per
        speaker).
    """

    source_file: str
    source_name: str
    processed_at: str
    total_duration: float
    num_speakers: int
    turns: list[Turn] = field(default_factory=list)
    speaker_stats: dict = field(default_factory=dict)
    original_turns: list[Turn] = field(default_factory=list)
    merge_gap_secs: float = 1.0
    name_hints: dict = field(default_factory=dict)

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
        enable_ner: bool = True,
        device: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        merge_gap_secs: float = 1.0,
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
        enable_ner:
            Run spaCy NER over transcripts to extract PERSON name hints per
            speaker. Requires enable_transcription=True; silently skipped
            otherwise.
        device:
            "mps" | "cuda" | "cpu". Auto-detected when None.
        num_speakers:
            Fix speaker count (overrides min/max hints).
        min_speakers:
            Lower bound hint for speaker count.
        max_speakers:
            Upper bound hint for speaker count.
        merge_gap_secs:
            Maximum silence gap in seconds between consecutive turns by the
            same speaker that will be merged into one. Set to 0.0 to disable
            merging entirely.
        """

        self.enable_transcription = enable_transcription
        self.enable_sentiment = enable_sentiment
        self.enable_ner = enable_ner
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.merge_gap_secs = merge_gap_secs

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

        # ==============================================================
        # spaCy NER (optional — skipped if transcription is off)
        # ==============================================================
        self._nlp = None
        if self.enable_ner and self.enable_transcription:
            try:
                import sys as _sys
                # spaCy depends on a third-party package also called 'catalogue'.
                # Our local catalogue.py shadows it because the project directory
                # sits at the front of sys.path. Guard the import by temporarily
                # removing our directory and our catalogue module from Python's
                # search paths so spaCy can find its real dependency.
                if 'spacy' not in _sys.modules:
                    _proj = str(Path(__file__).parent.resolve())
                    _removed: list[tuple[int, str]] = []
                    for _i, _p in reversed(list(enumerate(_sys.path))):
                        _r = str(Path(_p).resolve()) if _p else str(Path.cwd().resolve())
                        if _r == _proj:
                            _removed.append((_i, _p))
                            del _sys.path[_i]
                    _our_cat = _sys.modules.pop('catalogue', None)
                    _is_ours = _our_cat is not None and not hasattr(_our_cat, 'create')
                    if not _is_ours and _our_cat is not None:
                        _sys.modules['catalogue'] = _our_cat  # restore spaCy's if present
                    try:
                        import spacy
                    finally:
                        for _i, _p in sorted(_removed):
                            _sys.path.insert(_i, _p)
                        if _is_ours:
                            _sys.modules['catalogue'] = _our_cat
                else:
                    import spacy
                self._nlp = spacy.load("en_core_web_sm")
                log.info("spaCy en_core_web_sm loaded for NER")
            except ImportError:
                log.warning(
                    "spaCy not installed — NER disabled. "
                    "pip install spacy && python -m spacy download en_core_web_sm"
                )
            except OSError:
                log.warning(
                    "spaCy model en_core_web_sm not found — NER disabled. "
                    "Run: python -m spacy download en_core_web_sm"
                )
            except AttributeError as e:
                log.warning(
                    f"spaCy import conflict (likely 'catalogue' name collision) — "
                    f"NER disabled. ({e})"
                )

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
        # 4b. Merge contiguous same-speaker turns
        # ==============================================================
        pre_merge_turns = list(turns)
        turns = self._merge_turns(turns)

        # ==============================================================
        # 4c. NER name hints
        # ==============================================================
        name_hints = self._extract_name_hints(turns) if self.enable_ner else {}

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
            original_turns=pre_merge_turns if self.merge_gap_secs > 0.0 else [],
            merge_gap_secs=self.merge_gap_secs,
            name_hints=name_hints,
        )

    # ==================================================================
    # Internal steps
    # ==================================================================

    def _merge_turns(self, turns: list[Turn]) -> list[Turn]:
        """Merge consecutive same-speaker turns separated by a small gap."""
        merged = merge_turns(turns, self.merge_gap_secs)
        log.info(
            f"  Turn merge: {len(turns)} → {len(merged)} turns "
            f"(gap ≤ {self.merge_gap_secs}s)"
        )
        return merged

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

    def _extract_name_hints(self, turns: list[Turn]) -> dict[str, list[str]]:
        """
        Run spaCy NER and regex patterns over each speaker's transcripts to
        extract PERSON entity candidates. Returns a dict mapping speaker_id
        to a frequency-ranked list of name strings (up to 5 per speaker).
        """
        if self._nlp is None:
            return {}

        # Build per-speaker corpus
        speaker_corpus: dict[str, str] = {}
        for turn in turns:
            text = (turn.transcript or "").strip()
            if not text:
                continue
            if turn.speaker_id not in speaker_corpus:
                speaker_corpus[turn.speaker_id] = ""
            speaker_corpus[turn.speaker_id] += text + "\n"

        if not speaker_corpus:
            return {}

        # Patterns for self-identification and reporter sign-off
        _SELF_ID = re.compile(
            r"(?:I(?:'m| am))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        )
        _REPORTER = re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:reporting|here|joining)",
        )

        # Introduction patterns — anchor names the guest; attribute to next speaker
        _INTRO_PATTERNS = [
            re.compile(r"joining me (?:now )?is ([A-Z][a-z]+ (?:[A-Z][a-z]+ )+)", re.IGNORECASE),
            re.compile(r"(?:welcome|with me now)[,\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", re.IGNORECASE),
        ]

        # Per-speaker frequency counters
        counters: dict[str, Counter] = {sid: Counter() for sid in speaker_corpus}

        # --- spaCy + regex over per-speaker corpus ---
        for sid, text in speaker_corpus.items():
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if len(name) >= 2 and " " in name:
                        counters[sid][name] += 1

            for m in _SELF_ID.finditer(text):
                counters[sid][m.group(1).strip()] += 1
            for m in _REPORTER.finditer(text):
                counters[sid][m.group(1).strip()] += 1

        # --- Introduction pattern: attribute to next speaker ---
        sorted_turns = sorted(turns, key=lambda t: t.start)
        for i, turn in enumerate(sorted_turns):
            text = (turn.transcript or "").strip()
            if not text:
                continue
            for pat in _INTRO_PATTERNS:
                for m in pat.finditer(text):
                    name = m.group(1).strip()
                    if not name:
                        continue
                    # Find the next speaker after this turn ends
                    next_speaker: Optional[str] = None
                    for j in range(i + 1, len(sorted_turns)):
                        if sorted_turns[j].speaker_id != turn.speaker_id:
                            next_speaker = sorted_turns[j].speaker_id
                            break
                    if next_speaker and next_speaker in counters:
                        counters[next_speaker][name] += 1

        # Build result: top-5 names per speaker, frequency descending
        result: dict[str, list[str]] = {}
        for sid, counter in counters.items():
            if counter:
                result[sid] = [name for name, _ in counter.most_common(5)]

        n_with_hints = sum(1 for v in result.values() if v)
        log.info(f"  NER: name hints found for {n_with_hints}/{len(speaker_corpus)} speakers")
        return result

# ======================================================================
# Public merge helper — usable outside the pipeline (e.g. API re-merge)
# ======================================================================

def merge_turns(turns: list[Turn], gap_secs: float) -> list[Turn]:
    """
    Merge consecutive same-speaker turns whose silence gap is ≤ gap_secs.

    Pass gap_secs=0.0 to disable (returns turns unchanged).
    Each merged Turn's merged_count accumulates the constituent counts.
    """
    if gap_secs == 0.0 or not turns:
        return turns

    merged: list[Turn] = [Turn(
        speaker_id=turns[0].speaker_id,
        start=turns[0].start,
        end=turns[0].end,
        duration=turns[0].duration,
        transcript=turns[0].transcript,
        sentiment=turns[0].sentiment,
        sentiment_score=turns[0].sentiment_score,
        merged_count=turns[0].merged_count,
    )]

    for cur in turns[1:]:
        prev = merged[-1]
        gap = cur.start - prev.end
        if cur.speaker_id == prev.speaker_id and gap <= gap_secs:
            new_end = cur.end
            new_duration = new_end - prev.start

            if prev.sentiment_score is not None and cur.sentiment_score is not None:
                new_score: Optional[float] = round(
                    (prev.sentiment_score * prev.duration + cur.sentiment_score * cur.duration)
                    / new_duration,
                    4,
                )
            elif prev.sentiment_score is not None:
                new_score = prev.sentiment_score
            elif cur.sentiment_score is not None:
                new_score = cur.sentiment_score
            else:
                new_score = None

            if new_score is None:
                new_sentiment: Optional[str] = None
            elif new_score > 0.1:
                new_sentiment = "positive"
            elif new_score < -0.1:
                new_sentiment = "negative"
            else:
                new_sentiment = "neutral"

            left = (prev.transcript or "").strip()
            right = (cur.transcript or "").strip()
            new_transcript: Optional[str] = (
                (left + " " + right).strip() if (left or right) else None
            )

            prev.end = new_end
            prev.duration = round(new_duration, 3)
            prev.transcript = new_transcript
            prev.sentiment_score = new_score
            prev.sentiment = new_sentiment
            prev.merged_count += cur.merged_count
        else:
            merged.append(Turn(
                speaker_id=cur.speaker_id,
                start=cur.start,
                end=cur.end,
                duration=cur.duration,
                transcript=cur.transcript,
                sentiment=cur.sentiment,
                sentiment_score=cur.sentiment_score,
                merged_count=cur.merged_count,
            ))

    return merged


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
