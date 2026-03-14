# News Source Diarization Pipeline

Batch speaker diarization, transcription, and sentiment analysis for TV/radio
news audio. Built on **pyannote.audio 3.x**, **OpenAI Whisper**, and
**DistilBERT SST-2**. Supports local audio files and direct YouTube ingestion.
Optimised for Apple Silicon (Mac Studio M2).

---

## Architecture

```
Local audio file                         YouTube URL
(wav / mp3 / m4a / flac)                     |
         |                           youtube.py (yt-dlp)
         |                           +-- fetch metadata
         |                           +-- download best audio
         |                           +-- re-encode to 16kHz mono WAV
         |                                     |
         +-------------------+-----------------+
                             |
                             v
          +------------------------------------------+
          |  NewsDiarizer  (diarizer.py)             |
          |                                          |
          |  1. torchaudio.load()                    |
          |  2. pyannote/speaker-diarization-3.1     | [CPU]
          |        -> (speaker, start, end) per turn |
          |  3. openai-whisper base.en               | [MPS]
          |        -> transcript per turn            |
          |  4. distilbert SST-2 sentiment           | [CPU]
          |        -> +1.0 ... -1.0 score per turn   |
          |                                          |
          |  Returns: DiarizationResult              |
          +------------------------------------------+
                  |                   |
                  v                   v
         output/<n>.json     SpeakerCatalogue  (catalogue.py)
                                  db/speakers.db
                                  +-- sessions      raw results
                                  +-- appearances   ephemeral->named links
                                  +-- speakers      enriched profiles
```

---

## Project files

| File | Purpose |
|---|---|
| `diarizer.py` | Core pipeline: pyannote + Whisper + sentiment |
| `youtube.py` | YouTube audio fetcher via yt-dlp |
| `catalogue.py` | SQLite speaker catalogue |
| `main.py` | CLI entry point |
| `.env` | Your local credentials (gitignored) |
| `.env.example` | Credentials template (committed) |

---

## Setup

### 1. Python environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. System FFmpeg

Required by both Whisper and yt-dlp for audio decoding and conversion.

```bash
brew install ffmpeg
```

### 3. HuggingFace token

1. Create an account at https://huggingface.co
2. Accept the model licence pages:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Generate a **read** token at https://huggingface.co/settings/tokens
4. Copy the env template and add your token:

```bash
cp .env.example .env
```

Then edit `.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The application loads `.env` automatically on startup via `python-dotenv`.
Your token never needs to be exported into the shell or added to `.zshrc`.

> **Note:** `.env` is listed in `.gitignore` and will never be committed.
> `.env.example` (no real credentials) is tracked by git and should be kept up to date.

---

## Usage

### Process a local audio file

```bash
# Full pipeline: diarize + transcribe + sentiment
python main.py process audio/interview.wav --source "BBC Radio 4"

# Provide speaker count hints for better accuracy
python main.py process audio/panel.mp3 \
    --source "ITV News At Ten" \
    --min-speakers 2 \
    --max-speakers 6

# Fast diarization-only (no Whisper, no sentiment)
python main.py process audio/clip.wav --no-transcription --no-sentiment
```

Results are saved to `output/<filename>_diarization.json`.

### Process a YouTube video

```bash
# Full pipeline from a YouTube URL
python main.py process-youtube "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# Override the source label (default is "YouTube · <channel name>")
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" \
    --source "BBC News" \
    --max-speakers 4

# Download audio only, without running diarization
# Useful for inspecting the file before committing to a full run
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" --download-only
```

Downloaded WAV files are cached in `audio/` by video ID. Re-running the same
URL skips the download and uses the cached file automatically.

The source label in results and the catalogue defaults to `YouTube · <channel>`,
e.g. `YouTube · BBC News`. Use `--source` to override this.

### Manage the speaker catalogue

```bash
# List all processed sessions
python main.py sessions

# Register a known speaker
python main.py add-speaker \
    --name "Jane Smith" \
    --affiliation "Reuters" \
    --role "Political Correspondent"

# After reviewing the JSON, link SPEAKER_00 -> SPK-0001
python main.py link \
    --session interview_2024-03-11T14-00-00 \
    --ephemeral SPEAKER_00 \
    --catalogue SPK-0001

# Top 20 most-interviewed speakers
python main.py speakers --top 20

# Search by name with full appearance history
python main.py speakers --search "Smith" --history

# Filter by affiliation
python main.py speakers --affiliation "Labour"
```

---

## Output JSON

```jsonc
{
  "source_file": "https://www.youtube.com/watch?v=XXXXXXXXXXX",
  "source_name": "YouTube · BBC News",
  "processed_at": "2024-03-11T14:32:00+00:00",
  "total_duration": 182.4,
  "num_speakers": 3,
  "turns": [
    {
      "speaker_id": "SPEAKER_00",
      "start": 0.512,
      "end": 8.341,
      "duration": 7.829,
      "transcript": "Good evening. Tonight's top story ...",
      "sentiment": "neutral",
      "sentiment_score": 0.021
    }
  ],
  "speaker_stats": {
    "SPEAKER_00": {
      "total_speaking_time": 94.2,
      "turn_count": 12,
      "pct_of_audio": 51.6,
      "avg_turn_duration": 7.85,
      "avg_sentiment": 0.031
    }
  }
}
```

---

## Speaker identification workflow

pyannote assigns *ephemeral* IDs (`SPEAKER_00`, `SPEAKER_01` ...) that reset
every file. The catalogue bridges this with a human review step:

```
process file -> listen / read JSON -> identify speakers by context
    -> add-speaker (if new) -> link ephemeral -> catalogue ID
```

Over multiple sessions the catalogue builds up per-speaker statistics:
- **Appearance frequency** across sources (TV, radio, YouTube)
- **Total airtime** and average turn length
- **Affiliation / role** metadata for political/editorial analysis
- **Average sentiment** to track how speakers are portrayed

---

## Roadmap / next steps

| Feature | Notes |
|---|---|
| Voice-print matching | Store pyannote speaker embeddings; cosine-search catalogue at link time to suggest matches automatically |
| Topic detection | Add BERTopic / zero-shot classifier step after transcription |
| Batch ingestion | Glob-process a directory of recordings or a YouTube playlist overnight |
| REST API | FastAPI wrapper around `NewsDiarizer` + `SpeakerCatalogue` |
| Dashboard | Streamlit or Plotly Dash for airtime / sentiment visualisation |