# News Source Diarization Pipeline

Batch speaker diarization, transcription, sentiment analysis, and automatic visual speaker identification for TV/radio news audio. Built on **pyannote.audio 3.x**, **OpenAI Whisper**, **DistilBERT SST-2**, and **Claude Vision**. Supports local audio files and direct YouTube ingestion. Optimised for Apple Silicon (Mac Studio M2).

---

## Architecture

```
Local audio file                         YouTube URL
(wav / mp3 / m4a / flac)                     │
         │                           youtube.py (yt-dlp)
         │                           ├── fetch metadata
         │                           ├── download best audio
         │                           └── re-encode to 16kHz mono WAV
         │                                     │
         └───────────────┬─────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────────────┐
          │  NewsDiarizer  (diarizer.py)             │
          │                                          │
          │  1. torchaudio.load()                    │
          │  2. pyannote/speaker-diarization-3.1     │ [CPU]
          │        → (speaker, start, end) per turn  │
          │  3. openai-whisper base.en               │ [MPS]
          │        → transcript per turn             │
          │  4. distilbert SST-2 sentiment           │ [CPU]
          │        → +1.0 … -1.0 score per turn      │
          │                                          │
          │  Returns: DiarizationResult              │
          └──────────────────────────────────────────┘
                  │                        │
                  ▼                        ▼
         output/<n>.json       SpeakerCatalogue (catalogue.py)
                                    db/speakers.db
                                    ├── sessions      raw results
                                    ├── appearances   ephemeral→named links
                                    └── speakers      enriched profiles
                  │
                  ▼ (YouTube sources only)
          ┌──────────────────────────────────────────┐
          │  ScreenCapture  (screen_capture.py)      │
          │                                          │
          │  1. Resolve video stream URL (yt-dlp)    │
          │  2. Extract frames at speaker-change      │
          │     timestamps (ffmpeg)                  │
          │  3. Claude Vision OCR: read lower-thirds │
          │     chyrons, name cards                  │
          │                                          │
          │  Returns: {speaker_id: FrameResult}      │
          └──────────────────────────────────────────┘
                  │
                  ▼
         output/<session_id>/
             captures.json
             frames/<speaker_id>_<timestamp>.png
```

---

## Project files

| File | Purpose |
|---|---|
| [main.py](main.py) | CLI entry point — all commands |
| [diarizer.py](diarizer.py) | Core pipeline: pyannote + Whisper + sentiment |
| [youtube.py](youtube.py) | YouTube audio fetcher via yt-dlp |
| [catalogue.py](catalogue.py) | SQLite speaker catalogue |
| [screen_capture.py](screen_capture.py) | Frame extraction + Claude Vision speaker identification |
| [api.py](api.py) | FastAPI REST API wrapper |
| [.env](.env) | Your local credentials (gitignored) |
| [requirements.txt](requirements.txt) | Python package dependencies |

---

## Installation

### Prerequisites

- **macOS** (optimised for Apple Silicon M1/M2/M3) or Linux
- **Python 3.10 or later**
- **Homebrew** (macOS) — [install from brew.sh](https://brew.sh)
- **Git**

### 1. Clone the repository

```bash
git clone <repo-url>
cd Bias-ometer
```

### 2. Install system dependencies

FFmpeg is required by Whisper, yt-dlp, and the screen capture module for audio decoding, video conversion, and frame extraction.

```bash
brew install ffmpeg
```

Verify the installation:

```bash
ffmpeg -version
```

### 3. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`. You must activate this environment every time you open a new terminal session.

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all packages including pyannote.audio, Whisper, PyTorch, transformers, yt-dlp, FastAPI, and the Anthropic SDK. Initial installation downloads several hundred MB of model weights on first use — this is expected.

### 5. Configure credentials

Copy the environment template and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` — two credentials are required:

```
# Required for speaker diarization (pyannote models)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Required for Visual speaker identification (Claude Vision)
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxx
```

The application loads `.env` automatically at startup via `python-dotenv`. You do not need to export variables to your shell.

> `.env` is listed in `.gitignore` and will never be committed. Keep your keys out of version control.

#### 5a. HuggingFace token (pyannote diarization)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model licence pages — **both links are required**:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Paste the token into `.env` as `HF_TOKEN=hf_…`

The pyannote model weights (~1 GB) are downloaded from HuggingFace on the first run and cached locally in `~/.cache/huggingface/`.

#### 5b. Anthropic API key (Claude Vision)

1. Create an account at [console.anthropic.com](https://console.anthropic.com)
2. Generate an API key under **API Keys**
3. Paste it into `.env` as `ANTHROPIC_API_KEY=sk-ant-…`

The Anthropic key is only used by the screen capture module to read on-screen text from video frames. If you use `--no-vision` or `--no-capture`, this key is not needed.

### 6. Verify the setup

```bash
python main.py --help
```

You should see a list of available commands. If you see import errors, ensure the virtual environment is activated and all packages installed.

---

## Running the application

### CLI (command-line interface)

All diarization and catalogue management is done through `main.py`:

```bash
# Activate the virtual environment first
source .venv/bin/activate

python main.py <command> [options]
```

See [CLI Reference](#cli-reference) below for all commands.

### REST API

The FastAPI server exposes the same functionality over HTTP, suitable for integration with a frontend or other services:

```bash
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive documentation is auto-generated at:

- `http://localhost:8000/docs` — Swagger UI
- `http://localhost:8000/redoc` — ReDoc

See [REST API Reference](#rest-api-reference) below for all endpoints.

---

## CLI Reference

### `process` — Diarise a local audio file

Runs the full pipeline (diarization → transcription → sentiment) on a local audio file. Supports WAV, MP3, M4A, and FLAC.

```bash
python main.py process <audio_file> [options]
```

| Argument | Description |
|---|---|
| `audio_file` | Path to the audio file (required) |
| `--source TEXT` | Human-readable label for the source, e.g. `"BBC Radio 4"` |
| `--output-dir DIR` | Directory for the JSON output (default: `output`) |
| `--num-speakers N` | Fix the exact number of speakers (overrides min/max hints) |
| `--min-speakers N` | Minimum number of speakers (improves accuracy when known) |
| `--max-speakers N` | Maximum number of speakers (improves accuracy when known) |
| `--no-transcription` | Skip Whisper transcription (faster diarization-only run) |
| `--no-sentiment` | Skip sentiment analysis |

**Examples:**

```bash
# Full pipeline with a source label
python main.py process audio/interview.wav --source "BBC Radio 4"

# Provide speaker count hints for better diarization accuracy
python main.py process audio/panel.mp3 \
    --source "ITV News At Ten" \
    --min-speakers 2 \
    --max-speakers 6

# Fast diarization-only run (no Whisper, no sentiment)
python main.py process audio/clip.wav --no-transcription --no-sentiment

# Save output to a custom directory
python main.py process audio/clip.wav --output-dir results/march
```

Results are saved to `<output-dir>/<filename>_diarization.json` and a session is recorded in the SQLite catalogue at `db/speakers.db`.

---

### `process-youtube` — Download and diarise a YouTube video

Downloads the audio from a YouTube URL, runs the full pipeline, and optionally captures video frames for visual speaker identification.

```bash
python main.py process-youtube <url> [options]
```

| Argument | Description |
|---|---|
| `url` | YouTube video URL (required) |
| `--source TEXT` | Override the source label (default: `YouTube · <channel name>`) |
| `--audio-dir DIR` | Directory for cached WAV files (default: `audio`) |
| `--output-dir DIR` | Directory for JSON output (default: `output`) |
| `--num-speakers N` | Fix the exact number of speakers |
| `--min-speakers N` | Minimum speaker count hint |
| `--max-speakers N` | Maximum speaker count hint |
| `--no-transcription` | Skip Whisper transcription |
| `--no-sentiment` | Skip sentiment analysis |
| `--download-only` | Download audio without running the pipeline |
| `--no-capture` | Skip frame capture and Claude Vision identification |
| `--no-vision` | Capture frames but skip the Claude Vision API call |
| `--cookies-from-browser BROWSER` | Use browser cookies for YouTube authentication (fixes HTTP 403 errors). Accepts: `safari`, `chrome`, `firefox`, `edge`, `chromium`, `brave`, `opera`, `vivaldi` |
| `--cookies-file PATH` | Path to a Netscape-format `cookies.txt` file |

**Examples:**

```bash
# Full pipeline from a YouTube URL
python main.py process-youtube "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# Override the source label and limit speaker count
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" \
    --source "BBC News" \
    --max-speakers 4

# Download audio only (inspect before committing to a full run)
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" --download-only

# Authenticate with YouTube via browser cookies (fixes most 403 errors)
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" \
    --cookies-from-browser safari

# Use a pre-exported cookies file
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" \
    --cookies-file cookies.txt

# Skip visual identification (faster, no Anthropic API calls)
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" --no-capture

# Capture frames but skip Vision (save screenshots as audit trail only)
python main.py process-youtube "https://youtu.be/XXXXXXXXXXX" --no-vision
```

Downloaded WAV files are cached in `audio/` by video ID. Re-running the same URL skips the download and uses the cached file automatically.

**YouTube 403 / authentication errors:**

YouTube increasingly blocks unauthenticated downloads. Fix this by passing your browser's logged-in session cookies:

```bash
# Option 1: Use cookies directly from your browser (must be logged in to YouTube)
python main.py process-youtube "https://..." --cookies-from-browser safari

# Option 2: Export cookies once to a file, then reuse
yt-dlp --cookies-from-browser safari --cookies cookies.txt \
    --skip-download "https://www.youtube.com/watch?v=..."

python main.py process-youtube "https://..." --cookies-file cookies.txt
```

---

### `sessions` — List recorded sessions

Lists all diarization sessions stored in the catalogue, ordered newest first.

```bash
python main.py sessions
```

Output shows: Session ID, source name, date, duration, and speaker count.

---

### `speakers` — Query the speaker catalogue

Lists or searches speakers in the catalogue.

```bash
python main.py speakers [options]
```

| Argument | Description |
|---|---|
| `--top N` | Show the top N most-appearing speakers (default: 20) |
| `--search TEXT` | Filter by name substring |
| `--affiliation TEXT` | Filter by affiliation substring |
| `--role TEXT` | Filter by role substring |
| `--history` | Show full appearance history for each matched speaker |

**Examples:**

```bash
# Top 20 most-interviewed speakers
python main.py speakers --top 20

# Search by name with full appearance history
python main.py speakers --search "Smith" --history

# Filter by affiliation
python main.py speakers --affiliation "Labour"

# Filter by role
python main.py speakers --role "Political Correspondent"

# Combine filters
python main.py speakers --affiliation "BBC" --role "Reporter"
```

---

### `add-speaker` — Register a new named speaker

Registers a new speaker profile in the catalogue. After adding, use `link` to associate pyannote's ephemeral IDs with this catalogue entry.

```bash
python main.py add-speaker [options]
```

| Argument | Description |
|---|---|
| `--name TEXT` | Speaker's full name |
| `--affiliation TEXT` | Organisation or party affiliation |
| `--role TEXT` | Job title or role |
| `--notes TEXT` | Free-text notes |

**Example:**

```bash
python main.py add-speaker \
    --name "Jane Smith" \
    --affiliation "Reuters" \
    --role "Political Correspondent" \
    --notes "Regular panel contributor"
```

Prints the assigned catalogue ID (e.g. `SPK-0001`).

---

### `update-speaker` — Edit a speaker's metadata

Updates metadata fields for an existing catalogue entry.

```bash
python main.py update-speaker <catalogue_id> [options]
```

| Argument | Description |
|---|---|
| `catalogue_id` | The speaker's catalogue ID, e.g. `SPK-0001` (required) |
| `--name TEXT` | New display name |
| `--affiliation TEXT` | New affiliation |
| `--role TEXT` | New role |
| `--notes TEXT` | New notes |

**Example:**

```bash
python main.py update-speaker SPK-0001 \
    --affiliation "Associated Press" \
    --role "Senior Correspondent"
```

---

### `link` — Link an ephemeral speaker ID to a catalogue entry

Binds a pyannote ephemeral ID (e.g. `SPEAKER_00`) from a specific session to a named catalogue entry. Speaking time and turn statistics are pulled automatically from the stored session JSON.

```bash
python main.py link --session <session_id> --ephemeral <SPEAKER_XX> --catalogue <SPK-XXXX>
```

| Argument | Description |
|---|---|
| `--session TEXT` | Session ID from `python main.py sessions` (required) |
| `--ephemeral TEXT` | Ephemeral speaker ID, e.g. `SPEAKER_00` (required) |
| `--catalogue TEXT` | Catalogue ID, e.g. `SPK-0001` (required) |

**Example:**

```bash
# First review the session list
python main.py sessions

# Then link a speaker
python main.py link \
    --session interview_2024-03-11T14-00-00 \
    --ephemeral SPEAKER_00 \
    --catalogue SPK-0001
```

---

## Screen Capture Module

`screen_capture.py` can also be run standalone as a quick test harness, without needing to run a full diarization pipeline.

```bash
python screen_capture.py <video_source> [options]
```

| Argument | Description |
|---|---|
| `source` | Local video file path or YouTube URL (required) |
| `--timestamp N`, `-t N` | Seconds into the video to capture (default: 10) |
| `--output-dir DIR`, `-o DIR` | Directory for output frames (default: `output/test_captures`) |
| `--no-vision` | Save the screenshot without sending it to Claude Vision |
| `--no-multi-frame` | Capture a single frame instead of sampling 3 candidates |
| `--cookies-from-browser BROWSER` | Browser for yt-dlp cookie auth |
| `--cookies FILE` | Path to Netscape cookies.txt for yt-dlp |
| `--debug` | Enable debug logging |

**Examples:**

```bash
# Capture a frame at 42 seconds and identify any on-screen speaker
python screen_capture.py "https://www.youtube.com/watch?v=..." --timestamp 42

# Test with a local video file
python screen_capture.py /path/to/video.mp4 --timestamp 10.5

# Save screenshot only, no API call
python screen_capture.py "https://youtu.be/..." --timestamp 30 --no-vision

# Authenticate with YouTube
python screen_capture.py "https://youtu.be/..." --timestamp 15 \
    --cookies-from-browser safari
```

---

## REST API Reference

Start the server:

```bash
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

### Jobs

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/jobs` | Submit a new processing job |
| `GET` | `/jobs` | List all recent jobs |
| `GET` | `/jobs/{job_id}` | Poll a job's status and results |

**POST /jobs** — accepts `multipart/form-data`:

| Field | Type | Description |
|---|---|---|
| `url` | string | YouTube URL (mutually exclusive with `audio_file`) |
| `audio_file` | file | Audio file upload (mutually exclusive with `url`) |
| `source_name` | string | Optional label for the source |
| `num_speakers` | integer | Fix exact speaker count |
| `min_speakers` | integer | Minimum speaker count hint |
| `max_speakers` | integer | Maximum speaker count hint |
| `enable_transcription` | boolean | Enable Whisper (default: true) |
| `enable_sentiment` | boolean | Enable sentiment analysis (default: true) |

Job status cycles through: `queued` → `running` → `complete` / `error`

Progress is reported via `progress_pct` (0–100), `progress_stage`, and `progress_detail` fields on the job object.

### Sessions

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/sessions` | List all recorded sessions |
| `GET` | `/sessions/{session_id}` | Get full session detail including transcript and speaker links |

### Speakers

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/speakers` | List or search the speaker catalogue |
| `POST` | `/speakers` | Register a new speaker |
| `PUT` | `/speakers/{catalogue_id}` | Update speaker metadata |
| `GET` | `/speakers/{catalogue_id}/appearances` | Get full appearance history |

**GET /speakers** — query parameters:

| Parameter | Description |
|---|---|
| `search` | Filter by name substring |
| `affiliation` | Filter by affiliation substring |
| `role` | Filter by role substring |
| `top` | Number of top speakers to return (default: 50) |

### Linking

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/link` | Link an ephemeral speaker ID to a catalogue entry |

**POST /link** — accepts `multipart/form-data`:

| Field | Description |
|---|---|
| `session_id` | Session ID (required) |
| `ephemeral_id` | Ephemeral ID e.g. `SPEAKER_00` (required) |
| `catalogue_id` | Catalogue ID e.g. `SPK-0001` (required) |

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check server status and configuration |

---

## Output format

### Diarization JSON (`output/<name>_diarization.json`)

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

### Screen capture JSON (`output/<session_id>/captures.json`)

```jsonc
{
  "session_id": "...",
  "source_url": "https://www.youtube.com/watch?v=...",
  "captured_at": "2024-03-11T14:35:00+00:00",
  "speakers": {
    "SPEAKER_00": {
      "speaker_id": "SPEAKER_00",
      "timestamp": 1.5,
      "frame_path": "output/<session_id>/frames/SPEAKER_00_1.50.png",
      "suggested_name": "Jane Smith",
      "suggested_title": "Political Correspondent",
      "suggested_org": "Reuters",
      "confidence": "high",
      "raw_text": "Jane Smith | Political Correspondent",
      "vision_used": true,
      "identified": true,
      "error": null
    }
  }
}
```

---

## Speaker identification workflow

pyannote assigns *ephemeral* IDs (`SPEAKER_00`, `SPEAKER_01` …) that reset with every file. The catalogue bridges this with a human review step:

```
process file
  → review transcript JSON / listen to audio
  → (for YouTube) review Vision-identified names in captures.json
  → add-speaker (if new)
  → link ephemeral ID → catalogue ID
```

Over multiple sessions, the catalogue accumulates per-speaker statistics:

- **Appearance frequency** across sources (TV, radio, YouTube)
- **Total airtime** and average turn length
- **Affiliation / role** metadata for political and editorial analysis
- **Average sentiment** to track how speakers are portrayed

---

## Directory structure

```
Bias-ometer/
├── main.py               # CLI
├── api.py                # REST API
├── diarizer.py           # Core diarization pipeline
├── youtube.py            # YouTube audio downloader
├── catalogue.py          # Speaker catalogue (SQLite)
├── screen_capture.py     # Frame extraction + Claude Vision
├── requirements.txt      # Python dependencies
├── .env                  # Credentials (gitignored)
├── .env.example          # Credentials template (committed)
├── audio/                # Cached YouTube WAV files
├── db/
│   └── speakers.db       # SQLite catalogue database
└── output/               # Processing results
    ├── <name>_diarization.json
    └── <session_id>/
        ├── captures.json
        └── frames/
            └── SPEAKER_00_1.50.png
```
