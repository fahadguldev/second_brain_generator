# Second Brain Generator (`second_brain_generator`)

A modular, standalone Python toolkit that enables anyone to build their own AI Second Brain from scratch using raw media (video, audio), notes (markdown, text), and social comments/QA exports.

---

## Features

- **Media Transcription**: Speech-to-text with `faster-whisper`, Voice Activity Detection (VAD) filtering to strip dead air, and incremental skipping to avoid redundant reprocessing.
- **Multi-Source Knowledge Compilation**: Ingests transcripts (`.txt`), raw text notes (`.txt`), Markdown files (`.md` with YAML frontmatter), and comment exports (`.json`, `.csv`).
- **Smart Filtering & Classification**: Detects language (English, Hinglish, Hindi, Urdu), tags topic taxonomies, filters out DM deflections (e.g. "check dm"), and generates RFC 4122 deterministic UUIDv5 identifiers.
- **Gemini Vector Embeddings**: Generates 3072-dimensional vector embeddings with Google Gemini (`models/gemini-embedding-2`), supporting multi-key rotation, sliding-window rate limiting, and batch checkpointing.
- **Offline Evaluation**: High-performance local NumPy cosine similarity search and benchmark evaluation.
- **Qdrant Vector Indexing**: Auto-creates collections and batch-upserts points with deterministic UUIDs and rich metadata payloads.
- **Unified CLI**: Run the entire pipeline at once or execute individual stages step-by-step.

---

## Directory Structure

Drop your raw content into the corresponding folders under `data/`:

```
second_brain_generator/
├── brain_config.yaml         # Persona, tone guidelines, topic taxonomy, paths
├── run_generator.py          # Unified CLI orchestrator
├── test_brain.py             # Standalone offline retrieval test tool
├── uploader.py               # Standalone Qdrant batch uploader
├── data/                     # Raw input content
│   ├── vids/                 # .mp4, .mov, .mkv video recordings
│   ├── audios/               # .mp3, .wav, .m4a audio notes
│   ├── text/                 # .txt raw notes
│   ├── md/                   # .md markdown notes
│   └── comments/             # .json and .csv comment/QA exports
└── brain_data/               # Generated artifacts
    ├── text/                 # Whisper transcripts
    ├── records.jsonl         # Normalized knowledge base records
    ├── rag_records.jsonl     # Search-ready RAG chunks with metadata
    └── gemini_embeddings.npz # 3072-dimensional vector embeddings
```

---

## Quickstart

### 1. Set Up Virtual Environment & Dependencies

You can set up your environment using either **`uv`** (fastest) or standard **`python3` / `pip`**:

#### Option A: Using `uv` (Recommended)
```bash
# Create virtual environment with uv
uv venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies with uv pip
uv pip install -r requirements.txt
```

#### Option B: Using standard `python` & `pip`
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Directories

Scaffold the input and output directory tree:

```bash
python run_generator.py --init
```

### 3. Configure Environment

Create a `.env` file in your project root with your credentials:

```dotenv
# Gemini API Key(s) - Supports multi-key rotation
GEMINI_API_KEY=AIzaSy...
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=second_brain
```

### 4. Add Your Content

Drop your media files in `data/vids/` and `data/audios/`, notes in `data/text/` and `data/md/`, and comments in `data/comments/`.

---

## Pipeline Execution

### Run Everything at Once

To run the complete pipeline end-to-end:

```bash
python run_generator.py --all
```

This sequentially executes:
1. Directory scaffolding (`--init`)
2. Media transcription (`--transcribe`)
3. Knowledge compilation (`--process`)
4. Vector embedding generation (`--embed`)
5. Offline retrieval evaluation (`--test`)
6. Qdrant vector indexing (`--upload`)

---

### Run Step-by-Step

You can execute any individual stage on demand:

| Step | Command | Description |
|---|---|---|
| **Init** | `python run_generator.py --init` | Scaffolds `data/` and `brain_data/` directories |
| **Transcribe** | `python run_generator.py --transcribe` | Transcribes audio & video using faster-whisper |
| **Process** | `python run_generator.py --process` | Compiles notes and comments into JSONL records |
| **Embed** | `python run_generator.py --embed` | Generates 3072-dim embeddings via Gemini API |
| **Test** | `python run_generator.py --test` | Evaluates retrieval with cosine similarity |
| **Upload** | `python run_generator.py --upload` | Batch upserts vectors and payloads into Qdrant |

---

## CLI Options & Flags

```
--all, --transcribe, --process, --embed, --test, --upload
```

- `--all`: Execute all pipeline stages in sequence.
- `--transcribe`: Run speech-to-text transcription only.
- `--process`: Run compilation and classification stage only.
- `--embed`: Generate vector embeddings only.
- `--test`: Run retrieval benchmark tests only.
- `--upload`: Index and upload points to Qdrant only.
- `--init`: Initialize directory hierarchy.
- `--dry-run`: Simulate pipeline actions without modifying files or making network calls.
- `--force`: Force re-transcription / re-processing of already existing output files.
- `--config PATH`: Path to custom `brain_config.yaml` (defaults to `brain_config.yaml`).
- `--data-dir PATH`: Override the root data directory path.

---

## Testing Retrieval Offline

You can test queries directly against your local embeddings without starting a backend server:

```bash
python test_brain.py "How do I get started with cloud computing?"
python test_brain.py --top-k 10 "Should I prioritize degree or skills?"
```

---

## Running the Test Suite

The package includes a comprehensive 4-tier test suite covering 213 test scenarios:

```bash
python tests/runner.py
# Or run a specific tier:
python tests/runner.py --tier 1
python tests/runner.py --tier 2
python tests/runner.py --tier 3
python tests/runner.py --tier 4
```
