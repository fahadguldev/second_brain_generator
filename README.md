# Ask GanjiSwag — AI Second Brain Monorepo

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React_18_%2B_Vite-61DAFB?style=flat&logo=react)](https://react.dev/)
[![Framer Motion](https://img.shields.io/badge/UI-Motion_Primitives-F43F5E?style=flat&logo=framer)](https://motion.dev/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant_(3072--dim)-DC2626?style=flat&logo=qdrant)](https://qdrant.tech/)
[![Gemini](https://img.shields.io/badge/AI_Engine-Gemini_Embedding_%26_Flash-4285F4?style=flat&logo=google)](https://ai.google.dev/)

An end-to-end, production-grade AI Second Brain grounded in hundreds of hours of **Junaid Akram's (GanjiSwag)** podcasts and video episodes. Translates and bridges cross-lingual Hindi/Devanagari transcripts into authentic, long-form **Roman Urdu** mentorship, career guidance, and social commentary.

---

## Architecture Overview

```
ganjiswag/
├── .venv/                         # Unified Root Python Virtual Environment
├── README.md                      # Monorepo Guide & Documentation
├── .gitignore                     # Repository-wide ignore rules
├── testing.py                     # Root CLI proxy runner for quick pipeline queries
│
├── brain_generator/               # THE DATA & EMBEDDING FACTORY
│   ├── brain_config.yaml          # Creator persona, taxonomy, & pipeline configuration
│   ├── requirements.txt           # Pipeline dependencies (Whisper, Gemini, Qdrant)
│   ├── run_generator.py           # Unified runner (Transcribe, Process, Embed, Upload)
│   ├── testing.py                 # Offline/Online Roman Urdu CLI testing suite
│   ├── uploader.py                # Standalone batch Qdrant point upserter
│   ├── purge_mock_vectors.py      # Checkpoint cleaning & verification utility
│   ├── brain_data/                # Checkpoints (gemini_embeddings_checkpoint.npz, RAG JSONL)
│   ├── data/                      # Raw inputs (vids, audios, text, md, comments)
│   ├── second_brain_generator/    # Python engine package:
│   │   ├── transcriber/           # faster-whisper speech-to-text
│   │   ├── compiler/              # Chunker, classifier, and RAG builder
│   │   ├── embedder/              # Gemini 3072-dim multi-key rotating embedder
│   │   ├── evaluator/             # Offline similarity & benchmark evaluation
│   │   ├── responder/             # Roman Urdu prompt engine & fallback synthesizer
│   │   └── uploader/              # Qdrant batch point ingestion
│   └── tests/                     # Unit and E2E test suites
│
└── app/                           # THE PRODUCTION APPLICATION
    ├── backend/                   # FASTAPI BACKEND SERVER
    │   ├── src/
    │   │   ├── main.py            # FastAPI entry point & CORS configuration
    │   │   ├── routes.py          # REST & SSE streaming endpoints (/api/ask, /api/conversations)
    │   │   ├── rag.py             # Qdrant retrieval, source re-ranking & Roman Urdu prompts
    │   │   ├── llm.py             # Gemini streaming & key rotation client
    │   │   ├── chat_database.py   # SQLAlchemy session, conversations & feedback models
    │   │   ├── voice_service.py   # ElevenLabs voice facade with persistent disk caching
    │   │   ├── elevenlabs_service.py # ElevenLabs TTS & ConvAI client with key pool rotation
    │   │   └── config.py          # App settings & environment resolution
    │   ├── requirements.txt       # Backend Python dependencies
    │   └── tests/                 # Backend unit & integration tests
    │
    └── frontend/                  # REACT 18 + VITE + MOTION FRONTEND
        ├── src/
        │   ├── components/        # UI & Chat components
        │   │   ├── chat/          # SourceList, VideoRecommendations, MetaRow
        │   │   ├── ui/            # AudioWaveform, CopyButton, ScrollToBottomButton
        │   │   ├── voice/         # VoiceCallModal, VoiceVisualizer, VoiceControlsDock
        │   │   ├── Composer.tsx   # Auto-resizing textarea with keyboard shortcuts
        │   │   ├── EmptyState.tsx # Framer motion animated ambient aura & topic cards
        │   │   └── MessageBubble  # Streaming message renderer with copy & TTS audio
        │   ├── hooks/             # useChat (SSE), useVoiceCall, useVoicePlayer, useTheme
        │   └── lib/               # Supabase authentication client
        ├── package.json           # Node dependencies (Motion, Tailwind, Phosphor Icons)
        └── tailwind.config.js     # Luxury dark-mode design tokens
```

---

## Getting Started

### 1. Prerequisites
- **Python 3.10+** (Python 3.13 recommended)
- **Node.js 18+** and **pnpm** (`pnpm -v`)
- **uv** (recommended for blazingly fast Python package management)
- **Qdrant Vector Database** (Local Docker on port `6333` or Qdrant Cloud)
- **Google Gemini API Key(s)**

---

### 2. Environment Setup

#### Unified Python Virtual Environment (Root)
Create and activate the virtual environment at the repository root:

```powershell
# Using uv (recommended)
uv venv .venv
.\.venv\Scripts\activate

# Install all backend and pipeline dependencies
uv pip install -r brain_generator/requirements.txt
uv pip install -r app/backend/requirements.txt
```

#### Frontend Setup
```powershell
cd app/frontend
pnpm install
```

#### Environment Variables
Create `.env` in `brain_generator/` and `app/backend/`:
```env
GEMINI_API_KEY=AIzaSy...
GEMINI_API_KEY_1=AIzaSy...
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=second_brain
```

---

### 3. Usage & Execution

#### A. Roman Urdu CLI Testing Pipeline
Test your Second Brain immediately from the command line:

```powershell
# Ask any question directly in Roman Urdu:
python testing.py "Karachi mein street crime se bachne ke kya tareeqay hain?"

# Run interactive conversation loop:
python testing.py --interactive

# Run automated benchmark suite:
python testing.py --demo

# Force offline mode (deterministic mock without consuming API quota):
python testing.py --offline "CS degree zaroori hai ya skills?"
```

#### B. Running the Data Generator (ETL Factory)
```powershell
cd brain_generator

# 1. Transcribe media -> Compile RAG records -> Embed into 3072-dim vectors:
python run_generator.py

# 2. Batch upload 13,400+ vectors into Qdrant:
python uploader.py --batch-size 100
```

#### C. Running the Full-Stack Application

**Start the FastAPI Backend:**
```powershell
# Option 1: If you are inside the app/backend directory:
cd app/backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Option 2: From the project root:
uvicorn src.main:app --app-dir app/backend --host 0.0.0.0 --port 8000 --reload
# (or: .\.venv\Scripts\uvicorn app.backend.src.main:app --host 0.0.0.0 --port 8000 --reload)
```
API documentation will be accessible at: `http://localhost:8000/docs`

**Start the React 18 Frontend:**
```powershell
cd app/frontend
pnpm dev
```
Interactive Web UI will be live at: `http://localhost:5173`

---

## Key Highlights

- **Cross-Lingual Bridge**: Seamlessly connects user queries in Roman Urdu or English to 13,400+ podcast chunks transcribed in Devanagari Hindi using Gemini's 3072-dimensional vector space.
- **Maximal Long-Form Roman Urdu Responses**: Structured 5-stage responses covering unfiltered takes, societal analysis, podcast excerpts, actionable advice, and final takeaways.
- **Motion Primitives & Luxury UI**: Framer motion-powered ambient glow, fluid springs, audio waveforms, and responsive sidebars designed for high-aesthetic AI experiences.
- **ElevenLabs Voice Mode**: Real-time 1-on-1 speech-to-speech calls with natural barge-in and keyboard fallback, plus a one-click **Listen** button on every generated response to hear the creator's voice in authentic Roman Urdu via ElevenLabs Text-to-Speech and optional Conversational AI agent integration.

---

## License
MIT
