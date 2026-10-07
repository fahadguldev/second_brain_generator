"""Whisper-based media transcription engine with incremental idempotency and VAD filtering."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import BrainConfig

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
SUPPORTED_MEDIA_EXTS = SUPPORTED_VIDEO_EXTS | SUPPORTED_AUDIO_EXTS


class MediaTranscriber:
    """
    Transcribes audio and video files into text transcripts using faster-whisper.
    Supports VAD filtering, incremental skips, and configurable compute/device types.
    """

    def __init__(
        self,
        config: BrainConfig,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        self.config = config

        # Resolve parameters from config or arguments
        cfg_whisper = getattr(config, "whisper", None)
        self.model_name = (
            model_name
            or (cfg_whisper.model_size if cfg_whisper else None)
            or "small"
        )
        self.device = (
            device
            or (cfg_whisper.device if cfg_whisper else None)
            or ("cuda" if self._has_cuda() else "cpu")
        )
        self.compute_type = (
            compute_type
            or (cfg_whisper.compute_type if cfg_whisper else None)
            or ("float16" if self.device == "cuda" else "int8")
        )
        self.vad_filter = (
            cfg_whisper.vad_filter if cfg_whisper and hasattr(cfg_whisper, "vad_filter") else True
        )
        self.language = (
            cfg_whisper.language if cfg_whisper and hasattr(cfg_whisper, "language") else None
        )

        self._model = None

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except (ImportError, Exception):
            return False

    @property
    def model(self):
        """Lazy loader for faster-whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(
                    "Loading WhisperModel(%s, device=%s, compute_type=%s)",
                    self.model_name, self.device, self.compute_type
                )
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except ImportError:
                raise RuntimeError(
                    "faster-whisper is not installed. Install with `pip install faster-whisper`"
                )
        return self._model

    def transcribe_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        force: bool = False,
    ) -> Optional[Path]:
        """
        Transcribe a single media file into a text file.

        Args:
            input_path: Source media file path (.mp4, .mov, .mkv, .mp3, .wav, .m4a).
            output_path: Destination .txt transcript path.
            force: Retranscribe even if output already exists with content.

        Returns:
            Path to written transcript, or None if skipped/failed.
        """
        inp = Path(input_path).resolve()
        out = Path(output_path).resolve()

        if not inp.exists():
            logger.warning("Input file does not exist: %s", inp)
            return None

        # Check idempotency: skip if non-empty output already exists
        if out.exists() and out.stat().st_size > 0 and not force:
            logger.info("Skipping existing transcript: %s", out.name)
            return None

        # Ensure destination directory exists
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Transcribing %s -> %s", inp.name, out.name)
            segments, info = self.model.transcribe(
                str(inp),
                language=self.language,
                vad_filter=self.vad_filter,
            )

            lines = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
            transcript_text = "\n".join(lines)

            # Atomic write
            temp_out = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
            temp_out.write_text(transcript_text, encoding="utf-8")
            temp_out.replace(out)

            logger.info("Successfully transcribed %s (%d lines)", inp.name, len(lines))
            return out
        except Exception as e:
            logger.error("Failed to transcribe %s: %s", inp, e)
            raise

    def transcribe_all(self, force: bool = False) -> Dict[str, int]:
        """
        Scans data/vids and data/audios and transcribes all discovered media files.

        Returns:
            Dictionary with counts: {"processed": int, "skipped": int, "failed": int}
        """
        stats = {"processed": 0, "skipped": 0, "failed": 0}

        paths = self.config.paths
        base_dir = Path(getattr(self.config, "base_dir", Path.cwd()))
        transcripts_dir = (base_dir / paths.get("transcripts_dir", "brain_data/text")).resolve()
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        media_files: List[Path] = []
        for key in ["vids_dir", "audios_dir"]:
            d_path = paths.get(key)
            if d_path:
                full_dir = (base_dir / d_path).resolve()
                if full_dir.exists() and full_dir.is_dir():
                    for p in full_dir.rglob("*"):
                        if p.is_file() and p.suffix.lower() in SUPPORTED_MEDIA_EXTS:
                            media_files.append(p)

        logger.info("Found %d media files for transcription", len(media_files))

        for media in media_files:
            out_file = transcripts_dir / f"{media.stem}.txt"
            if out_file.exists() and out_file.stat().st_size > 0 and not force:
                stats["skipped"] += 1
                continue

            try:
                result = self.transcribe_file(media, out_file, force=force)
                if result:
                    stats["processed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:
                logger.error("Error processing %s: %s", media, exc)
                stats["failed"] += 1

        logger.info("Transcription completed: %s", stats)
        return stats
