"""Multi-source readers for transcripts, markdown notes, text notes, and comment exports."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)


def read_transcripts(transcripts_dir: Path) -> List[Dict[str, Any]]:
    """Reads all non-empty .txt files from the transcripts directory."""
    items = []
    if not transcripts_dir.exists() or not transcripts_dir.is_dir():
        return items

    for p in sorted(transcripts_dir.rglob("*.txt")):
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            items.append({
                "source_type": "video_transcript",
                "file_path": str(p),
                "file_name": p.name,
                "text": content,
                "question": None,
                "date": None,
                "extra": {"stem": p.stem},
            })
        except Exception as e:
            logger.warning("Error reading transcript %s: %s", p, e)

    return items


def read_text_notes(text_dir: Path) -> List[Dict[str, Any]]:
    """Reads all .txt files from data/text/."""
    items = []
    if not text_dir.exists() or not text_dir.is_dir():
        return items

    for p in sorted(text_dir.rglob("*.txt")):
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            items.append({
                "source_type": "text_note",
                "file_path": str(p),
                "file_name": p.name,
                "text": content,
                "question": None,
                "date": None,
                "extra": {"stem": p.stem},
            })
        except Exception as e:
            logger.warning("Error reading text note %s: %s", p, e)

    return items


def read_markdown_notes(md_dir: Path) -> List[Dict[str, Any]]:
    """Reads all .md files from data/md/, extracting optional YAML frontmatter."""
    items = []
    if not md_dir.exists() or not md_dir.is_dir():
        return items

    for p in sorted(md_dir.rglob("*.md")):
        if not p.is_file():
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                continue

            metadata = {}
            body = raw

            # Check for YAML frontmatter between --- and ---
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        if isinstance(frontmatter, dict):
                            metadata = frontmatter
                        body = parts[2].strip()
                    except Exception:
                        # Fallback: treat whole file as body if frontmatter is corrupted
                        body = raw

            if not body:
                continue

            date_val = str(metadata.get("date")) if metadata.get("date") else None
            items.append({
                "source_type": "markdown_note",
                "file_path": str(p),
                "file_name": p.name,
                "text": body,
                "question": None,
                "date": date_val,
                "extra": {"frontmatter": metadata, "stem": p.stem},
            })
        except Exception as e:
            logger.warning("Error reading markdown note %s: %s", p, e)

    return items


def _is_creator(author: Optional[str], creator_handles: List[str]) -> bool:
    if not author:
        return False
    norm_author = author.lower().strip().lstrip("@")
    for handle in creator_handles:
        norm_h = handle.lower().strip().lstrip("@")
        if norm_author == norm_h:
            return True
    return False


def read_comments_json(comments_dir: Path, creator_handles: List[str]) -> List[Dict[str, Any]]:
    """Reads comment trees and QA pairs from data/comments/*.json."""
    items = []
    if not comments_dir.exists() or not comments_dir.is_dir():
        return items

    for p in sorted(comments_dir.rglob("*.json")):
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue

                    # Direct Q&A format
                    q = entry.get("question") or entry.get("question_text") or entry.get("q")
                    r = entry.get("reply") or entry.get("reply_text") or entry.get("answer") or entry.get("a")
                    if r and str(r).strip():
                        items.append({
                            "source_type": "social_comment_reply",
                            "file_path": str(p),
                            "file_name": p.name,
                            "text": str(r).strip(),
                            "question": str(q).strip() if q else None,
                            "date": entry.get("date") or entry.get("timestamp"),
                            "extra": {k: v for k, v in entry.items() if k not in ["question", "reply", "text", "q", "a"]},
                        })

                    # Thread format with replies
                    replies = entry.get("replies", [])
                    if isinstance(replies, list):
                        parent_q = entry.get("text") or entry.get("comment") or q
                        for rep in replies:
                            if isinstance(rep, dict):
                                author = rep.get("author") or rep.get("user")
                                rep_text = rep.get("text") or rep.get("reply")
                                if rep_text and (_is_creator(author, creator_handles) or not creator_handles):
                                    items.append({
                                        "source_type": "social_comment_reply",
                                        "file_path": str(p),
                                        "file_name": p.name,
                                        "text": str(rep_text).strip(),
                                        "question": str(parent_q).strip() if parent_q else None,
                                        "date": rep.get("date") or rep.get("timestamp"),
                                        "extra": {"author": author},
                                    })
        except Exception as e:
            logger.warning("Error reading comment json %s: %s", p, e)

    return items


def read_comments_csv(comments_dir: Path) -> List[Dict[str, Any]]:
    """Reads comment and FAQ exports from data/comments/*.csv."""
    items = []
    if not comments_dir.exists() or not comments_dir.is_dir():
        return items

    for p in sorted(comments_dir.rglob("*.csv")):
        if not p.is_file():
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue

                for row in reader:
                    # Match question header
                    q_col = next((k for k in row if k and k.lower() in ["question", "question_text", "prompt", "q"]), None)
                    r_col = next((k for k in row if k and k.lower() in ["reply", "answer", "reply_text", "response", "a", "text"]), None)
                    d_col = next((k for k in row if k and k.lower() in ["date", "timestamp", "created_at"]), None)

                    r_val = row.get(r_col, "").strip() if r_col else ""
                    if not r_val:
                        continue

                    q_val = row.get(q_col, "").strip() if q_col else None
                    d_val = row.get(d_col, "").strip() if d_col else None

                    items.append({
                        "source_type": "social_comment_reply",
                        "file_path": str(p),
                        "file_name": p.name,
                        "text": r_val,
                        "question": q_val if q_val else None,
                        "date": d_val if d_val else None,
                        "extra": {k: v for k, v in row.items() if k not in [q_col, r_col, d_col]},
                    })
        except Exception as e:
            logger.warning("Error reading comments csv %s: %s", p, e)

    return items
