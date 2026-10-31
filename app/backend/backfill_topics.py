"""One-time backfill: give every Qdrant point a metadata.topics tag list.

Covers the legacy social-import corpus (TikTok DMs/comments, LinkedIn posts,
video transcripts) and any older chat/upload points indexed before topics were
stored. Uses the current brain_config.yaml lexicon so query-side classification
and point-side tags share one vocabulary.

Usage:
    python backfill_topics.py            # backfill points with missing/empty tags
    python backfill_topics.py --force    # re-classify every point
    python backfill_topics.py --limit 100
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from qdrant_client.models import Filter, FieldCondition, MatchAny

from src.classifier import classify_topics, determine_domain, get_topics_lexicon
from src.database import get_client, get_collection_name

SCROLL_BATCH = 200


def _needs_backfill(payload: Dict[str, Any], force: bool) -> bool:
    if force:
        return True
    metadata = payload.get("metadata") or {}
    topics = metadata.get("topics") or (payload.get("classification") or {}).get("topics")
    return not (isinstance(topics, list) and len(topics) > 0)


def _build_tags(payload: Dict[str, Any], lexicon_keys: set) -> Tuple[List[str], str]:
    metadata = payload.get("metadata") or {}
    text = payload.get("text") or ""
    source = metadata.get("source") or payload.get("source") or {}
    filename = source.get("file_name") or ""

    topics = set(classify_topics(text=text, filename=filename))
    # Keep legacy classifier topics that exist in the current lexicon too.
    legacy = (
        metadata.get("topics")
        or (metadata.get("classification") or {}).get("topics")
        or (payload.get("classification") or {}).get("topics")
        or []
    )
    topics.update(t for t in legacy if t in lexicon_keys)

    domain = (
        metadata.get("domain")
        or (metadata.get("classification") or {}).get("domain")
        or (payload.get("classification") or {}).get("domain")
    )
    if not domain:
        domain = determine_domain(sorted(topics))
    return sorted(topics), str(domain)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-classify even points that already have tags")
    parser.add_argument("--limit", type=int, default=0, help="stop after N updates (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = parser.parse_args()

    client = get_client()
    collection = get_collection_name()
    lexicon_keys = set(get_topics_lexicon().keys())

    scanned = 0
    skipped = 0
    # Group identical tag payloads so thousands of noise points cost a few calls.
    pending: Dict[Tuple[Tuple[str, ...], str], List[Any]] = {}

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            scanned += 1
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            if not _needs_backfill(payload, args.force):
                skipped += 1
                continue
            topics, domain = _build_tags(payload, lexicon_keys)
            pending.setdefault((tuple(topics), domain), []).append(point.id)
        if offset is None:
            break

    updates = sum(len(ids) for ids in pending.values())
    print(f"scanned={scanned} need_backfill={updates} already_tagged={skipped} unique_payloads={len(pending)}")

    if args.dry_run:
        for (topics, domain), ids in sorted(pending.items(), key=lambda kv: -len(kv[1]))[:15]:
            print(f"  {len(ids):5d} pts -> topics={list(topics)} domain={domain}")
        return

    done = 0
    for (topics, domain), ids in pending.items():
        if args.limit and done >= args.limit:
            break
        client.set_payload(
            collection_name=collection,
            payload={"topics": list(topics), "domain": domain},
            points=ids,
            key="metadata",
            wait=False,
        )
        done += len(ids)
    try:
        client._client.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    print(f"backfilled {done} point(s)")


if __name__ == "__main__":
    main()
