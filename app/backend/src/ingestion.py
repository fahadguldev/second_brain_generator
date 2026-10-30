import hashlib
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client.models import PointStruct
from sqlalchemy.orm import Session

from src.chat_database import IngestionJob, KnowledgeChunk, KnowledgeItem, SessionLocal, utcnow
from src.database import delete_points, insert_points
from src.embeddings import embedding_generator
from src.classifier import classify_topics, determine_domain, detect_language


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _append_job_log(db: Session, job: IngestionJob, step: str, message: str, level: str = "info") -> None:
    """Append a structured timestamped log entry to the job and update current_step."""
    timestamp = utcnow().strftime("%H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "step": step,
        "level": level,
        "message": message,
    }
    existing_logs = list(job.logs or [])
    existing_logs.append(entry)
    job.logs = existing_logs
    job.current_step = step
    db.commit()


def process_single_job(db: Session, job_id: str) -> bool:
    """Process an ingestion job step by step with rich visual flow tracking and logs."""
    job = db.get(IngestionJob, job_id)
    if not job:
        return False
    
    item = db.get(KnowledgeItem, job.knowledge_item_id)
    if not item or item.status != "approved":
        job.status = "failed"
        job.current_step = "failed"
        job.error = "Only approved knowledge can be indexed"
        job.finished_at = utcnow()
        _append_job_log(db, job, "failed", "Validation failed: Item must be approved before ingestion", level="error")
        db.commit()
        return False

    try:
        # Step 1: Queued / Initializing
        job.status = "indexing"
        job.current_step = "queued"
        _append_job_log(db, job, "queued", f"Ingestion job started for item '{item.question or item.id[:8]}'")
        
        # Step 2: Chunking & Preprocessing
        job.current_step = "chunking"
        _append_job_log(db, job, "chunking", "Preparing canonical content and chunking text...")
        canonical = (
            f"Question: {item.question}\nAnswer: {item.content}"
            if item.source_type == "chat" and item.question else item.content
        )
        chunks = chunk_text(canonical)
        if not chunks:
            raise ValueError("Knowledge item has no indexable text after preprocessing")
        
        job.chunks_total = len(chunks)
        _append_job_log(db, job, "chunking", f"Text split into {len(chunks)} chunk(s) (1500 chars max, 200 overlap)")

        # Step 3: Generating Embeddings
        job.current_step = "embedding"
        _append_job_log(db, job, "embedding", f"Calling Gemini API to generate 3072-dim embeddings for {len(chunks)} chunk(s)...")
        vectors = embedding_generator.embed_batch(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedding count mismatch: expected {len(chunks)}, got {len(vectors)}")
        _append_job_log(db, job, "embedding", f"Successfully generated {len(vectors)} embedding vector(s)")

        # Step 4: Pushing to Qdrant Vector Database
        job.current_step = "pushing_qdrant"
        _append_job_log(db, job, "pushing_qdrant", f"Constructing Qdrant points and upserting {len(vectors)} point(s) to collection...")
        
        points = []
        rows = []
        all_topics_set = set()
        filename_hint = item.question if item.source_type == "upload" else ""
        question_hint = item.question if item.source_type != "upload" else None

        for position, (text, vector) in enumerate(zip(chunks, vectors)):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            point_id = str(uuid5(NAMESPACE_URL, f"knowledge:{item.id}:{position}:{digest}"))

            # Classify topics, domain, and language using brain_config.yaml
            topics = classify_topics(text=text, question=question_hint, filename=filename_hint)
            all_topics_set.update(topics)
            domain = determine_domain(topics)
            lang = detect_language(text)

            payload = {
                "text": text,
                "metadata": {
                    "source": {
                        "type": item.source_type,
                        "knowledge_item_id": item.id,
                        "file_name": filename_hint or None,
                    },
                    "approved_at": item.approved_at.isoformat() if item.approved_at else None,
                    "classification": {
                        "domain": domain,
                        "topics": topics,
                    },
                    "topics": topics,
                    "domain": domain,
                    "language": lang,
                    "chunk_index": position,
                    "chunks_total": len(chunks),
                },
            }

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))
            rows.append(KnowledgeChunk(
                id=str(uuid4()),
                knowledge_item_id=item.id,
                qdrant_point_id=point_id,
                position=position,
                content_hash=digest,
                text=text,
            ))

        if all_topics_set:
            _append_job_log(db, job, "classifying", f"Classified with topics: {', '.join(sorted(all_topics_set))} (domain: {domain})")
        item.topics = sorted(all_topics_set)

        old_rows = db.query(KnowledgeChunk).filter(KnowledgeChunk.knowledge_item_id == item.id).all()
        
        # Batch insert to Qdrant (in chunks of 50 if large)
        batch_size = 50
        for i in range(0, len(points), batch_size):
            pts_chunk = points[i:i + batch_size]
            if not insert_points(pts_chunk):
                raise RuntimeError(f"Qdrant rejected batch insertion at index {i}")

        _append_job_log(db, job, "pushing_qdrant", f"Qdrant confirmed upsert of {len(points)} vector point(s)")

        # Cleanup stale points if replaced
        new_ids = {row.qdrant_point_id for row in rows}
        stale_ids = [row.qdrant_point_id for row in old_rows if row.qdrant_point_id not in new_ids]
        if stale_ids:
            _append_job_log(db, job, "pushing_qdrant", f"Cleaning up {len(stale_ids)} stale vector(s)...")
            if not delete_points(stale_ids):
                _append_job_log(db, job, "pushing_qdrant", "Notice: Stale vectors could not be fully purged", level="warning")

        # Step 5: Finalizing & Completed
        db.query(KnowledgeChunk).filter(KnowledgeChunk.knowledge_item_id == item.id).delete()
        db.add_all(rows)
        
        job.chunks_indexed = len(rows)
        job.status = "indexed"
        job.current_step = "indexed"
        job.finished_at = utcnow()
        item.status = "indexed"
        item.updated_at = utcnow()
        _append_job_log(db, job, "indexed", f"Ingestion complete! {len(rows)} chunk(s) indexed into Qdrant & database.")
        db.commit()
        return True

    except Exception as exc:
        db.rollback()
        job = db.get(IngestionJob, job_id)
        if job:
            job.status = "failed"
            job.current_step = "failed"
            job.error = str(exc)[:2000]
            job.finished_at = utcnow()
            _append_job_log(db, job, "failed", f"Ingestion failed: {str(exc)}", level="error")
            db.commit()
        return False


def run_ingestion_job(job_id: str) -> None:
    """Execute a single ingestion job."""
    db = SessionLocal()
    try:
        process_single_job(db, job_id)
    finally:
        db.close()


def run_batch_ingestion_jobs(job_ids: list[str], batch_size: int = 5) -> None:
    """
    Execute multiple ingestion jobs in controlled batches.
    Handles edge cases like high volume, rate limits, and partial failures.
    """
    db = SessionLocal()
    try:
        total = len(job_ids)
        for i in range(0, total, batch_size):
            chunk_job_ids = job_ids[i:i + batch_size]
            for job_id in chunk_job_ids:
                try:
                    process_single_job(db, job_id)
                except Exception as e:
                    print(f"Error executing batch job {job_id}: {e}")
                # Short breathing room between individual items
                time.sleep(0.2)
            
            # Short pause between batches to protect against API rate limits
            if i + batch_size < total:
                time.sleep(1.0)
    finally:
        db.close()
