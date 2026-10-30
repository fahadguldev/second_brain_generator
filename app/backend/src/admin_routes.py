import hashlib
import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from src.admin_auth import AdminUser, require_admin
from src.chat_database import Conversation, IngestionJob, KnowledgeChunk, KnowledgeItem, Message, get_db, utcnow
from src.database import delete_points, get_collection_stats
from src.ingestion import run_batch_ingestion_jobs, run_ingestion_job
from src.classifier import determine_domain

router = APIRouter(prefix="/api/admin", tags=["admin"])
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


class KnowledgeCreate(BaseModel):
    question: Optional[str] = Field(default=None, max_length=10_000)
    content: str = Field(min_length=1, max_length=500_000)
    source_type: str = Field(default="chat", max_length=32)
    source_message_id: Optional[str] = None


class KnowledgeUpdate(BaseModel):
    question: Optional[str] = Field(default=None, max_length=10_000)
    content: str = Field(min_length=1, max_length=500_000)


class QuickActionPayload(BaseModel):
    question: Optional[str] = Field(default=None, max_length=10_000)
    content: str = Field(min_length=1, max_length=500_000)
    source_type: str = Field(default="chat", max_length=32)
    source_message_id: Optional[str] = None
    action: str = Field(default="draft")  # "draft", "approve", "push"


class KnowledgeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    source_type: str
    topics: Optional[list[str]] = None
    domain: str = "professional"
    source_message_id: Optional[str]
    question: Optional[str]
    content: str
    status: str
    chunks_count: Optional[int] = 0
    created_by: str
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime]

    @model_validator(mode="after")
    def set_domain(self) -> "KnowledgeResponse":
        """Expose the domain derived from the item's classified topics."""
        self.domain = determine_domain(self.topics or [])
        return self


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    knowledge_item_id: str
    status: str
    current_step: Optional[str] = "queued"
    chunks_total: int
    chunks_indexed: int
    logs: Optional[list] = []
    error: Optional[str]
    created_at: datetime
    finished_at: Optional[datetime]


class BatchIngestRequest(BaseModel):
    item_ids: Optional[list[str]] = None


class MessagePush(BaseModel):
    source_message_id: str
    question: Optional[str] = Field(default=None, max_length=10_000)
    content: str = Field(min_length=1, max_length=500_000)


class PushMessagesRequest(BaseModel):
    messages: list[MessagePush] = Field(min_length=1, max_length=500)
    source_type: str = Field(default="chat", max_length=32)


class BatchIngestResponse(BaseModel):
    queued_count: int
    job_ids: list[str]
    message: str


def _purge_knowledge_item(db: Session, item: KnowledgeItem) -> None:
    """Delete a knowledge item together with its chunks, Qdrant vectors and jobs."""
    chunks = db.scalars(
        select(KnowledgeChunk).where(KnowledgeChunk.knowledge_item_id == item.id)
    ).all()
    qdrant_ids = [c.qdrant_point_id for c in chunks if c.qdrant_point_id]
    if qdrant_ids:
        try:
            delete_points(qdrant_ids)
        except Exception as exc:
            print(f"Error purging Qdrant points during delete: {exc}")

    db.query(KnowledgeChunk).filter(KnowledgeChunk.knowledge_item_id == item.id).delete()
    db.query(IngestionJob).filter(IngestionJob.knowledge_item_id == item.id).delete()
    db.delete(item)


def _purge_knowledge_for_messages(db: Session, messages: list[Message]) -> int:
    """Remove every knowledge item sourced from these chat messages so nothing
    stays in brain memory after the chat content is deleted."""
    if not messages:
        return 0

    message_ids = [m.id for m in messages]
    items = list(
        db.scalars(
            select(KnowledgeItem).where(KnowledgeItem.source_message_id.in_(message_ids))
        ).all()
    )
    known = {item.id for item in items}

    # Legacy/manual chat items carry no source_message_id; match those by content.
    contents = {(m.content or "").strip() for m in messages}
    contents.discard("")
    if contents:
        orphans = db.scalars(
            select(KnowledgeItem).where(
                KnowledgeItem.source_type == "chat",
                KnowledgeItem.source_message_id.is_(None),
            )
        ).all()
        for item in orphans:
            if item.id in known:
                continue
            if (item.content or "").strip() in contents:
                items.append(item)
                known.add(item.id)

    for item in items:
        _purge_knowledge_item(db, item)
    return len(items)


@router.get("/conversations")
def all_conversations(
    admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)
):
    rows = db.scalars(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.updated_at.desc())
        .limit(200)
    ).all()

    # Map source_message_id -> KnowledgeItem status
    knowledge_items = db.scalars(
        select(KnowledgeItem).where(KnowledgeItem.source_message_id.isnot(None))
    ).all()

    # Identify items that have verified chunks in Qdrant
    pushed_item_ids = set()
    try:
        pushed_item_ids = set(
            db.scalars(select(KnowledgeChunk.knowledge_item_id).distinct()).all()
        )
    except Exception as exc:
        print(f"Notice: unable to query KnowledgeChunk points: {exc}")

    # Build knowledge_map prioritizing indexed items > approved > draft
    knowledge_map = {}
    content_pushed_map = {}
    for k in knowledge_items:
        is_item_indexed = (k.status == "indexed") or (k.id in pushed_item_ids)
        if k.source_message_id:
            msg_id = k.source_message_id
            existing = knowledge_map.get(msg_id)
            if not existing:
                knowledge_map[msg_id] = k
            elif is_item_indexed:
                knowledge_map[msg_id] = k
            elif existing.status != "indexed" and k.status == "approved":
                knowledge_map[msg_id] = k

        if is_item_indexed and k.content and k.content.strip():
            content_pushed_map[k.content.strip()] = k

    def resolve_item(msg):
        item = knowledge_map.get(msg.id)
        if not item and msg.content and msg.content.strip():
            item = content_pushed_map.get(msg.content.strip())
        return item

    return [{
        "id": row.id,
        "user_id": row.user_id,
        "title": row.title,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "messages": [{
            "id": msg.id,
            "role": msg.role,
            "text": msg.content or "",
            "pipeline_status": resolve_item(msg).status if resolve_item(msg) else None,
            "is_pushed": bool(
                resolve_item(msg)
                and (resolve_item(msg).status == "indexed" or resolve_item(msg).id in pushed_item_ids)
            ),
            "knowledge_item_id": resolve_item(msg).id if resolve_item(msg) else None,
        } for msg in (row.messages or [])],
    } for row in rows]


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a whole chat: messages, knowledge items, vectors and jobs."""
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = list(conv.messages)
    purged = _purge_knowledge_for_messages(db, messages)

    db.delete(conv)
    db.commit()

    return {
        "status": "ok",
        "deleted_conversation_id": conversation_id,
        "deleted_messages": len(messages),
        "purged_knowledge_items": purged,
    }


@router.delete("/messages/{message_id}")
def delete_message(
    message_id: str,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a single chat message. A user message also removes its answer."""
    msg = db.get(Message, message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    conv = msg.conversation
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    ordered = list(conv.messages)
    targets = [msg]
    if msg.role == "user":
        idx = next((i for i, m in enumerate(ordered) if m.id == message_id), None)
        if idx is not None and idx + 1 < len(ordered):
            following = ordered[idx + 1]
            if following.role == "assistant":
                targets.append(following)

    purged = _purge_knowledge_for_messages(db, targets)
    target_ids = {m.id for m in targets}
    for target in targets:
        db.delete(target)

    remaining = db.scalar(
        select(func.count(Message.id)).where(
            Message.conversation_id == conv.id,
            Message.id.notin_(target_ids),
        )
    )

    deleted_conversation_id = None
    if not remaining:
        db.delete(conv)
        deleted_conversation_id = conv.id
    else:
        conv.updated_at = utcnow()

    db.commit()

    return {
        "status": "ok",
        "deleted_message_ids": sorted(target_ids),
        "purged_knowledge_items": purged,
        "deleted_conversation_id": deleted_conversation_id,
    }


@router.get("/qdrant/stats")
def qdrant_stats(admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)):
    """Fetch live vector count and statistics from Qdrant, including verified pushed records count."""
    stats = get_collection_stats()
    pushed_records_count = db.scalar(
        select(func.count(KnowledgeItem.id)).where(KnowledgeItem.status == "indexed")
    ) or 0
    pushed_chunks_count = db.scalar(
        select(func.count(KnowledgeChunk.id))
    ) or 0
    stats["pushed_records_count"] = pushed_records_count
    stats["pushed_chunks_count"] = pushed_chunks_count
    return stats


@router.get("/knowledge", response_model=list[KnowledgeResponse])
def list_knowledge(
    status: Optional[str] = None,
    admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db),
):
    query = select(KnowledgeItem).order_by(KnowledgeItem.updated_at.desc())
    if status:
        query = query.where(KnowledgeItem.status == status)
    items = list(db.scalars(query).all())

    # Attach chunk counts for verification
    if items:
        counts = dict(
            db.execute(
                select(KnowledgeChunk.knowledge_item_id, func.count(KnowledgeChunk.id))
                .where(KnowledgeChunk.knowledge_item_id.in_([item.id for item in items]))
                .group_by(KnowledgeChunk.knowledge_item_id)
            ).all()
        )
        for item in items:
            item.chunks_count = counts.get(item.id, 0)
    return items


@router.post("/knowledge", response_model=KnowledgeResponse, status_code=201)
def create_knowledge(
    payload: KnowledgeCreate, admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)
):
    if payload.source_message_id and not db.get(Message, payload.source_message_id):
        raise HTTPException(status_code=404, detail="Source message not found")
    if payload.source_message_id:
        duplicate = db.scalar(select(KnowledgeItem).where(
            KnowledgeItem.source_type == payload.source_type,
            KnowledgeItem.source_message_id == payload.source_message_id,
        ))
        if duplicate:
            raise HTTPException(status_code=409, detail="This source is already in review or pushed")
    item = KnowledgeItem(
        id=str(uuid4()), source_type=payload.source_type,
        source_message_id=payload.source_message_id, question=payload.question,
        content=payload.content.strip(), created_by=admin.email,
    )
    db.add(item)
    db.commit()
    return item


@router.post("/knowledge/quick-action")
def quick_action(
    payload: QuickActionPayload,
    tasks: BackgroundTasks,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Directly create or update a knowledge item from chat, with instant actions:
    - 'draft': saves as draft in review
    - 'approve': saves and marks approved (goes to Ready to Push)
    - 'push': saves, approves, and immediately launches ingestion into Qdrant
    """
    item = None
    if payload.source_message_id:
        item = db.scalar(select(KnowledgeItem).where(
            KnowledgeItem.source_message_id == payload.source_message_id,
        ))

    if not item:
        item = KnowledgeItem(
            id=str(uuid4()),
            source_type=payload.source_type,
            source_message_id=payload.source_message_id,
            question=payload.question,
            content=payload.content.strip(),
            created_by=admin.email,
        )
        db.add(item)
    else:
        item.question = payload.question
        item.content = payload.content.strip()
        item.updated_at = utcnow()

    job_data = None
    if payload.action == "approve":
        item.status = "approved"
        item.approved_at = utcnow()
    elif payload.action == "push":
        item.status = "approved"
        item.approved_at = utcnow()
        db.commit()

        # Check for existing active job or create new one
        job = db.scalar(select(IngestionJob).where(
            IngestionJob.knowledge_item_id == item.id,
            IngestionJob.status.in_(["queued", "indexing"]),
        ))
        if not job:
            job = IngestionJob(
                id=str(uuid4()),
                knowledge_item_id=item.id,
                current_step="queued",
                logs=[{"timestamp": utcnow().strftime("%H:%M:%S"), "step": "queued", "level": "info", "message": "Direct push queued from chat"}],
            )
            db.add(job)
            db.commit()
            tasks.add_task(run_ingestion_job, job.id)
        job_data = JobResponse.model_validate(job)
    else:
        item.status = "draft"
        item.approved_at = None

    db.commit()
    return {
        "item": KnowledgeResponse.model_validate(item),
        "job": job_data,
        "message": f"Successfully performed '{payload.action}'",
    }


@router.put("/knowledge/{item_id}", response_model=KnowledgeResponse)
def update_knowledge(
    item_id: str, payload: KnowledgeUpdate,
    admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db),
):
    item = db.get(KnowledgeItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    item.question = payload.question
    item.content = payload.content.strip()
    item.status = "draft"
    item.approved_at = None
    item.updated_at = utcnow()
    db.commit()
    return item


@router.post("/knowledge/{item_id}/approve", response_model=KnowledgeResponse)
def approve_knowledge(
    item_id: str, admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)
):
    item = db.get(KnowledgeItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    item.status = "approved"
    item.approved_at = utcnow()
    item.updated_at = utcnow()
    db.commit()
    return item


@router.post("/knowledge/{item_id}/revert", response_model=KnowledgeResponse)
def revert_knowledge(
    item_id: str, admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)
):
    """Revert an approved or indexed item back to draft review."""
    item = db.get(KnowledgeItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    item.status = "draft"
    item.approved_at = None
    item.updated_at = utcnow()
    db.commit()
    return item


@router.delete("/knowledge/{item_id}")
def delete_knowledge(
    item_id: str, admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)
):
    """Delete a knowledge item and its associated chunks and jobs."""
    item = db.get(KnowledgeItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")

    _purge_knowledge_item(db, item)
    db.commit()
    return {"status": "ok", "deleted_item_id": item_id}


@router.post("/knowledge/{item_id}/ingest", response_model=JobResponse, status_code=202)
def ingest_knowledge(
    item_id: str, tasks: BackgroundTasks,
    admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db),
):
    item = db.get(KnowledgeItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    if item.status != "approved":
        raise HTTPException(status_code=409, detail="Approve the item before ingestion")
    active = db.scalar(select(IngestionJob).where(
        IngestionJob.knowledge_item_id == item_id,
        IngestionJob.status.in_(["queued", "indexing"]),
    ))
    if active:
        return active
    job = IngestionJob(
        id=str(uuid4()),
        knowledge_item_id=item_id,
        current_step="queued",
        logs=[{"timestamp": utcnow().strftime("%H:%M:%S"), "step": "queued", "level": "info", "message": "Job queued for processing"}],
    )
    db.add(job)
    db.commit()
    tasks.add_task(run_ingestion_job, job.id)
    return job


@router.post("/knowledge/batch-ingest", response_model=BatchIngestResponse, status_code=202)
def batch_ingest_knowledge(
    payload: BatchIngestRequest,
    tasks: BackgroundTasks,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Ingest multiple approved items at once with safe batching to handle large volumes and rate limits.
    If payload.item_ids is empty/omitted, all approved items will be processed.
    """
    query = select(KnowledgeItem).where(KnowledgeItem.status == "approved")
    if payload.item_ids:
        query = query.where(KnowledgeItem.id.in_(payload.item_ids))
    
    approved_items = db.scalars(query).all()
    if not approved_items:
        raise HTTPException(status_code=400, detail="No approved items found to push to vector DB")

    created_jobs = []
    job_ids = []

    for item in approved_items:
        # Check if an active job already exists
        active = db.scalar(select(IngestionJob).where(
            IngestionJob.knowledge_item_id == item.id,
            IngestionJob.status.in_(["queued", "indexing"]),
        ))
        if active:
            job_ids.append(active.id)
            continue

        job = IngestionJob(
            id=str(uuid4()),
            knowledge_item_id=item.id,
            current_step="queued",
            logs=[{"timestamp": utcnow().strftime("%H:%M:%S"), "step": "queued", "level": "info", "message": "Batch ingestion queued"}],
        )
        db.add(job)
        created_jobs.append(job)
        job_ids.append(job.id)

    db.commit()

    if job_ids:
        tasks.add_task(run_batch_ingestion_jobs, job_ids, 5)

    return BatchIngestResponse(
        queued_count=len(job_ids),
        job_ids=job_ids,
        message=f"Queued {len(job_ids)} item(s) for batch embedding and vector ingestion",
    )


@router.post("/knowledge/push-messages", status_code=202)
def push_messages(
    payload: PushMessagesRequest,
    tasks: BackgroundTasks,
    admin: AdminUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Create or update knowledge items for the given chat messages, mark them
    approved, and queue a single batched ingestion for all of them at once.
    Used when the admin curates a whole conversation in the chat editor and
    then pushes the entire chat to Qdrant.
    """
    seen_message_ids = set()
    for entry in payload.messages:
        if not db.get(Message, entry.source_message_id):
            raise HTTPException(
                status_code=404,
                detail=f"Source message {entry.source_message_id[:8]} not found",
            )
        if entry.source_message_id in seen_message_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate message {entry.source_message_id[:8]} in payload",
            )
        seen_message_ids.add(entry.source_message_id)

    job_ids: list[str] = []

    for entry in payload.messages:
        item = db.scalar(select(KnowledgeItem).where(
            KnowledgeItem.source_message_id == entry.source_message_id,
        ))

        if not item:
            item = KnowledgeItem(
                id=str(uuid4()),
                source_type=payload.source_type,
                source_message_id=entry.source_message_id,
                question=entry.question,
                content=entry.content.strip(),
                created_by=admin.email,
            )
            db.add(item)
            # autoflush is off and there is no ORM relationship between
            # KnowledgeItem and IngestionJob, so SQLAlchemy may insert the
            # job first and hit the FK constraint. Flush the parent row now.
            db.flush()
        else:
            item.question = entry.question
            item.content = entry.content.strip()
            item.updated_at = utcnow()

        item.status = "approved"
        item.approved_at = utcnow()

        active = db.scalar(select(IngestionJob).where(
            IngestionJob.knowledge_item_id == item.id,
            IngestionJob.status.in_(["queued", "indexing"]),
        ))
        if active:
            job_ids.append(active.id)
            continue

        job = IngestionJob(
            id=str(uuid4()),
            knowledge_item_id=item.id,
            current_step="queued",
            logs=[{
                "timestamp": utcnow().strftime("%H:%M:%S"),
                "step": "queued",
                "level": "info",
                "message": "Chat batch push queued",
            }],
        )
        db.add(job)
        job_ids.append(job.id)

    db.commit()

    if job_ids:
        tasks.add_task(run_batch_ingestion_jobs, job_ids, 5)

    return {
        "queued_count": len(job_ids),
        "job_ids": job_ids,
        "message": f"Queued {len(job_ids)} chat message(s) for embedding and vector ingestion",
    }


@router.post("/uploads", response_model=KnowledgeResponse, status_code=201)
async def upload_text(
    file: UploadFile = File(...),
    admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db),
):
    filename = os.path.basename(file.filename or "upload.txt")
    if not filename.lower().endswith((".txt", ".md")):
        raise HTTPException(status_code=415, detail="Only .txt and .md files are supported")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 2 MB limit")
    try:
        content = raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded") from exc
    # Remove null bytes to ensure compatibility with PostgreSQL text fields
    content = content.replace("\x00", "")
    if not content:
        raise HTTPException(status_code=422, detail="File is empty")
    digest = hashlib.sha256(raw).hexdigest()
    # 36-character prefix guarantees compatibility with existing VARCHAR(36) column definitions in production PostgreSQL
    digest_key = digest[:36]
    duplicate = db.scalar(select(KnowledgeItem).where(
        KnowledgeItem.source_type == "upload",
        KnowledgeItem.source_message_id.in_([digest, digest_key]),
    ))
    if duplicate:
        raise HTTPException(status_code=409, detail="This file has already been uploaded")
    now = utcnow()
    item = KnowledgeItem(
        id=str(uuid4()),
        source_type="upload",
        source_message_id=digest_key,
        question=filename[:160],
        content=content,
        status="draft",
        created_by=admin.email,
        created_at=now,
        updated_at=now,
    )
    try:
        db.add(item)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error saving uploaded file: {str(exc)}") from exc
    return item


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(admin: AdminUser = Depends(require_admin), db: Session = Depends(get_db)):
    return db.scalars(select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(100)).all()


class CategoryPayload(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    keywords: list[str] = Field(default_factory=list)
    domain: str = Field(default="professional")


class CategoryResponse(BaseModel):
    name: str
    keywords: list[str]
    domain: str


@router.get("/categories", response_model=list[CategoryResponse])
def get_categories(admin: AdminUser = Depends(require_admin)):
    from src.classifier import list_categories
    return list_categories()


@router.post("/categories", response_model=CategoryResponse)
def create_or_update_category(
    payload: CategoryPayload,
    admin: AdminUser = Depends(require_admin),
):
    from src.classifier import save_category
    clean_name = payload.name.strip().lower()
    if not clean_name:
        raise HTTPException(status_code=422, detail="Category name cannot be empty")
    return save_category(name=clean_name, keywords=payload.keywords, domain=payload.domain)


@router.delete("/categories/{name}")
def delete_category(
    name: str,
    admin: AdminUser = Depends(require_admin),
):
    from src.classifier import remove_category
    success = remove_category(name)
    if not success:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": f"Category '{name}' deleted successfully"}


LEGACY_SOURCE_TYPES = ("tiktok_dm", "tiktok_comment_reply", "linkedin_post", "video_transcript")


class LegacyPointResponse(BaseModel):
    id: str
    text: str
    source_type: str
    topics: list[str]
    domain: Optional[str] = None


class LegacyPointListResponse(BaseModel):
    points: list[LegacyPointResponse]
    next_offset: Optional[str] = None
    total_matched: int


class LegacyPointUpdate(BaseModel):
    topics: list[str] = Field(default_factory=list, max_length=24)
    domain: Optional[str] = Field(default=None, max_length=32)


def _parse_point_id(point_id: str) -> int | str:
    return int(point_id) if point_id.isdigit() else point_id


@router.get("/legacy-points", response_model=LegacyPointListResponse)
def list_legacy_points(
    source_type: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = 24,
    offset: Optional[str] = None,
    admin: AdminUser = Depends(require_admin),
):
    """Browse the legacy social-import corpus (TikTok DMs/comments, LinkedIn,
    video transcripts) that lives only in Qdrant, with scroll pagination."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    from src.database import get_client, get_collection_name

    limit = max(1, min(limit, 100))
    must = [FieldCondition(key="metadata.source.type", match=MatchAny(any=list(LEGACY_SOURCE_TYPES)))]
    if source_type and source_type in LEGACY_SOURCE_TYPES:
        must = [FieldCondition(key="metadata.source.type", match=MatchValue(value=source_type))]
    if topic:
        must.append(FieldCondition(key="metadata.topics", match=MatchValue(value=topic.strip().lower())))

    client = get_client()
    try:
        points, next_offset = client.scroll(
            collection_name=get_collection_name(),
            limit=limit,
            offset=_parse_point_id(offset) if offset else None,
            scroll_filter=Filter(must=must),
            with_payload=True,
            with_vectors=False,
        )
        total = client.count(
            collection_name=get_collection_name(),
            count_filter=Filter(must=must),
        ).count
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}") from exc

    items: list[LegacyPointResponse] = []
    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        source = metadata.get("source") or {}
        topics = metadata.get("topics")
        items.append(LegacyPointResponse(
            id=str(point.id),
            text=(payload.get("text") or "")[:1200],
            source_type=source.get("type") or "unknown",
            topics=topics if isinstance(topics, list) else [],
            domain=metadata.get("domain"),
        ))
    return LegacyPointListResponse(
        points=items,
        next_offset=str(next_offset) if next_offset is not None else None,
        total_matched=total,
    )


@router.put("/legacy-points/{point_id}")
def update_legacy_point(
    point_id: str,
    payload: LegacyPointUpdate,
    admin: AdminUser = Depends(require_admin),
):
    """Re-tag a legacy Qdrant point: overwrite metadata.topics / metadata.domain
    (shallow-merged, source/classification metadata stays intact)."""
    from src.classifier import determine_domain
    from src.database import get_client, get_collection_name

    topics = [t.strip().lower() for t in payload.topics if t.strip()]
    domain = (payload.domain or "").strip().lower() or determine_domain(topics)
    if domain not in ("personal", "professional"):
        raise HTTPException(status_code=422, detail="domain must be 'personal' or 'professional'")

    client = get_client()
    try:
        existing = client.retrieve(
            collection_name=get_collection_name(),
            ids=[_parse_point_id(point_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Point not found")
        client.set_payload(
            collection_name=get_collection_name(),
            payload={"topics": topics, "domain": domain},
            points=[_parse_point_id(point_id)],
            key="metadata",
            wait=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}") from exc
    return {"id": point_id, "topics": topics, "domain": domain}
