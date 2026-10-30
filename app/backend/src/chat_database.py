from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from src.config import settings


def _database_url() -> str:
    url = settings.DATABASE_URL
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


engine = create_engine(
    _database_url(),
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(160), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    latency: Mapped[float | None] = mapped_column(nullable=True)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32))
    topics: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    source_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="draft", index=True)
    created_by: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="item")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_item_id: Mapped[str] = mapped_column(ForeignKey("knowledge_items.id"), index=True)
    item: Mapped["KnowledgeItem"] = relationship(back_populates="jobs")
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    current_step: Mapped[str | None] = mapped_column(String(32), default="queued", nullable=True)
    chunks_total: Mapped[int] = mapped_column(default=0)
    chunks_indexed: Mapped[int] = mapped_column(default=0)
    logs: Mapped[list] = mapped_column(JSON, default=list, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_item_id: Mapped[str] = mapped_column(ForeignKey("knowledge_items.id"), index=True)
    qdrant_point_id: Mapped[str] = mapped_column(String(36), unique=True)
    position: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def init_chat_database() -> None:
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            if engine.dialect.name == "postgresql":
                try:
                    conn.execute(text("ALTER TABLE knowledge_items ALTER COLUMN source_message_id TYPE VARCHAR(64)"))
                    conn.commit()
                except Exception as exc:
                    print(f"Migration notice for source_message_id: {exc}")
                    conn.rollback()

                try:
                    conn.execute(text("ALTER TABLE knowledge_items ADD COLUMN IF NOT EXISTS topics JSONB"))
                    conn.commit()
                except Exception as exc:
                    print(f"Migration notice for knowledge_items.topics: {exc}")
                    conn.rollback()

                try:
                    conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS current_step VARCHAR(32) DEFAULT 'queued'"))
                    conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS logs JSONB DEFAULT '[]'::jsonb"))
                    conn.commit()
                except Exception as exc:
                    print(f"Migration notice for ingestion_jobs: {exc}")
                    conn.rollback()
            elif engine.dialect.name == "sqlite":
                res = conn.execute(text("PRAGMA table_info(ingestion_jobs)")).fetchall()
                cols = [r[1] for r in res]
                if "current_step" not in cols:
                    conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN current_step VARCHAR(32) DEFAULT 'queued'"))
                if "logs" not in cols:
                    conn.execute(text("ALTER TABLE ingestion_jobs ADD COLUMN logs JSON DEFAULT '[]'"))
                knowledge_cols = [
                    row[1] for row in conn.execute(text("PRAGMA table_info(knowledge_items)")).fetchall()
                ]
                if "topics" not in knowledge_cols:
                    conn.execute(text("ALTER TABLE knowledge_items ADD COLUMN topics JSON"))
                conn.commit()
        except Exception as exc:
            print(f"Migration check notice: {exc}")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
