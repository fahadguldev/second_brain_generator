from uuid import UUID, uuid4

from fastapi import Cookie, Depends, Header, HTTPException, Response
from itsdangerous import BadSignature, URLSafeSerializer
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.chat_database import Conversation, Message, User, get_db, utcnow
from src.config import settings

COOKIE_NAME = "second_brain_visitor"
serializer = URLSafeSerializer(settings.SESSION_SECRET, salt="anonymous-visitor")


def _read_visitor_id(cookie: str | None, header: str | None = None) -> str | None:
    if header:
        try:
            return str(UUID(header.strip()))
        except (ValueError, TypeError):
            pass
    if not cookie:
        return None
    try:
        return str(UUID(serializer.loads(cookie)))
    except (BadSignature, ValueError, TypeError):
        return None


def get_current_user(
    response: Response,
    visitor_cookie: str | None = Cookie(default=None, alias=COOKIE_NAME),
    visitor_header: str | None = Header(default=None, alias="X-Visitor-Id"),
    db: Session = Depends(get_db),
) -> User:
    existing_id = _read_visitor_id(visitor_cookie, visitor_header)
    user_id = existing_id or str(uuid4())
    user = db.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        db.add(user)
    user.last_seen_at = utcnow()
    db.commit()
    response.headers["X-Visitor-Id"] = user_id
    if existing_id != user_id:
        response.set_cookie(
            COOKIE_NAME, serializer.dumps(user_id), max_age=60 * 60 * 24 * 365,
            httponly=True, secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE, domain=settings.COOKIE_DOMAIN, path="/",
        )
    return user


def owned_conversation(db: Session, conversation_id: str, user_id: str) -> Conversation:
    conversation = db.scalar(select(Conversation).where(
        Conversation.id == conversation_id, Conversation.user_id == user_id
    ))
    if conversation is None:
        # Check if conversation exists (e.g. created before visitor cookie/header was synchronized)
        conv_by_id = db.get(Conversation, conversation_id)
        if conv_by_id is not None:
            conv_by_id.user_id = user_id
            db.commit()
            return conv_by_id
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def serialize_message(message: Message) -> dict:
    return {
        "id": message.id, "role": message.role, "text": message.content,
        "created_at": message.created_at,
        "info": ({"latency": message.latency, "model": message.model,
                  "embedding_model": message.embedding_model, "sources": message.sources or []}
                 if message.role == "assistant" else None),
    }
