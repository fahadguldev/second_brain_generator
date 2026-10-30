import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.chat_database import Conversation, Message, SessionLocal, User, get_db, utcnow
from src.chat_service import get_current_user, owned_conversation, serialize_message
from src.config import settings
from src.database import ensure_collection
from src.llm import GenerationError
from src.rag import generate_answer, generate_answer_stream
from fastapi import File, Form, UploadFile
from src.voice_service import (
    get_voice_status,
    get_active_provider,
    synthesize_speech,
    synthesize_speech_stream,
    clone_voice,
    VoiceConfigurationError,
    VoiceServiceError,
)
from src.elevenlabs_service import (
    get_convai_signed_url,
    ElevenLabsConfigurationError,
    ElevenLabsServiceError,
)

router = APIRouter(prefix="/api", tags=["second-brain"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_dict: Optional[dict] = None
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    latency: float
    top_k: int
    model: str
    embedding_model: str
    conversation_id: str
    user_message_id: str
    assistant_message_id: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    id: str
    role: str
    text: str
    created_at: datetime
    info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embedding_model: str
    generation_model: str


@router.on_event("startup")
def startup_event():
    ensure_collection(vector_size=settings.EMBEDDING_DIMENSION)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    from src.database import _get_client
    qdrant_status = "connected"
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Qdrant unavailable")
        client.get_collection(collection_name=settings.QDRANT_COLLECTION)
    except Exception:
        qdrant_status = "error"
    return HealthResponse(
        status="ok" if qdrant_status == "connected" else "degraded",
        qdrant=qdrant_status,
        embedding_model=settings.EMBEDDING_MODEL,
        generation_model=settings.GENERATION_MODEL,
    )


@router.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.scalars(
        select(Conversation).where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).all()


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
def create_conversation(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = Conversation(id=str(uuid4()), user_id=user.id)
    db.add(conversation)
    db.commit()
    return conversation


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
def conversation_messages(
    conversation_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    conversation = owned_conversation(db, conversation_id, user.id)
    return [serialize_message(message) for message in conversation.messages]


@router.post("/ask", response_model=AskResponse)
def ask_question(
    req: AskRequest,
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
) -> AskResponse:
    if req.conversation_id:
        conversation = owned_conversation(db, req.conversation_id, user.id)
    else:
        conversation = Conversation(id=str(uuid4()), user_id=user.id)
        db.add(conversation)

    previous_messages = [
        {"role": message.role, "content": message.content}
        for message in conversation.messages[-10:]
    ]
    user_message = Message(id=str(uuid4()), conversation=conversation, role="user", content=req.question)
    db.add(user_message)
    if not previous_messages:
        conversation.title = req.question.strip()[:80]
    conversation.updated_at = utcnow()
    db.commit()

    try:
        result = generate_answer(
            question=req.question, top_k=req.top_k, filter_dict=req.filter_dict,
            conversation_history=previous_messages,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    assistant_message = Message(
        id=str(uuid4()), conversation_id=conversation.id, role="assistant",
        content=result["answer"], model=result["model"], embedding_model=result["embedding_model"],
        latency=result["latency"], sources=result["sources"],
    )
    db.add(assistant_message)
    conversation.updated_at = utcnow()
    db.commit()

    return AskResponse(
        answer=result["answer"], sources=result["sources"], latency=result["latency"],
        top_k=req.top_k, model=result["model"], embedding_model=result["embedding_model"],
        conversation_id=conversation.id,
        user_message_id=user_message.id, assistant_message_id=assistant_message.id,
    )


@router.post("/ask/stream")
def ask_question_stream(
    req: AskRequest,
    response: Response,
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
) -> StreamingResponse:
    if req.conversation_id:
        conversation = owned_conversation(db, req.conversation_id, user.id)
    else:
        conversation = Conversation(id=str(uuid4()), user_id=user.id)
        db.add(conversation)

    previous_messages = [
        {"role": message.role, "content": message.content}
        for message in conversation.messages[-10:]
    ]
    user_message = Message(
        id=str(uuid4()), conversation=conversation, role="user", content=req.question,
    )
    db.add(user_message)
    if not previous_messages:
        conversation.title = req.question.strip()[:80]
    conversation.updated_at = utcnow()
    db.commit()

    try:
        result = generate_answer_stream(
            question=req.question, top_k=req.top_k, filter_dict=req.filter_dict,
            conversation_history=previous_messages,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not prepare response") from exc

    conversation_id = conversation.id
    user_message_id = user_message.id

    def events():
        yield json.dumps({
            "type": "metadata", "sources": result["sources"],
            "latency": result["latency"], "model": result["model"],
            "embedding_model": result["embedding_model"],
        }) + "\n"
        answer_parts: list[str] = []
        stream_db = SessionLocal()
        try:
            for text in result["stream"]:
                answer_parts.append(text)
                yield json.dumps({"type": "delta", "text": text}) + "\n"

            assistant_message = Message(
                id=str(uuid4()), conversation_id=conversation_id, role="assistant",
                content="".join(answer_parts), model=result["model"],
                embedding_model=result["embedding_model"], latency=result["latency"],
                sources=result["sources"],
            )
            stream_conversation = stream_db.get(Conversation, conversation_id)
            stream_db.add(assistant_message)
            if stream_conversation:
                stream_conversation.updated_at = utcnow()
            stream_db.commit()
            yield json.dumps({
                "type": "done", "conversation_id": conversation_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message.id,
            }) + "\n"
        except Exception:
            stream_db.rollback()
            yield json.dumps({
                "type": "error", "message": "Response generation was interrupted.",
            }) + "\n"
        finally:
            stream_db.close()

    stream_response = StreamingResponse(
        events(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    for k, v in response.headers.items():
        if k.lower() not in ("content-type", "content-length"):
            stream_response.headers[k] = v
    for header_name, header_value in response.raw_headers:
        if header_name.lower() == b"set-cookie":
            stream_response.raw_headers.append((header_name, header_value))
    return stream_response


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    voice_id: Optional[str] = Field(default=None, max_length=128)
    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    model_id: Optional[str] = Field(default=None, max_length=128)
    provider: Optional[str] = Field(default=None, max_length=32)


class VoiceStatusResponse(BaseModel):
    provider: str = "elevenlabs"
    configured: bool
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    speed: float = 1.0
    has_convai_agent: bool = False
    convai_agent_id: Optional[str] = None
    has_user_id: Optional[bool] = None
    key_pool: Dict[str, Any] = Field(default_factory=dict)


class VoiceCloneResponse(BaseModel):
    success: bool
    voice_id: Optional[str]
    name: str
    gender: str
    language: str


class ConvAISignedUrlResponse(BaseModel):
    signed_url: str
    agent_id: str


@router.get("/voice/status", response_model=VoiceStatusResponse)
def voice_status() -> VoiceStatusResponse:
    return VoiceStatusResponse(**get_voice_status())


@router.get("/tts/status", response_model=VoiceStatusResponse)
def tts_status() -> VoiceStatusResponse:
    return VoiceStatusResponse(**get_voice_status())


@router.get("/voice/convai/signed-url", response_model=ConvAISignedUrlResponse)
async def convai_signed_url_endpoint(agent_id: Optional[str] = None):
    """
    Generate a secure signed WebSocket URL for ElevenLabs Conversational AI Agent.
    """
    try:
        result = await get_convai_signed_url(agent_id=agent_id)
        return ConvAISignedUrlResponse(**result)
    except (VoiceConfigurationError, ElevenLabsConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (VoiceServiceError, ElevenLabsServiceError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(exc)}") from exc


@router.post("/tts")
async def generate_speech(req: TTSRequest):
    try:
        audio_bytes = await synthesize_speech(
            req.text,
            voice_id=req.voice_id,
            speed=req.speed,
            model_id=req.model_id,
            provider=req.provider,
        )
        media_type = "audio/wav" if audio_bytes.startswith(b"RIFF") else "audio/mpeg"
        filename = "speech.wav" if media_type == "audio/wav" else "speech.mp3"
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "public, max-age=86400",
            },
        )
    except (VoiceConfigurationError, ElevenLabsConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (VoiceServiceError, ElevenLabsServiceError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(exc)}") from exc


@router.post("/tts/stream")
async def stream_speech(req: TTSRequest):
    try:
        active_p = (req.provider or get_active_provider()).lower()
        media_type = "audio/mpeg" if active_p == "elevenlabs" else "audio/wav"
        return StreamingResponse(
            synthesize_speech_stream(
                req.text,
                voice_id=req.voice_id,
                speed=req.speed,
                model_id=req.model_id,
                provider=req.provider,
            ),
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except (VoiceConfigurationError, ElevenLabsConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS stream failed: {str(exc)}") from exc


@router.post("/voice/clone", response_model=VoiceCloneResponse)
async def clone_voice_endpoint(
    audio: UploadFile = File(...),
    name: str = Form(...),
    gender: str = Form(default="Male"),
    language: str = Form(default="en"),
):
    """
    Clone a voice from an audio sample via JSF Labs API:
    POST https://www.jsflabs.io/api/v1/voice-clone
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")
    try:
        result = await clone_voice(
            audio_bytes=audio_bytes,
            name=name,
            gender=gender,
            language=language,
            filename=audio.filename or "sample.wav",
        )
        return VoiceCloneResponse(**result)
    except VoiceConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(exc)}") from exc
