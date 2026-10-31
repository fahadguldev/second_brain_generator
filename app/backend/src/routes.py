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
from src.voice_service import (
    get_voice_status,
    synthesize_speech,
    synthesize_speech_stream,
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
def list_conversations():
    # DB chat history commented out for stateless deployment:
    # return db.scalars(select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc())).all()
    return []


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
def create_conversation():
    # conversation = Conversation(id=str(uuid4()), user_id=user.id)
    # db.add(conversation)
    # db.commit()
    # return conversation
    now = utcnow()
    return ConversationResponse(id=str(uuid4()), title="New Chat", created_at=now, updated_at=now)


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
def conversation_messages(conversation_id: str):
    # conversation = owned_conversation(db, conversation_id, user.id)
    # return [serialize_message(message) for message in conversation.messages]
    return []


@router.post("/ask", response_model=AskResponse)
def ask_question(
    req: AskRequest,
) -> AskResponse:
    # DB persistence commented out for stateless deployment:
    # if req.conversation_id:
    #     conversation = owned_conversation(db, req.conversation_id, user.id)
    # else:
    #     conversation = Conversation(id=str(uuid4()), user_id=user.id)
    #     db.add(conversation)
    conv_id = req.conversation_id or str(uuid4())

    try:
        result = generate_answer(
            question=req.question, top_k=req.top_k, filter_dict=req.filter_dict,
            conversation_history=[],
        )
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # assistant_message = Message(id=str(uuid4()), conversation_id=conversation.id, role="assistant", ...)
    # db.add(assistant_message)
    # db.commit()

    return AskResponse(
        answer=result["answer"], sources=result["sources"], latency=result["latency"],
        top_k=req.top_k, model=result["model"], embedding_model=result["embedding_model"],
        conversation_id=conv_id,
        user_message_id=str(uuid4()), assistant_message_id=str(uuid4()),
    )


@router.post("/ask/stream")
def ask_question_stream(
    req: AskRequest,
    response: Response,
) -> StreamingResponse:
    # DB persistence commented out for stateless deployment:
    # if req.conversation_id:
    #     conversation = owned_conversation(db, req.conversation_id, user.id)
    # else:
    #     conversation = Conversation(id=str(uuid4()), user_id=user.id)
    #     db.add(conversation)
    conv_id = req.conversation_id or str(uuid4())
    user_msg_id = str(uuid4())
    asst_msg_id = str(uuid4())

    try:
        result = generate_answer_stream(
            question=req.question, top_k=req.top_k, filter_dict=req.filter_dict,
            conversation_history=[],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not prepare response") from exc

    def events():
        yield json.dumps({
            "type": "metadata", "sources": result["sources"],
            "latency": result["latency"], "model": result["model"],
            "embedding_model": result["embedding_model"],
        }) + "\n"

        # stream_db = SessionLocal()
        try:
            for text in result["stream"]:
                yield json.dumps({"type": "delta", "text": text}) + "\n"

            # DB persistence commented out:
            # assistant_message = Message(...)
            # stream_db.add(assistant_message)
            # stream_db.commit()

            yield json.dumps({
                "type": "done", "conversation_id": conv_id,
                "user_message_id": user_msg_id,
                "assistant_message_id": asst_msg_id,
            }) + "\n"
        except Exception:
            yield json.dumps({
                "type": "error", "message": "Response generation was interrupted.",
            }) + "\n"
        finally:
            pass

    stream_response = StreamingResponse(
        events(), media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
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


class VoiceStatusResponse(BaseModel):
    provider: str = "elevenlabs"
    configured: bool
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    speed: float = 1.0
    has_convai_agent: bool = False
    convai_agent_id: Optional[str] = None
    key_pool: Dict[str, Any] = Field(default_factory=dict)


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
        )
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'inline; filename="speech.mp3"',
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
        return StreamingResponse(
            synthesize_speech_stream(
                req.text,
                voice_id=req.voice_id,
                speed=req.speed,
                model_id=req.model_id,
            ),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except (VoiceConfigurationError, ElevenLabsConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS stream failed: {str(exc)}") from exc
