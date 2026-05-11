# ============================================================
# main.py - Updated for small_talk + general_advisory support
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import text
import uuid
import os
from dotenv import load_dotenv
from database import SessionLocal
from retrieval import retrieve
from llm import generate_response
from query_processor import classify_intent

load_dotenv()

app = FastAPI(
    title="Crop Disease Advisory Chatbot",
    description="RAG-based crop disease advisor for Pakistani farmers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/images",
    StaticFiles(directory="images"),
    name="images"
)

class ChatRequest(BaseModel):
    query: str
    crop: Optional[str] = None
    province: Optional[str] = None
    language: str = "english"
    user_id: Optional[int] = None
    session_id: Optional[str] = None

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    data_consent: bool
    preferred_language: str = "english"

class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
def health_check():
    return {
        "status"  : "running",
        "message" : "Crop Disease Advisory API is online"
    }


@app.post("/signup")
def signup(request: SignupRequest):
    if not request.data_consent:
        raise HTTPException(status_code=400, detail="Data consent is required to register")

    db = SessionLocal()
    try:
        existing = db.execute(
            text("SELECT id FROM users WHERE username = :username"),
            {"username": request.username}
        ).fetchone()

        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")

        import bcrypt
        hashed = bcrypt.hashpw(
            request.password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        result = db.execute(text("""
            INSERT INTO users
                (username, email, hashed_password,
                 data_consent, consent_timestamp, preferred_language)
            VALUES
                (:username, :email, :hashed_password,
                 :data_consent, NOW(), :preferred_language)
            RETURNING id
        """), {
            "username"          : request.username,
            "email"             : request.email,
            "hashed_password"   : hashed,
            "data_consent"      : request.data_consent,
            "preferred_language": request.preferred_language
        })

        new_user_id = result.fetchone()[0]
        db.commit()

        return {
            "message" : "Account created successfully",
            "user_id" : new_user_id,
            "username": request.username
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/login")
def login(request: LoginRequest):
    db = SessionLocal()
    try:
        user = db.execute(
            text("""
                SELECT id, username, hashed_password, preferred_language
                FROM users WHERE username = :username
            """),
            {"username": request.username}
        ).fetchone()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        import bcrypt
        password_match = bcrypt.checkpw(
            request.password.encode("utf-8"),
            user.hashed_password.encode("utf-8")
        )

        if not password_match:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        return {
            "message"           : "Login successful",
            "user_id"           : user.id,
            "username"          : user.username,
            "preferred_language": user.preferred_language
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/chat")
def chat(request: ChatRequest):
    db = SessionLocal()
    try:
        if not request.query or request.query.strip() == "":
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        print(f"\n{'='*50}")
        print(f"📨 New chat request")
        print(f"   Query   : {request.query}")
        print(f"   Crop    : {request.crop}")
        print(f"   Province: {request.province}")
        print(f"   Language: {request.language}")
        print(f"{'='*50}")

        session_id = request.session_id or str(uuid.uuid4())

        # ── Fetch prior conversation history for this session ─
        conversation_history = []
        history_crop     = None
        history_province = None

        if session_id and request.session_id:
            prior_rows = db.execute(text("""
                SELECT query_text, response_text, crop_selected, province_selected
                FROM chat_logs
                WHERE session_id = :session_id
                ORDER BY timestamp ASC
                LIMIT 10
            """), {"session_id": session_id}).fetchall()

            for row in prior_rows:
                conversation_history.append({"role": "user",      "content": row.query_text})
                conversation_history.append({"role": "assistant",  "content": row.response_text})
                # Carry forward crop/province from earlier turns if not set now
                if row.crop_selected and not history_crop:
                    history_crop = row.crop_selected
                if row.province_selected and not history_province:
                    history_province = row.province_selected

        # ── Build a context-enriched query using recent history ──────────
        # This ensures follow-up messages like "is it a disease?" or
        # "what about the tubers?" are retrieved correctly
        effective_crop     = request.crop     or history_crop
        effective_province = request.province or history_province

        # Prepend the last 2 user turns to the current query for embedding
        recent_user_turns = [m["content"] for m in conversation_history if m["role"] == "user"][-2:]
        if recent_user_turns:
            enriched_query = " ".join(recent_user_turns) + " " + request.query
            print(f"   🔗 Context-enriched query: '{enriched_query[:120]}...'")
        else:
            enriched_query = request.query

        # ── Classify intent FIRST using query_processor ──────
        intent = classify_intent(enriched_query)
        print(f"   🎯 Classified intent: {intent}")

        # ── Handle academic_general intent: LLM answers directly
        # No RAG retrieval needed — retrieve broad context for the crop
        # mentioned so the LLM can give a proper overview answer
        if intent == "academic_general":
            print(f"   🎓 Academic/general query — retrieving broad context for overview")
            chunks = retrieve(
                query=enriched_query,
                crop=effective_crop,   # filter by crop if selected, else None
                province=None,
                top_k=5               # more chunks = better overview
            )
            top_chunk_type = "academic_general"

        # ── Handle price/market intent ────────────────────────
        elif intent == "price_market":
            print(f"   💰 Price/market query — no retrieval needed")
            chunks = []
            top_chunk_type = "price_market"

        else:
            # ── Detect chunk type from DB for other intents ───
            top_chunk_type = None

            broad_chunks = retrieve(
                query=enriched_query,
                crop=None,
                province=None,
                top_k=1
            )

            if broad_chunks:
                top_chunk_type = broad_chunks[0].get("chunk_type", "identification")

            # ── If small_talk or general_advisory — use broad results directly
            if top_chunk_type in ("small_talk", "general_advisory"):
                print(f"   🔀 Detected chunk_type: {top_chunk_type} — skipping crop/province filters")
                chunks = retrieve(
                    query=enriched_query,
                    crop=None,
                    province=None,
                    top_k=3
                )

            # ── Otherwise apply crop/province filters ─────────
            else:
                print(f"   🔀 Detected chunk_type: {top_chunk_type} — applying crop/province filters")
                print(f"   🌾 Effective crop: {effective_crop} | Province: {effective_province}")
                chunks = retrieve(
                    query=enriched_query,
                    crop=effective_crop,
                    province=effective_province,
                    top_k=3
                )

        # ── Determine confidence ──────────────────────────────
        if chunks:
            overall_confidence = chunks[0]["confidence"]
        else:
            overall_confidence = "Insufficient information in agricultural records"

        # ── Generate response via LLM ─────────────────────────
        response_text = generate_response(
            query=request.query,
            chunks=chunks,
            language=request.language,
            confidence=overall_confidence,
            conversation_history=conversation_history,
            intent=intent
        )

        # ── Image: only show for disease chunks ──────────────
        # small_talk, general_advisory, and academic_general have no single image
        image_file = None
        if chunks and top_chunk_type not in ("small_talk", "general_advisory", "academic_general", "price_market"):
            image_file = chunks[0].get("image_file") or None

        # ── Log to database ───────────────────────────────────
        chat_log_result = db.execute(text("""
            INSERT INTO chat_logs
                (user_id, session_id, query_text, response_text,
                 crop_selected, province_selected, language_used, timestamp)
            VALUES
                (:user_id, :session_id, :query_text, :response_text,
                 :crop_selected, :province_selected, :language_used, NOW())
            RETURNING id
        """), {
            "user_id"          : request.user_id,
            "session_id"       : session_id,
            "query_text"       : request.query,
            "response_text"    : response_text,
            "crop_selected"    : request.crop,
            "province_selected": request.province,
            "language_used"    : request.language
        })

        chat_log_id = chat_log_result.fetchone()[0]

        for chunk in chunks:
            db.execute(text("""
                INSERT INTO retrieved_chunks
                    (chat_log_id, chunk_text, disease_name,
                     similarity_score, source, image_file)
                VALUES
                    (:chat_log_id, :chunk_text, :disease_name,
                     :similarity_score, :source, :image_file)
            """), {
                "chat_log_id"     : chat_log_id,
                "chunk_text"      : chunk["chunk_text"],
                "disease_name"    : chunk["disease_name"],
                "similarity_score": chunk["similarity_score"],
                "source"          : str(chunk["source"]),
                "image_file"      : chunk.get("image_file") or None
            })

        db.commit()

        # ── Build evidence list ───────────────────────────────
        # Only include evidence panel for disease chunks
        evidence = []
        if top_chunk_type not in ("small_talk", "general_advisory"):
            for chunk in chunks:
                evidence.append({
                    "disease_name"    : chunk["disease_name"],
                    "chunk_type"      : chunk["chunk_type"],
                    "chunk_text"      : chunk["chunk_text"],
                    "similarity_score": chunk["similarity_score"],
                    "confidence"      : chunk["confidence"],
                    "source"          : chunk["source"],
                    "image_file"      : chunk.get("image_file") or None
                })

        return {
            "session_id"  : session_id,
            "response"    : response_text,
            "image_file"  : image_file,
            "image_url"   : f"http://localhost:8000/images/{image_file}" if image_file else None,
            "confidence"  : overall_confidence,
            "evidence"    : evidence,
            "chat_log_id" : chat_log_id,
            "chunk_type"  : top_chunk_type   # NEW: frontend can use this to style response
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/history/{user_id}")
def get_history(user_id: int):
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT session_id, query_text, response_text,
                   crop_selected, province_selected, language_used, timestamp
            FROM chat_logs
            WHERE user_id = :user_id
            ORDER BY timestamp DESC
            LIMIT 50
        """), {"user_id": user_id}).fetchall()

        history = []
        for row in rows:
            history.append({
                "session_id"       : str(row.session_id),
                "query_text"       : row.query_text,
                "response_text"    : row.response_text,
                "crop_selected"    : row.crop_selected,
                "province_selected": row.province_selected,
                "language_used"    : row.language_used,
                "timestamp"        : str(row.timestamp)
            })

        return {"history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/session/{session_id}")
def get_session(session_id: str):
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT id, query_text, response_text, language_used, timestamp
            FROM chat_logs
            WHERE session_id = :session_id
            ORDER BY timestamp ASC
        """), {"session_id": session_id}).fetchall()

        messages = []
        for row in rows:
            chunk = db.execute(text("""
                SELECT image_file, disease_name, similarity_score
                FROM retrieved_chunks
                WHERE chat_log_id = :id
                ORDER BY similarity_score DESC LIMIT 1
            """), {"id": row.id}).fetchone()

            messages.append({
                "chat_log_id"   : row.id,
                "query_text"    : row.query_text,
                "response_text" : row.response_text,
                "language_used" : row.language_used,
                "timestamp"     : str(row.timestamp),
                "image_file"    : chunk.image_file if chunk else None,
                "image_url"     : f"http://localhost:8000/images/{chunk.image_file}" if chunk and chunk.image_file else None,
                "confidence"    : chunk.similarity_score if chunk else None,
            })

        return {"messages": messages}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.delete("/chat/{chat_log_id}")
def delete_chat(chat_log_id: int):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM retrieved_chunks WHERE chat_log_id = :id"), {"id": chat_log_id})
        db.execute(text("DELETE FROM chat_logs WHERE id = :id"), {"id": chat_log_id})
        db.commit()
        return {"message": "Chat deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    db = SessionLocal()
    try:
        rows = db.execute(text("SELECT id FROM chat_logs WHERE session_id = :sid"), {"sid": session_id}).fetchall()
        for row in rows:
            db.execute(text("DELETE FROM retrieved_chunks WHERE chat_log_id = :id"), {"id": row.id})
        db.execute(text("DELETE FROM chat_logs WHERE session_id = :sid"), {"sid": session_id})
        db.commit()
        return {"message": "Session deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/evidence/{chat_log_id}")
def get_evidence(chat_log_id: int):
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT chunk_text, disease_name, similarity_score, source, image_file
            FROM retrieved_chunks
            WHERE chat_log_id = :chat_log_id
            ORDER BY similarity_score DESC
        """), {"chat_log_id": chat_log_id}).fetchall()

        evidence = []
        for row in rows:
            evidence.append({
                "disease_name"    : row.disease_name,
                "chunk_text"      : row.chunk_text,
                "similarity_score": row.similarity_score,
                "source"          : row.source,
                "image_file"      : row.image_file
            })

        return {"evidence": evidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()