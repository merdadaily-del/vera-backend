import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feeds import FEEDS
from rss import fetch_all
from clustering import cluster_articles
from ai import verify_event, build_briefing, AIError


app = FastAPI(title="VERA Editorial Backend", version="2.0")

frontend_url = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if frontend_url == "*" else [frontend_url],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {
    "articles": [],
    "events": [],
    "briefing": None,
    "last_refresh": None,
    "refreshing": False,
    "last_error": None,
}


class BriefingRequest(BaseModel):
    interests: list[str] = Field(default_factory=list)
    max_items: int = Field(default=8, ge=3, le=15)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


async def refresh_pipeline(interests: Optional[list[str]] = None, max_items: int = 8):
    if STATE["refreshing"]:
        return STATE["briefing"]

    STATE["refreshing"] = True
    STATE["last_error"] = None

    try:
        # 1. RSS paralleli
        articles = await fetch_all(FEEDS)

        # 2. Clustering locale: nessuna richiesta AI
        events = cluster_articles(articles)

        # 3. Seleziona candidati: fonti multiple + freschezza.
        candidates = []
        for event in events:
            outlets = {a["outlet"] for a in event["articles"]}
            if len(event["articles"]) >= 2 or len(outlets) >= 2:
                candidates.append(event)

        # Evita di mandare centinaia di eventi all'AI.
        candidates = candidates[:30]

        # 4. Verifica solo i candidati più importanti.
        verified = []
        for event in candidates:
            try:
                event["verification"] = verify_event(event)
                status = event["verification"].get("status")
                if status in {"CONFIRMED", "REPORTED"}:
                    verified.append(event)
            except AIError:
                # Se l'AI non è disponibile, non inventiamo una verifica.
                continue

        # 5. Rassegna finale
        if verified:
            briefing = build_briefing(
                verified[:20],
                interests or [],
            )
        else:
            briefing = {
                "headline": "VERA non ha ancora una rassegna verificata",
                "intro": "Le fonti sono state aggiornate, ma non ci sono abbastanza elementi verificati per costruire una rassegna affidabile.",
                "items": [],
            }

        STATE["articles"] = articles
        STATE["events"] = verified
        STATE["briefing"] = {
            **briefing,
            "created_at": now_iso(),
            "source_count": len(articles),
            "event_count": len(verified),
            "provider": "groq-primary/gemini-fallback",
        }
        STATE["last_refresh"] = now_iso()
        return STATE["briefing"]

    except Exception as exc:
        STATE["last_error"] = str(exc)
        raise
    finally:
        STATE["refreshing"] = False


@app.get("/")
def root():
    return {
        "service": "VERA Editorial Backend",
        "status": "ok",
        "feeds": len(FEEDS),
        "last_refresh": STATE["last_refresh"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "feeds": len(FEEDS),
        "last_refresh": STATE["last_refresh"],
        "refreshing": STATE["refreshing"],
        "last_error": STATE["last_error"],
    }


@app.get("/v1/briefing")
def get_briefing():
    return {
        "briefing": STATE["briefing"],
        "last_refresh": STATE["last_refresh"],
        "refreshing": STATE["refreshing"],
    }


@app.post("/v1/briefing")
async def create_briefing(request: BriefingRequest):
    try:
        briefing = await refresh_pipeline(request.interests, request.max_items)
        return {"briefing": briefing}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/v1/refresh")
async def refresh(request: BriefingRequest = BriefingRequest()):
    # Primo avvio: aspetta il risultato.
    # Chiamate successive: il frontend può mostrare il briefing precedente
    # mentre questa pipeline aggiorna i dati.
    if STATE["briefing"] is None:
        return {"briefing": await refresh_pipeline(request.interests, request.max_items)}

    if not STATE["refreshing"]:
        asyncio.create_task(refresh_pipeline(request.interests, request.max_items))

    return {
        "accepted": True,
        "message": "Aggiornamento avviato",
        "last_briefing": STATE["briefing"],
        "last_refresh": STATE["last_refresh"],
    }


@app.get("/v1/events")
def get_events(limit: int = Query(default=20, ge=1, le=100)):
    return {
        "events": STATE["events"][:limit],
        "last_refresh": STATE["last_refresh"],
    }


@app.get("/v1/sources")
def get_sources():
    return {"sources": FEEDS, "count": len(FEEDS)}
