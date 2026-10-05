import asyncio
import feedparser
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializza il client di Google GenAI leggendo la chiave dalle variabili d'ambiente
client = None
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)

RSS_URLS = {
    "ANSA": "https://www.ansa.it/sito/ansait_rss.xml",
    "RaiNews": "https://www.rainews.it/rss",
    "Il Sole 24 Ore": "https://www.ilsole24ore.com/rss/italia.xml",
    "Corriere": "https://www.corriere.it/rss/homepage.xml",
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml"
}

async def fetch_feed(client_http: httpx.AsyncClient, name: str, url: str):
    try:
        response = await client_http.get(url, timeout=5.0)
        if response.status_code == 200:
            feed = feedparser.parse(response.text)
            articles = []
            for entry in feed.entries[:5]:
                articles.append({
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "source": name
                })
            return articles
    except Exception:
        pass
    return []

@app.get("/api/briefing")
async def get_briefing():
    async with httpx.AsyncClient(follow_redirects=True) as client_http:
        tasks = [fetch_feed(client_http, name, url) for name, url in RSS_URLS.items()]
        results = await asyncio.gather(*tasks)
        all_articles = [art for sublist in results for art in sublist]

    evidence_text = ""
    for i, art in enumerate(all_articles[:15], 1):
        evidence_text += f"{i}. [{art['source']}] {art['title']} - Link: {art['url']}\n"

    prompt = f"""
Sei il curatore di una rassegna stampa professionale. Basandoti ESCLUSIVAMENTE sui seguenti titoli, genera un briefing sintetico e discorsivo in italiano. 
Per ogni notizia cita la testata e inserisci il link originale fornito. Non inventare nulla.

TITOLI:
{evidence_text}
"""

    briefing_content = ""
    try:
        if client:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt
            )
            briefing_content = response.text
        else:
            briefing_content = "Errore: GOOGLE_API_KEY non configurata."
    except Exception as e:
        briefing_content = f"Errore nella generazione con Google Gemini: {e}"

    return {
        "status": "success",
        "briefing": briefing_content,
        "total_articles": len(all_articles)
    }
