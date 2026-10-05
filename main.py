import asyncio
import feedparser
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura Google Gemini con la chiave d'ambiente
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

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

    try:
       model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        briefing_content = response.text
    except Exception as e:
        briefing_content = f"Errore nella generazione con Google Gemini: {e}"

    return {
        "status": "success",
        "briefing": briefing_content,
        "total_articles": len(all_articles)
    }
