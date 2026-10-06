# VERA Backend 2.0

Questo backend trasforma il prototipo VERA in una pipeline reale:

100 RSS -> acquisizione parallela -> deduplicazione -> clustering locale
-> verifica editoriale AI -> briefing strutturato.

## Provider AI

Primario: Groq
Fallback: Gemini

Variabili:
- GROQ_API_KEY
- GEMINI_API_KEY (oppure GOOGLE_API_KEY)

Modelli predefiniti:
- GROQ_MODEL_FAST=openai/gpt-oss-20b
- GROQ_MODEL_DEEP=openai/gpt-oss-120b
- GEMINI_MODEL=gemini-2.5-flash

## Render

Build:
pip install -r requirements.txt

Start:
uvicorn main:app --host 0.0.0.0 --port $PORT

## Endpoint

GET /
GET /health
GET /v1/sources
GET /v1/briefing
POST /v1/refresh
POST /v1/briefing
GET /v1/events

POST /v1/briefing body:
{
  "interests": ["Italia", "Esteri", "Economia"],
  "max_items": 8
}

## Importante

Il database persistente non è ancora obbligatorio per il primo test:
lo stato corrente è in memoria.

Per una versione pubblica bisogna aggiungere Render Postgres e spostare:
- articoli
- eventi
- briefing
- stato dei feed (ETag/Last-Modified)
- preferenze utente

in database.

Il filesystem locale dei Free Web Services Render non è persistente.
