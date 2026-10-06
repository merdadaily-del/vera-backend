import json
import os
import time
from typing import Any

from groq import Groq
from google import genai
from google.genai import types


GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
GROQ_MODEL_DEEP = os.getenv("GROQ_MODEL_DEEP", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class AIError(Exception):
    pass


def _json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception as exc:
        raise AIError(f"Risposta AI non valida come JSON: {exc}") from exc


def _groq(prompt: str, model: str, max_tokens: int = 1600) -> dict[str, Any]:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise AIError("GROQ_API_KEY non configurata")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei il motore editoriale di VERA. "
                    "Non inventare fatti. Usa esclusivamente le fonti fornite. "
                    "Rispondi SOLO con JSON valido."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return _json_from_text(response.choices[0].message.content)


def _gemini(prompt: str, max_tokens: int = 1600) -> dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise AIError("GEMINI_API_KEY/GOOGLE_API_KEY non configurata")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    return _json_from_text(response.text)


def generate_json(prompt: str, deep: bool = False) -> dict[str, Any]:
    """
    Provider principale: Groq.
    Fallback automatico: Gemini.
    Se entrambi falliscono, solleva AIError: il backend non inventa una rassegna.
    """
    groq_model = GROQ_MODEL_DEEP if deep else GROQ_MODEL_FAST

    try:
        return _groq(prompt, groq_model)
    except Exception as groq_error:
        try:
            return _gemini(prompt)
        except Exception as gemini_error:
            raise AIError(
                f"Groq fallito ({groq_error}); Gemini fallback fallito ({gemini_error})"
            ) from gemini_error


def verify_event(event: dict[str, Any]) -> dict[str, Any]:
    sources = event["articles"]
    source_text = "\n\n".join(
        f"SOURCE_ID={a['id']}\nTESTATA={a['outlet']}\nDATA={a['published']}\nTITOLO={a['title']}\nDESCRIZIONE={a['summary'][:1200]}\nURL={a['link']}"
        for a in sources
    )

    prompt = f"""
VERIFICA EDITORIALE VERA.

Devi verificare un singolo evento giornalistico usando SOLO le fonti sotto.
Non usare conoscenze esterne.

Regole:
- una notizia non è confermata solo perché compare molte volte;
- se più testate sembrano riprendere la stessa agenzia/origine, non contarle come conferme indipendenti;
- distingui FATTO, DICHIARAZIONE, ACCUSA e INTERPRETAZIONE;
- non colmare buchi con supposizioni;
- se le fonti sono in conflitto, dichiaralo;
- per guerre, morti, attentati, accuse, elezioni, emergenze e numeri sensibili usa una soglia più severa;
- CONFIRMED richiede almeno un'evidenza forte o più fonti realmente indipendenti;
- REPORTED se una fonte affidabile lo riferisce ma non c'è sufficiente conferma indipendente;
- UNVERIFIED se non è possibile stabilire cosa sia verificato.

EVENTO:
{json.dumps({
    "title": event["title"],
    "category": event["category"],
    "articles": sources
}, ensure_ascii=False)}

FONTI:
{source_text}

Restituisci ESATTAMENTE:
{{
  "status": "CONFIRMED|REPORTED|UNVERIFIED",
  "confidence": 0,
  "confirmed_facts": ["..."],
  "reported_claims": ["..."],
  "contradictions": ["..."],
  "important_uncertainties": ["..."],
  "independent_source_ids": ["..."],
  "reason": "..."
}}
"""
    return generate_json(prompt, deep=True)


def build_briefing(events: list[dict[str, Any]], interests: list[str]) -> dict[str, Any]:
    evidence = []
    for e in events:
        evidence.append({
            "event_id": e["id"],
            "title": e["title"],
            "category": e["category"],
            "verification": e.get("verification", {}),
            "sources": [
                {
                    "id": a["id"],
                    "outlet": a["outlet"],
                    "published": a["published"],
                    "title": a["title"],
                    "link": a["link"],
                    "summary": a["summary"][:1200],
                }
                for a in e["articles"]
            ],
        })

    prompt = f"""
SEI VERA, una rassegna stampa intelligente e personalizzata.

Interessi dell'utente: {", ".join(interests) if interests else "tutti"}

Crea una rassegna composta SOLO dagli eventi forniti.
Non inventare informazioni e non usare fatti esterni.
Non riprodurre frasi dei giornali.
Non scrivere come un aggregatore di titoli.

Per ogni evento selezionato:
1. spiega cosa è successo;
2. spiega perché conta;
3. indica cosa è cambiato o cosa succede ora;
4. conserva le incertezze;
5. attribuisci esplicitamente le affermazioni non confermate;
6. usa solo le fonti associate all'evento.

Se un evento è UNVERIFIED, non presentarlo come fatto.
Privilegia eventi importanti e materialmente nuovi.
Non scegliere una notizia solo perché ha molte fonti duplicate.

Restituisci JSON:
{{
  "headline": "titolo della rassegna",
  "intro": "breve apertura",
  "items": [
    {{
      "event_id": "...",
      "title": "...",
      "what_happened": "...",
      "why_it_matters": "...",
      "what_changed": "...",
      "uncertainty": "...",
      "status": "CONFIRMED|REPORTED|UNVERIFIED",
      "sources": ["source_id"]
    }}
  ]
}}

EVENTI:
{json.dumps(evidence, ensure_ascii=False)}
"""
    return generate_json(prompt, deep=True)
