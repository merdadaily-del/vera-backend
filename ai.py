import json
import os
from typing import Any

from openai import OpenAI


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


class AIError(Exception):
    pass


def get_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise AIError("DEEPSEEK_API_KEY non configurata su Render")

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


def parse_json(text: str) -> dict[str, Any]:
    if not text:
        raise AIError("DeepSeek ha restituito una risposta vuota")

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except Exception as exc:
        raise AIError(
            f"Risposta DeepSeek non valida come JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise AIError("DeepSeek ha restituito un JSON non valido")

    return data


def generate_json(
    prompt: str,
    max_tokens: int = 2000,
) -> dict[str, Any]:

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei VERA, un motore editoriale giornalistico. "
                        "Usa esclusivamente le informazioni fornite. "
                        "Non inventare fatti, nomi, date, numeri o dichiarazioni. "
                        "Distingui fatti, dichiarazioni, accuse e informazioni "
                        "riportate. Se una notizia non è sufficientemente "
                        "confermata, dichiaralo. "
                        "Rispondi esclusivamente con JSON valido."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.1,
            max_tokens=max_tokens,

            response_format={
                "type": "json_object"
            },
        )

    except Exception as exc:
        raise AIError(
            f"Errore API DeepSeek: {exc}"
        ) from exc

    if not response.choices:
        raise AIError(
            "DeepSeek non ha restituito alcuna risposta"
        )

    content = response.choices[0].message.content

    return parse_json(content)


def compact_sources(
    articles: list[dict[str, Any]],
    max_sources: int = 3,
) -> list[dict[str, Any]]:

    result = []

    for article in articles[:max_sources]:

        result.append({
            "id": article.get("id"),
            "outlet": article.get("outlet"),
            "published": article.get("published"),
            "title": str(article.get("title", ""))[:300],
            "summary": str(article.get("summary", ""))[:600],
        })

    return result


def verify_event(
    event: dict[str, Any],
) -> dict[str, Any]:

    sources = compact_sources(
        event.get("articles", []),
        max_sources=3,
    )

    event_info = {
        "id": event.get("id"),
        "title": str(event.get("title", ""))[:400],
        "category": event.get("category", ""),
    }

    prompt = f"""
VERIFICA EDITORIALE VERA

Verifica l'evento utilizzando ESCLUSIVAMENTE
le fonti fornite.

NON usare conoscenze esterne.

REGOLE:

- CONFIRMED: le fonti forniscono elementi
  sufficienti per confermare il fatto.

- REPORTED: una fonte riporta il fatto,
  ma non c'è sufficiente conferma indipendente.

- UNVERIFIED: le fonti non permettono
  di stabilire adeguatamente cosa sia successo.

Distingui sempre tra fatti, dichiarazioni,
accuse e interpretazioni.

Non inventare informazioni.

EVENTO:

{json.dumps(event_info, ensure_ascii=False)}

FONTI:

{json.dumps(sources, ensure_ascii=False)}

Restituisci esclusivamente:

{{
  "status": "CONFIRMED|REPORTED|UNVERIFIED",
  "confidence": 0,
  "confirmed_facts": [],
  "reported_claims": [],
  "contradictions": [],
  "important_uncertainties": [],
  "independent_source_ids": [],
  "reason": ""
}}
"""

    return generate_json(
        prompt,
        max_tokens=2000,
    )


def build_briefing(
    events: list[dict[str, Any]],
    interests: list[str],
) -> dict[str, Any]:

    evidence = []

    for event in events[:6]:

        verification = event.get(
            "verification",
            {},
        )

        evidence.append({
            "event_id": event.get("id"),

            "title": str(
                event.get("title", "")
            )[:300],

            "category": event.get(
                "category",
                "",
            ),

            "verification": {
                "status": verification.get("status"),
                "confidence": verification.get("confidence"),
                "reason": str(
                    verification.get("reason", "")
                )[:400],
            },

            "sources": compact_sources(
                event.get("articles", []),
                max_sources=2,
            ),
        })

    prompt = f"""
SEI VERA, una rassegna stampa giornalistica.

Crea una rassegna in italiano usando
ESCLUSIVAMENTE gli eventi e le fonti fornite.

INTERESSI:

{json.dumps(
    interests or ["tutti"],
    ensure_ascii=False,
)}

REGOLE:

- Non inventare informazioni.
- Non usare conoscenze esterne.
- Non trasformare dichiarazioni in fatti.
- Mantieni le attribuzioni.
- REPORTED deve rimanere chiaramente attribuito.
- UNVERIFIED non deve essere presentato come fatto.
- Scrivi in italiano giornalistico naturale.
- Sii chiaro e sintetico.
- Dai priorità agli eventi più importanti.

Per ogni evento indica:
- cosa è successo;
- perché è importante;
- cosa cambia;
- eventuali incertezze.

EVENTI:

{json.dumps(
    evidence,
    ensure_ascii=False,
)}

Restituisci esclusivamente:

{{
  "headline": "",
  "intro": "",
  "items": [
    {{
      "event_id": "",
      "title": "",
      "what_happened": "",
      "why_it_matters": "",
      "what_changed": "",
      "uncertainty": "",
      "status": "CONFIRMED|REPORTED|UNVERIFIED",
      "sources": []
    }}
  ]
}}
"""

    return generate_json(
        prompt,
        max_tokens=3000,
    )
