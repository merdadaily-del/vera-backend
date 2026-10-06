import json
import os
from typing import Any

from groq import Groq
from google import genai
from google.genai import types


GROQ_MODEL_FAST = os.getenv(
    "GROQ_MODEL_FAST",
    "openai/gpt-oss-20b"
)

GROQ_MODEL_DEEP = os.getenv(
    "GROQ_MODEL_DEEP",
    "openai/gpt-oss-120b"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


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
        raise AIError(
            f"Risposta AI non valida come JSON: {exc}"
        ) from exc


def _groq(
    prompt: str,
    model: str,
    max_tokens: int = 700
) -> dict[str, Any]:

    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise AIError(
            "GROQ_API_KEY non configurata"
        )

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
                    "Non inventare fatti. "
                    "Usa esclusivamente le fonti fornite. "
                    "Distingui fatti, dichiarazioni, accuse "
                    "e interpretazioni. "
                    "Rispondi SOLO con JSON valido."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        response_format={
            "type": "json_object"
        },
    )

    return _json_from_text(
        response.choices[0].message.content
    )


def _gemini(
    prompt: str,
    max_tokens: int = 700
) -> dict[str, Any]:

    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not key:
        raise AIError(
            "GEMINI_API_KEY/GOOGLE_API_KEY non configurata"
        )

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


def generate_json(
    prompt: str,
    deep: bool = False
) -> dict[str, Any]:

    # Per VERA usiamo il modello veloce anche
    # nelle operazioni che prima usavano il 120B.
    #
    # Il motivo è il limite TPM gratuito di Groq:
    # 8000 token/minuto.

    groq_model = GROQ_MODEL_FAST

    try:

        return _groq(
            prompt,
            groq_model,
            max_tokens=700
        )

    except Exception as groq_error:

        try:

            return _gemini(
                prompt,
                max_tokens=700
            )

        except Exception as gemini_error:

            raise AIError(
                f"Groq fallito ({groq_error}); "
                f"Gemini fallback fallito ({gemini_error})"
            ) from gemini_error


def _compact_sources(
    sources: list[dict[str, Any]],
    max_sources: int = 3
) -> list[dict[str, Any]]:

    compact = []

    for article in sources[:max_sources]:

        compact.append({
            "id": article["id"],
            "outlet": article["outlet"],
            "published": article["published"],
            "title": article["title"][:300],
            "summary": article.get(
                "summary",
                ""
            )[:400],
        })

    return compact


def verify_event(
    event: dict[str, Any]
) -> dict[str, Any]:

    # Massimo 3 fonti per evento.
    sources = _compact_sources(
        event["articles"],
        max_sources=3
    )

    prompt = f"""
VERIFICA EDITORIALE VERA.

Verifica l'evento usando SOLO le fonti fornite.

Regole:
- non usare conoscenze esterne;
- non considerare automaticamente vere le informazioni;
- distingui fatti e dichiarazioni;
- se non c'è conferma sufficiente usa REPORTED;
- usa CONFIRMED solo quando l'evidenza è sufficiente;
- usa UNVERIFIED quando non è possibile stabilire il fatto;
- non inventare informazioni.

EVENTO:
{json.dumps({
    "title": event["title"],
    "category": event["category"]
}, ensure_ascii=False)}

FONTI:
{json.dumps(
    sources,
    ensure_ascii=False
)}

Restituisci SOLO questo JSON:

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
        deep=False
    )


def build_briefing(
    events: list[dict[str, Any]],
    interests: list[str]
) -> dict[str, Any]:

    evidence = []

    # MASSIMO 4 eventi nella rassegna finale.
    #
    # Questo è fondamentale per non superare
    # il limite gratuito di Groq.

    for event in events[:4]:

        sources = _compact_sources(
            event["articles"],
            max_sources=2
        )

        evidence.append({
            "event_id": event["id"],
            "title": event["title"][:300],
            "category": event["category"],

            "verification": {
                "status": event.get(
                    "verification",
                    {}
                ).get("status"),

                "confidence": event.get(
                    "verification",
                    {}
                ).get("confidence"),

                "reason": event.get(
                    "verification",
                    {}
                ).get("reason", "")[:300],
            },

            "sources": sources,
        })

    prompt = f"""
SEI VERA, una rassegna stampa giornalistica.

Interessi:
{", ".join(interests) if interests else "tutti"}

Crea una breve rassegna SOLO usando gli eventi forniti.

Non inventare fatti.
Non usare informazioni esterne.
Non copiare le frasi delle fonti.
Scrivi in italiano giornalistico naturale.

Per ogni evento:
- cosa è successo;
- perché è importante;
- cosa succede ora;
- eventuali incertezze.

Mantieni le attribuzioni quando un fatto è solo riferito.

EVENTI:
{json.dumps(
    evidence,
    ensure_ascii=False
)}

Restituisci SOLO JSON:

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
        deep=False
    )
