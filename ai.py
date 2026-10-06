import json
import os
from typing import Any

from openai import OpenAI


# ============================================================
# VERA - DEEPSEEK ONLY
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-flash"
)


class AIError(Exception):
    pass


# ============================================================
# DEEPSEEK CLIENT
# ============================================================

def get_client() -> OpenAI:

    if not DEEPSEEK_API_KEY:
        raise AIError(
            "DEEPSEEK_API_KEY non configurata su Render"
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text: str) -> dict[str, Any]:

    if not text:
        raise AIError(
            "DeepSeek ha restituito una risposta vuota"
        )

    text = text.strip()

    # Gestisce eventuale risposta:
    # ```json
    # {...}
    # ```

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

        raise AIError(
            "DeepSeek ha restituito un JSON non valido"
        )

    return data


# ============================================================
# CHIAMATA DEEPSEEK
# ============================================================

def generate_json(
    prompt: str,
    max_tokens: int = 2000
) -> dict[str, Any]:

    client = get_client()

    try:

        response = client.chat.completions.create(

            model=DEEPSEEK_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": """
Sei VERA, un motore editoriale giornalistico.

Usa esclusivamente le informazioni contenute
nelle fonti che ti vengono fornite.

Non inventare fatti, nomi, date, numeri,
luoghi, dichiarazioni o collegamenti.

Distingui sempre tra:
- fatti;
- dichiarazioni;
- accuse;
- informazioni riportate;
- interpretazioni.

Quando una notizia non può essere confermata,
devi indicarlo chiaramente.

Devi rispondere esclusivamente in formato JSON.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_tokens=max_tokens,

            response_format={
                "type": "json_object"
            }
        )

    except Exception as exc:

        raise AIError(
            f"Errore API DeepSeek: {exc}"
        ) from exc

    if not response.choices:

        raise AIError(
            "DeepSeek non ha restituito alcuna scelta"
        )

    message = response.choices[0].message

    content = message.content

    if not content:

        raise AIError(
            "DeepSeek ha restituito contenuto vuoto"
        )

    return parse_json(content)


# ============================================================
# COMPRESSIONE DELLE FONTI
# ============================================================

def compact_sources(
    articles: list[dict[str, Any]],
    max_sources: int = 3
) -> list[dict[str, Any]]:

    result = []

    for article in articles[:max_sources]:

        result.append({

            "id": article.get("id"),

            "outlet": article.get("outlet"),

            "published": article.get("published"),

            "title": str(
                article.get("title", "")
            )[:300],

            "summary": str(
                article.get("summary", "")
            )[:600]

        })

    return result


# ============================================================
# VERIFICA EDITORIALE
# ============================================================

def verify_event(
    event: dict[str, Any]
) -> dict[str, Any]:

    sources = compact_sources(
        event.get("articles", []),
        max_sources=3
    )

    event_info = {

        "id": event.get("id"),

        "title": str(
            event.get("title", "")
        )[:400],

        "category": event.get(
            "category",
            ""
        )

    }

    prompt = f"""
VERIFICA EDITORIALE VERA

Devi verificare l'evento giornalistico
descritto sotto.

USA ESCLUSIVAMENTE LE FONTI FORNITE.

NON usare conoscenze esterne.

REGOLE:

1. Una notizia non è automaticamente vera
   solo perché viene riportata da una fonte.

2. Distingui tra:
   - fatto;
   - dichiarazione;
   - accusa;
   - interpretazione.

3. Usa CONFIRMED quando le fonti forniscono
   elementi sufficienti per considerare
   il fatto confermato.

4. Usa REPORTED quando una fonte riporta
   il fatto ma non c'è sufficiente conferma
   indipendente.

5. Usa UNVERIFIED quando le fonti non
   permettono di stabilire cosa sia successo.

6. Non inventare informazioni mancanti.

7. Non attribuire a una fonte qualcosa
   che la fonte non dice.

8. Gli ID delle fonti devono essere copiati
   esattamente.

EVENTO:

{json.dumps(
    event_info,
    ensure_ascii=False
)}

FONTI:

{json.dumps(
    sources,
    ensure_ascii=False
)}

Restituisci ESCLUSIVAMENTE JSON.

Formato obbligatorio:

{{
    "status": "CONFIRMED",
    "confidence": 0,
    "confirmed_facts": [],
    "reported_claims": [],
    "contradictions": [],
    "important_uncertainties": [],
    "independent_source_ids": [],
    "reason": ""
}}

Il campo status deve essere esattamente uno di:

CONFIRMED
REPORTED
UNVERIFIED
"""

    return generate_json(
        prompt,
        max_tokens=2000
    )


# ============================================================
# BRIEFING VERA
# ============================================================

def build_briefing(
    events: list[dict[str, Any]],
    interests: list[str]
) -> dict[str, Any]:

    evidence = []

    # Massimo 6 eventi nel briefing.
    # Questo mantiene il prompt compatto.

    for event in events[:6]:

        verification = event.get(
            "verification",
            {}
        )

        sources = compact_sources(
            event.get("articles", []),
            max_sources=2
        )

        evidence.append({

            "event_id": event.get(
                "id"
            ),

            "title": str(
                event.get(
                    "title",
                    ""
                )
            )[:300],

            "category": event.get(
                "category",
                ""
            ),

            "verification": {

                "status": verification.get(
                    "status"
                ),

                "confidence": verification.get(
                    "confidence"
                ),

                "reason": str(
                    verification.get(
                        "reason",
                        ""
                    )
                )[:400]

            },

            "sources": sources

        })

    prompt = f"""
SEI VERA.

Devi creare una rassegna stampa
giornalistica in italiano.

USA ESCLUSIVAMENTE gli eventi e le fonti
fornite.

NON usare conoscenze esterne.

INTERESSI:

{json.dumps(
    interests or ["tutti"],
    ensure_ascii=False
)}

REGOLE EDITORIALI:

- Non inventare informazioni.
- Non aggiungere fatti non presenti
  nelle fonti.
- Non trasformare dichiarazioni
  in fatti.
- Mantieni le attribuzioni.
- Se lo status è REPORTED, fai capire
  chiaramente che la notizia è riportata
  dalle fonti.
- Non presentare UNVERIFIED come fatto.
- Scrivi in italiano naturale.
- Scrivi come una rassegna stampa
  professionale.
- Evita formule da chatbot.
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
    ensure_ascii=False
)}

Restituisci ESCLUSIVAMENTE JSON.

Formato obbligatorio:

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
            "status": "CONFIRMED",
            "sources": []
        }}
    ]
}}

Lo status deve essere uno di:

CONFIRMED
REPORTED
UNVERIFIED
"""

    return generate_json(
        prompt,
        max_tokens=3000
    )
