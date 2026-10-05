import os
from google import genai
from google.genai import errors

def genera_rassegna_stampa(testi_notizie):
    """
    Genera la rassegna stampa in modo sicuro, intercettando qualsiasi errore.
    """
    # 1. CONTROLLO PREVENTIVO DELLA CHIAVE API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ ERRORE DI SISTEMA: La variabile GOOGLE_API_KEY non è configurata su Render. Aggiungila nelle impostazioni di Environment."

    # 2. INIZIALIZZAZIONE SICURA DEL CLIENT
    try:
        # Passiamo la chiave esplicitamente per evitare ambiguità
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return f"⚠️ ERRORE DI INIZIALIZZAZIONE: Impossibile avviare il client Google. Dettaglio: {str(e)}"

    # 3. PROMPT STRUTTURATO PER IL COMPITO GIORNALISTICO
    # Qui definiamo esattamente come l'AI deve comportarsi
    prompt = f"""Sei un caporedattore esperto e un giornalista professionista.
Il tuo compito è creare una rassegna stampa istituzionale, chiara e impeccabile partendo dalle notizie fornite.

REGOLE FONDAMENTALI:
1. Tono: Giornalistico, oggettivo, formale.
2. Struttura: Assegna un Titolo in grassetto per ogni notizia, seguito da un riassunto conciso dei fatti chiave.
3. Accuratezza: Massima attenzione all'ortografia, all'uso corretto degli accenti e alla punteggiatura. Niente refusi.
4. Neutralità: Riporta i fatti, non aggiungere commenti o pareri personali.

NOTIZIE DA SINTETIZZARE:
{testi_notizie}

Genera la rassegna stampa:"""

    # 4. CHIAMATA ALL'API CON RETE DI SALVATAGGIO (TRY/EXCEPT)
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Puoi usare gemini-2.5-pro per analisi più complesse
            contents=prompt,
        )
        # Se la generazione va a buon fine, restituisce il testo
        return response.text

    # Gestisce gli errori specifici di Google (es. server down, limite di richieste superato)
    except errors.APIError as e:
        return f"⚠️ ERRORE API GOOGLE: Si è verificato un problema di comunicazione con Gemini. Dettaglio: {e.message}"
    
    # Gestisce qualsiasi altro errore imprevisto senza far crashare l'app
    except Exception as e:
        return f"⚠️ ERRORE IMPREVISTO DURANTE LA GENERAZIONE: {str(e)}"


# ESEMPIO DI UTILIZZO NEL TUO ENDPOINT (FastAPI, Flask, ecc.):
# 
# notizie_grezze = "..."
# risultato = genera_rassegna_stampa(notizie_grezze)
# return {"rassegna": risultato}
