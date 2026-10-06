import re
from collections import defaultdict
from rapidfuzz import fuzz


STOP = {
    "della","delle","degli","dello","dell","nella","nelle","negli","sulla","sulle",
    "con","per","tra","fra","che","come","dopo","prima","oggi","ieri","sono","una",
    "uno","gli","dei","del","dal","alla","alle","anche","questa","questo","news",
    "the","and","for","with","from","after","before","has","have","will","says"
}


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zàèéìòù0-9]{3,}", text.lower())
    return {w for w in words if w not in STOP}


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    fuzzy = fuzz.token_set_ratio(a, b) / 100
    return max(overlap, fuzzy * 0.8)


def cluster_articles(articles: list[dict]) -> list[dict]:
    """
    Clustering locale: nessuna chiamata AI.
    Questo evita di spendere richieste per ogni RSS.
    """
    events = []

    for article in sorted(articles, key=lambda x: x["published"], reverse=True):
        best = None
        best_score = 0.0
        for event in events:
            # Non unire categorie completamente diverse.
            if event["category"] != article["category"]:
                continue
            score = similarity(article["title"], event["title"])
            if score > best_score:
                best_score = score
                best = event

        if best is not None and best_score >= 0.58:
            best["articles"].append(article)
        else:
            events.append({
                "id": "evt_" + article["id"],
                "title": article["title"],
                "category": article["category"],
                "articles": [article],
            })

    # Ordina per numero di fonti e freschezza.
    events.sort(key=lambda e: (
        len({a["outlet"] for a in e["articles"]}),
        e["articles"][0]["published"]
    ), reverse=True)

    return events
