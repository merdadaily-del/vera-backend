import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx


TIMEOUT = float(__import__("os").getenv("RSS_TIMEOUT_SECONDS", "8"))
MAX_ITEMS_PER_FEED = int(__import__("os").getenv("MAX_ITEMS_PER_FEED", "25"))


def _iso(entry) -> str:
    for key in ("published_parsed", "updated_parsed"):
        value = getattr(entry, key, None)
        if value:
            return datetime(*value[:6], tzinfo=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _id(outlet: str, entry) -> str:
    raw = str(getattr(entry, "id", "") or getattr(entry, "link", "") or getattr(entry, "title", ""))
    return hashlib.sha1(f"{outlet}|{raw}".encode()).hexdigest()


async def fetch_feed(client: httpx.AsyncClient, feed: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        r = await client.get(feed["url"], timeout=TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
        out = []
        for entry in parsed.entries[:MAX_ITEMS_PER_FEED]:
            title = str(getattr(entry, "title", "")).strip()
            link = str(getattr(entry, "link", "")).strip()
            if not title or not link:
                continue
            summary = str(
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
                or ""
            )
            out.append({
                "id": _id(feed["name"], entry),
                "outlet": feed["name"],
                "category": feed["cat"],
                "title": title[:500],
                "summary": summary[:3000],
                "link": link,
                "published": _iso(entry),
                "feed_url": feed["url"],
            })
        return out
    except Exception:
        return []


async def fetch_all(feeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limits = httpx.Limits(max_connections=40, max_keepalive_connections=20)
    async with httpx.AsyncClient(
        headers={"User-Agent": "VERA/1.0 (+news research app)"},
        limits=limits,
    ) as client:
        batches = await asyncio.gather(
            *(fetch_feed(client, feed) for feed in feeds),
            return_exceptions=True,
        )
    articles = []
    for batch in batches:
        if isinstance(batch, list):
            articles.extend(batch)

    # Deduplica URL/titoli identici.
    seen = set()
    unique = []
    for a in articles:
        key = a["link"].split("#")[0].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)
    return unique
