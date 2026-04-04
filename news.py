"""News retrieval utilities for downstream sentiment processing."""

from __future__ import annotations

import json
import re
from html import unescape
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from schemas import Article


_USER_AGENT = "Mozilla/5.0 (compatible; ScarletHacksNewsBot/1.0)"
_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_ARTICLE_FETCH_TIMEOUT = 10


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def _normalize_title(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _parse_published(raw_date: str | None) -> datetime | None:
    if not raw_date:
        return None
    try:
        parsed = parsedate_to_datetime(raw_date)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _extract_items(rss_url: str) -> list[dict[str, Any]]:
    request = Request(rss_url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=12) as response:
        xml_payload = response.read()

    root = ET.fromstring(xml_payload)
    out: list[dict[str, Any]] = []
    for item in root.findall("./channel/item"):
        title = _clean_text(item.findtext("title"))
        link = _clean_text(item.findtext("link"))
        description = _strip_html_text(item.findtext("description") or "")
        rss_content = _extract_rss_content(item)
        pub_date = item.findtext("pubDate")

        source_node = item.find("source")
        source = _clean_text(source_node.text if source_node is not None else None)
        if not source:
            source = "Unknown"

        published = _parse_published(pub_date)
        if not title or not link or not published:
            continue

        out.append(
            {
                "title": title,
                "url": link,
                "source": source,
                "description": description,
                "content": rss_content,
                "published": published,
            }
        )
    return out


def _extract_rss_content(item: ET.Element) -> str | None:
    content_node = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
    if content_node is not None and content_node.text:
        return _strip_html_text(content_node.text)

    # Fallback for feeds using different prefixes or custom namespaces.
    for child in item:
        if child.tag.lower().endswith("encoded") and child.text:
            return _strip_html_text(child.text)
    return None


def _strip_html_text(html: str) -> str | None:
    # Remove non-content blocks first, then strip tags to produce readable text.
    no_scripts = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html)
    no_tags = re.sub(r"(?is)<[^>]+>", " ", no_scripts)
    text = _clean_text(unescape(no_tags))
    return text


def _extract_article_content(url: str) -> str | None:
    import trafilatura
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text:
            return _clean_text(text)
    # fallback: manual fetch + tag strip
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=_ARTICLE_FETCH_TIMEOUT) as response:
            payload = response.read()
    except Exception:
        return None
    html = payload.decode("utf-8", errors="ignore")
    return _strip_html_text(html)


def _is_relevant(item: dict[str, Any], ticker: str, company_name: str) -> bool:
    title = (item.get("title") or "").lower()
    description = (item.get("description") or "").lower()
    content = (item.get("content") or "").lower()
    blob = f"{title} {description} {content}".strip()

    ticker_token = ticker.lower().strip()
    if re.search(rf"\b{re.escape(ticker_token)}\b", title):
        return True

    if re.search(rf"\b{re.escape(ticker_token)}\b", blob):
        # Accept ticker mentions in body-level text only when the company is also named.
        if company_name and company_name.lower() in blob:
            return True

    company = company_name.lower().strip()
    if company and company in title:
        return True

    if company and company in blob and any(word in title for word in company.split() if len(word) >= 4):
        return True

    ignore_tokens = {"inc", "corp", "co", "ltd", "plc", "group", "holdings", "company"}
    words = [w for w in re.split(r"\W+", company) if len(w) >= 4 and w not in ignore_tokens]
    if not words:
        return False

    matched_title = sum(1 for word in words if re.search(rf"\b{re.escape(word)}\b", title))
    if matched_title >= 1:
        return True

    matched_blob = sum(1 for word in words if re.search(rf"\b{re.escape(word)}\b", blob))
    return matched_blob >= min(2, len(words))


def get_news(
    ticker: str,
    company_name: str,
    lookback_days: int,
    max_articles: int = 25,
) -> list[Article]:
    """Retrieve recent, company-relevant, deduplicated article records."""
    ticker = ticker.upper().strip()
    company_name = company_name.strip()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))

    query = quote_plus(f'"{ticker}" "{company_name}" stock OR earnings OR guidance')
    feeds = [_YAHOO_RSS.format(ticker=ticker), _GOOGLE_RSS.format(query=query)]

    raw_items: list[dict[str, Any]] = []
    for feed in feeds:
        try:
            raw_items.extend(_extract_items(feed))
        except Exception:
            # Keep pipeline resilient if one provider fails.
            continue

    filtered: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for item in raw_items:
        published = item["published"]
        if published < cutoff:
            continue
        if not _is_relevant(item, ticker=ticker, company_name=company_name):
            continue

        normalized_title = _normalize_title(item["title"])
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        filtered.append(item)

    filtered.sort(key=lambda i: i["published"], reverse=True)
    trimmed = filtered[:max_articles]

    for item in trimmed:
        fetched_content = _extract_article_content(item["url"])
        if fetched_content:
            item["content"] = fetched_content

    return [
        Article(
            ticker=ticker,
            company_name=company_name,
            title=item["title"],
            source=item["source"],
            published_at=item["published"].isoformat(),
            url=item["url"],
            description=item.get("description"),
            content=item.get("content"),
        )
        for item in trimmed
    ]


def get_news_json(ticker: str, company_name: str, lookback_days: int) -> str:
    """Return JSON-only payload of raw article fields for downstream use."""
    articles = get_news(ticker=ticker, company_name=company_name, lookback_days=lookback_days)
    payload = [article.__dict__ for article in articles]
    return json.dumps(payload, ensure_ascii=True)
