from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import unittest
from unittest.mock import patch

from utilities.news import get_news, get_news_json


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _rss_item(title: str, link: str, pub_date: datetime, source: str, description: str = "") -> str:
    return (
        "<item>"
        f"<title>{title}</title>"
        f"<link>{link}</link>"
        f"<description>{description}</description>"
        f"<pubDate>{format_datetime(pub_date)}</pubDate>"
        f"<source>{source}</source>"
        "</item>"
    )


def _rss_doc(items: list[str]) -> str:
    return "<?xml version='1.0' encoding='UTF-8'?><rss><channel>" + "".join(items) + "</channel></rss>"


class NewsTests(unittest.TestCase):
    def test_get_news_filters_and_deduplicates(self) -> None:
        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=1)
        old = now - timedelta(days=20)

        yahoo_xml = _rss_doc(
            [
                _rss_item(
                    title="AAPL beats expectations in quarterly earnings",
                    link="https://example.com/yahoo/aapl-earnings",
                    pub_date=recent,
                    source="Yahoo Finance",
                    description="Apple Inc. reported strong demand.",
                ),
                _rss_item(
                    title="AAPL beats expectations in quarterly earnings",
                    link="https://example.com/yahoo/aapl-earnings-dup",
                    pub_date=recent,
                    source="Yahoo Finance",
                    description="Duplicate headline to test dedupe.",
                ),
                _rss_item(
                    title="Apple unveils new enterprise partnership",
                    link="https://example.com/yahoo/aapl-partnership",
                    pub_date=recent - timedelta(hours=1),
                    source="Yahoo Finance",
                    description="Apple Inc. and a cloud vendor announced a new deal.",
                ),
            ]
        )

        google_xml = _rss_doc(
            [
                _rss_item(
                    title="S&P 500 mixed as treasury yields rise",
                    link="https://example.com/google/market-macro",
                    pub_date=recent,
                    source="Reuters",
                    description="Broad market update without company mention.",
                ),
                _rss_item(
                    title="AAPL legal update from prior year",
                    link="https://example.com/google/aapl-old",
                    pub_date=old,
                    source="Reuters",
                    description="Old story outside lookback window.",
                ),
            ]
        )

        def _fake_urlopen(request, timeout=12):
            url = request.full_url
            if "feeds.finance.yahoo.com" in url:
                return _FakeResponse(yahoo_xml)
            if "news.google.com" in url:
                return _FakeResponse(google_xml)
            if "example.com" in url:
                return _FakeResponse(
                    "<html><body><article>Apple reported results and guidance updates.</article></body></html>"
                )
            raise AssertionError(f"Unexpected URL fetched: {url}")

        with patch("news.urlopen", side_effect=_fake_urlopen):
            articles = get_news(ticker="AAPL", company_name="Apple Inc.", lookback_days=7, max_articles=25)

        self.assertEqual(len(articles), 2)
        self.assertTrue(all(a.ticker == "AAPL" for a in articles))
        self.assertTrue(all(a.company_name == "Apple Inc." for a in articles))
        self.assertGreaterEqual(articles[0].published_at, articles[1].published_at)

        titles = [a.title for a in articles]
        self.assertNotIn("S&P 500 mixed as treasury yields rise", titles)
        self.assertNotIn("AAPL legal update from prior year", titles)
        self.assertIn("Apple reported results and guidance updates.", articles[0].content or "")


    def test_get_news_json_returns_required_fields(self) -> None:
        now = datetime.now(timezone.utc)
        yahoo_xml = _rss_doc(
            [
                _rss_item(
                    title="AAPL product launch expands services bundle",
                    link="https://example.com/yahoo/aapl-launch",
                    pub_date=now - timedelta(days=1),
                    source="Yahoo Finance",
                    description="Apple Inc. launches a new subscription package.",
                )
            ]
        )
        google_xml = _rss_doc([])

        def _fake_urlopen(request, timeout=12):
            url = request.full_url
            if "feeds.finance.yahoo.com" in url:
                return _FakeResponse(yahoo_xml)
            if "news.google.com" in url:
                return _FakeResponse(google_xml)
            if "example.com" in url:
                return _FakeResponse(
                    "<html><body><p>Apple launched a new product and services bundle.</p></body></html>"
                )
            raise AssertionError(f"Unexpected URL fetched: {url}")

        with patch("news.urlopen", side_effect=_fake_urlopen):
            raw = get_news_json(ticker="AAPL", company_name="Apple Inc.", lookback_days=7)

        payload = json.loads(raw)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)

        required = {
            "ticker",
            "company_name",
            "title",
            "source",
            "published_at",
            "url",
            "description",
            "content",
        }
        self.assertTrue(required.issubset(payload[0].keys()))
        self.assertIn("Apple launched a new product and services bundle.", payload[0]["content"])


    def test_get_news_keeps_full_article_context(self) -> None:
        now = datetime.now(timezone.utc)
        yahoo_xml = _rss_doc(
            [
                _rss_item(
                    title="AAPL management discusses outlook in depth",
                    link="https://example.com/yahoo/aapl-context",
                    pub_date=now - timedelta(hours=2),
                    source="Yahoo Finance",
                    description="Apple Inc. leadership discussed strategy and demand.",
                )
            ]
        )
        google_xml = _rss_doc([])

        long_text = (
            "Sentence one. Sentence two. Sentence three with more context. "
            "Sentence four adds additional detail on guidance. "
            "Sentence five explains regional performance."
        )

        def _fake_urlopen(request, timeout=12):
            url = request.full_url
            if "feeds.finance.yahoo.com" in url:
                return _FakeResponse(yahoo_xml)
            if "news.google.com" in url:
                return _FakeResponse(google_xml)
            if "example.com" in url:
                return _FakeResponse(f"<html><body><article>{long_text}</article></body></html>")
            raise AssertionError(f"Unexpected URL fetched: {url}")

        with patch("news.urlopen", side_effect=_fake_urlopen):
            articles = get_news(ticker="AAPL", company_name="Apple Inc.", lookback_days=7)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].content, long_text)


if __name__ == "__main__":
    unittest.main()
