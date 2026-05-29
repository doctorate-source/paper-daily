import datetime as dt
import os
import urllib.error
import unittest

from scripts.collect_papers import (
    Topic,
    arxiv_query_for_topic,
    arxiv_retry_wait_seconds,
    collection_cutoff,
    crossref_authors,
    crossref_date,
    crossref_query_for_topic,
    inverted_index_to_text,
    is_retryable_arxiv_error,
    merge_with_retained_papers,
    openalex_authors,
    openalex_categories,
    openalex_query_for_topic,
    phrase_terms_present,
    score_paper,
    strip_markup,
    trim_papers_for_storage,
)


def paper(paper_id: str, level: str, published: str, keyword_hits: list[str] | None = None) -> dict:
    return {
        "id": paper_id,
        "title": paper_id,
        "published": published,
        "best_match": {
            "topic_id": "topic",
            "topic_name": "Topic",
            "score": {"high": 0.9, "medium": 0.5, "low": 0.2}[level],
            "level": level,
            "reason": "test",
            "keyword_hits": keyword_hits or [],
        },
        "matches": [],
        "chinese_summary": {},
    }


class RetentionTest(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("ARXIV_RETRY_MIN_SECONDS", None)
        os.environ.pop("ARXIV_RETRY_BASE_SECONDS", None)
        os.environ.pop("ARXIV_RETRY_MAX_SECONDS", None)

    def test_arxiv_retry_wait_uses_retry_after_header(self) -> None:
        os.environ["ARXIV_RETRY_MIN_SECONDS"] = "30"
        error = urllib.error.HTTPError(
            "https://export.arxiv.org/api/query",
            429,
            "Too Many Requests",
            {"Retry-After": "75"},
            None,
        )

        self.assertEqual(arxiv_retry_wait_seconds(error, 0), 75.0)

    def test_arxiv_retry_wait_clamps_short_retry_after_header(self) -> None:
        os.environ["ARXIV_RETRY_MIN_SECONDS"] = "30"
        error = urllib.error.HTTPError(
            "https://export.arxiv.org/api/query",
            503,
            "Service Unavailable",
            {"Retry-After": "0"},
            None,
        )

        self.assertEqual(arxiv_retry_wait_seconds(error, 0), 30.0)

    def test_arxiv_retry_wait_uses_capped_backoff(self) -> None:
        os.environ["ARXIV_RETRY_MIN_SECONDS"] = "5"
        os.environ["ARXIV_RETRY_BASE_SECONDS"] = "10"
        os.environ["ARXIV_RETRY_MAX_SECONDS"] = "25"

        self.assertEqual(arxiv_retry_wait_seconds(TimeoutError("timed out"), 0), 10.0)
        self.assertEqual(arxiv_retry_wait_seconds(TimeoutError("timed out"), 2), 25.0)

    def test_arxiv_retryable_errors(self) -> None:
        rate_limited = urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)
        not_found = urllib.error.HTTPError("url", 404, "Not Found", {}, None)

        self.assertTrue(is_retryable_arxiv_error(rate_limited))
        self.assertTrue(is_retryable_arxiv_error(TimeoutError("timed out")))
        self.assertFalse(is_retryable_arxiv_error(not_found))

    def test_arxiv_query_prefers_keywords_without_category_gate(self) -> None:
        topic = Topic(
            id="soil_carbon",
            name="Soil carbon",
            description="",
            keywords=["soil organic carbon", "remote sensing"],
            arxiv_categories=["cs.LG", "stat.ML"],
        )

        query = arxiv_query_for_topic(topic)

        self.assertIn('all:"soil organic carbon"', query)
        self.assertIn('all:"remote sensing"', query)
        self.assertNotIn("cat:cs.LG", query)
        self.assertNotIn(" AND ", query)

    def test_crossref_helpers_normalize_metadata(self) -> None:
        item = {
            "published-online": {"date-parts": [[2026, 5, 29]]},
            "author": [{"given": "Jane", "family": "Doe"}],
        }

        self.assertEqual(crossref_date(item), "2026-05-29T00:00:00+00:00")
        self.assertEqual(crossref_authors(item), ["Jane Doe"])
        self.assertEqual(strip_markup("<jats:p>Soil &amp; carbon</jats:p>"), "Soil & carbon")

    def test_crossref_query_uses_topic_keywords(self) -> None:
        topic = Topic(
            id="soil_carbon",
            name="Soil carbon",
            description="",
            keywords=["soil organic carbon", "mineral-associated organic carbon"],
            arxiv_categories=[],
        )

        self.assertEqual(
            crossref_query_for_topic(topic),
            "soil organic carbon OR mineral-associated organic carbon",
        )

    def test_openalex_helpers_normalize_metadata(self) -> None:
        item = {
            "authorships": [{"author": {"display_name": "Jane Doe"}}],
            "primary_location": {"source": {"display_name": "Remote Sensing"}},
            "primary_topic": {"display_name": "Digital Soil Mapping"},
            "topics": [{"display_name": "Soil Carbon"}],
            "keywords": [{"display_name": "Remote sensing"}],
        }

        self.assertEqual(inverted_index_to_text({"Soil": [0], "carbon": [2], "organic": [1]}), "Soil organic carbon")
        self.assertEqual(openalex_authors(item), ["Jane Doe"])
        self.assertEqual(openalex_categories(item), ["Remote Sensing", "Digital Soil Mapping", "Soil Carbon", "Remote sensing"])

    def test_openalex_query_uses_first_keyword(self) -> None:
        topic = Topic(
            id="soil_carbon",
            name="Soil carbon",
            description="",
            keywords=["soil organic carbon remote sensing", "soil carbon mapping"],
            arxiv_categories=[],
        )

        self.assertEqual(openalex_query_for_topic(topic), "soil organic carbon remote sensing")

    def test_keyword_score_accepts_non_contiguous_phrase_terms(self) -> None:
        topic = Topic(
            id="remote_sensing",
            name="Remote sensing",
            description="",
            keywords=["soil organic carbon remote sensing"],
            arxiv_categories=[],
        )
        paper = {
            "title": "Prediction of Surface Soil Organic Carbon Based on Multi-Temporal Remote Sensing Data",
            "summary": "",
            "categories": [],
        }

        match = score_paper(topic, paper)

        self.assertTrue(phrase_terms_present("soil organic carbon remote sensing", paper["title"]))
        self.assertEqual(match["keyword_hits"], ["soil organic carbon remote sensing"])

    def test_merge_retains_previous_high_medium_and_drops_existing_low(self) -> None:
        now = dt.datetime(2026, 5, 28, tzinfo=dt.timezone.utc)
        stale_low = paper("old-low", "low", "2026-03-01T00:00:00+00:00")
        stale_low["first_seen_at"] = "2026-03-02T00:00:00+00:00"
        existing = {
            "generated_at_iso": "2026-05-27T00:00:00+00:00",
            "papers": [
                paper("old-high", "high", "2026-05-26T00:00:00+00:00"),
                paper("old-medium", "medium", "2026-05-25T00:00:00+00:00"),
                paper("recent-low", "low", "2026-05-24T00:00:00+00:00"),
                stale_low,
            ],
        }

        merged, stats = merge_with_retained_papers(
            [paper("new-low", "low", "2026-05-28T00:00:00+00:00", ["soil organic carbon"])],
            existing,
            now,
            recent_history_days=45,
        )

        self.assertEqual({item["id"] for item in merged}, {"new-low", "old-high", "old-medium"})
        self.assertEqual(stats["retained_paper_count"], 2)
        self.assertEqual(stats["retained_recent_low_count"], 0)
        self.assertEqual(stats["dropped_low_relevance_count"], 2)
        self.assertTrue(next(item for item in merged if item["id"] == "old-high")["retained_from_previous_run"])

    def test_collection_cutoff_uses_previous_run_for_incremental_mode(self) -> None:
        now = dt.datetime(2026, 5, 28, 22, tzinfo=dt.timezone.utc)
        cutoff, mode = collection_cutoff(
            {"generated_at_iso": "2026-05-27T22:00:00+00:00"},
            now,
            days=7,
            incremental_since_last_run=True,
        )

        self.assertEqual(mode, "incremental")
        self.assertEqual(cutoff, dt.datetime(2026, 5, 27, 22, tzinfo=dt.timezone.utc))

    def test_collection_cutoff_falls_back_to_lookback(self) -> None:
        now = dt.datetime(2026, 5, 28, 22, tzinfo=dt.timezone.utc)
        cutoff, mode = collection_cutoff({}, now, days=7, incremental_since_last_run=True)

        self.assertEqual(mode, "lookback")
        self.assertEqual(cutoff, dt.datetime(2026, 5, 21, 22, tzinfo=dt.timezone.utc))

    def test_storage_trim_removes_low_then_oldest(self) -> None:
        payload = {
            "generated_at_iso": "2026-05-28T00:00:00+00:00",
            "papers": [
                paper("newer-high", "high", "2026-05-28T00:00:00+00:00"),
                paper("older-high", "high", "2026-05-20T00:00:00+00:00"),
                paper("newer-low", "low", "2026-05-28T00:00:00+00:00"),
            ],
            "stats": {},
        }

        trimmed, stats = trim_papers_for_storage(payload, max_stored_papers=2, max_data_bytes=0)
        self.assertEqual({item["id"] for item in trimmed}, {"newer-high", "older-high"})
        self.assertEqual(stats["storage_trimmed_by_level"]["low"], 1)

        payload["papers"] = trimmed
        trimmed, stats = trim_papers_for_storage(payload, max_stored_papers=1, max_data_bytes=0)
        self.assertEqual([item["id"] for item in trimmed], ["newer-high"])
        self.assertEqual(stats["storage_trimmed_by_level"]["high"], 1)


if __name__ == "__main__":
    unittest.main()
