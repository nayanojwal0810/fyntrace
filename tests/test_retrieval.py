"""Tests for BM25 retrieval engine using SQLite FTS5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import pytest

from fyntrace.retrieval import (
    build_index,
    get_index_metadata,
    search,
    tokenize_query,
    build_match_expression,
)


def _make_synthetic_corpus(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Create a synthetic chunks.jsonl file with diverse chunks."""
    chunks = [
        {
            "chunk_id": "MSFT_2024_10K_000095017024087843_00001",
            "ticker": "MSFT",
            "cik": "0000789019",
            "fiscal_year": 2024,
            "form": "10-K",
            "accession": "0000950170-24-087843",
            "section": "Item 7 - Management's Discussion and Analysis",
            "section_confidence": "high",
            "chunk_type": "text",
            "text": "Microsoft Cloud revenue grew 22 percent to 137 billion dollars driven by Azure expansion.",
            "char_start": 100,
            "char_end": 190,
            "source_url": "https://www.sec.gov/edgar/msft-2024.htm",
            "source_file_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "canonical_text_path": "data/processed/text/msft_2024.txt",
            "filing_date": "2024-07-30",
            "period_end_date": "2024-06-30",
            "table_title": None,
        },
        {
            "chunk_id": "MSFT_2024_10K_000095017024087843_00002",
            "ticker": "MSFT",
            "cik": "0000789019",
            "fiscal_year": 2024,
            "form": "10-K",
            "accession": "0000950170-24-087843",
            "section": "Item 8 - Financial Statements",
            "section_confidence": "high",
            "chunk_type": "table",
            "text": "[TABLE START]\nSegment | Revenue 2024 | Revenue 2023\nIntelligent Cloud | 105362 | 87907\n[TABLE END]",
            "char_start": 200,
            "char_end": 300,
            "source_url": "https://www.sec.gov/edgar/msft-2024.htm",
            "source_file_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "canonical_text_path": "data/processed/text/msft_2024.txt",
            "filing_date": "2024-07-30",
            "period_end_date": "2024-06-30",
            "table_title": "Segment Revenues",
        },
        {
            "chunk_id": "MSFT_2023_10K_000095017023035122_00001",
            "ticker": "MSFT",
            "cik": "0000789019",
            "fiscal_year": 2023,
            "form": "10-K",
            "accession": "0000950170-23-035122",
            "section": "Item 7 - Management's Discussion and Analysis",
            "section_confidence": "high",
            "chunk_type": "text",
            "text": "In fiscal year 2023 Microsoft Cloud revenue was 111 billion dollars.",
            "char_start": 50,
            "char_end": 118,
            "source_url": "https://www.sec.gov/edgar/msft-2023.htm",
            "source_file_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "canonical_text_path": "data/processed/text/msft_2023.txt",
            "filing_date": "2023-07-27",
            "period_end_date": "2023-06-30",
            "table_title": None,
        },
        {
            "chunk_id": "AAPL_2024_10K_000032019324000123_00001",
            "ticker": "AAPL",
            "cik": "0000320193",
            "fiscal_year": 2024,
            "form": "10-K",
            "accession": "0000320193-24-000123",
            "section": "Item 1A - Risk Factors",
            "section_confidence": "high",
            "chunk_type": "text",
            "text": "Global supply chain disruptions and semiconductor shortages could affect iPhone production.",
            "char_start": 300,
            "char_end": 390,
            "source_url": "https://www.sec.gov/edgar/aapl-2024.htm",
            "source_file_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "canonical_text_path": "data/processed/text/aapl_2024.txt",
            "filing_date": "2024-11-01",
            "period_end_date": "2024-09-28",
            "table_title": None,
        },
        {
            "chunk_id": "JPM_2024_10K_000001961724000273_00001",
            "ticker": "JPM",
            "cik": "0000019617",
            "fiscal_year": 2024,
            "form": "10-K",
            "accession": "0000019617-24-000273",
            "section": "Item 7 - Management's Discussion and Analysis",
            "section_confidence": "high",
            "chunk_type": "text",
            "text": "JPMorgan Chase reported net interest income of 90 billion dollars for the year.",
            "char_start": 400,
            "char_end": 479,
            "source_url": "https://www.sec.gov/edgar/jpm-2024.htm",
            "source_file_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "canonical_text_path": "data/processed/text/jpm_2024.txt",
            "filing_date": "2024-02-16",
            "period_end_date": "2023-12-31",
            "table_title": None,
        },
        {
            "chunk_id": "WMT_2024_10K_000010416924000033_00001",
            "ticker": "WMT",
            "cik": "0000104169",
            "fiscal_year": 2024,
            "form": "10-K",
            "accession": "0000104169-24-000033",
            "section": "Item 1 - Business",
            "section_confidence": "high",
            "chunk_type": "text",
            "text": "Walmart operates retail discount department stores and wholesale warehouse clubs.",
            "char_start": 500,
            "char_end": 581,
            "source_url": "https://www.sec.gov/edgar/wmt-2024.htm",
            "source_file_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "canonical_text_path": "data/processed/text/wmt_2024.txt",
            "filing_date": "2024-03-22",
            "period_end_date": "2024-01-31",
            "table_title": None,
        },
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return chunks_path, chunks


def test_correct_indexing_of_narrative_and_table_chunks(tmp_path: Path):
    """Both narrative and table chunks are indexed and searchable with metadata."""
    chunks_path, chunks = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"

    summary = build_index(chunks_path=chunks_path, db_path=db_path)
    assert summary["indexed_chunks"] == 6
    assert db_path.is_file()

    meta = get_index_metadata(db_path)
    assert meta["chunk_count"] == "6"
    assert "corpus_sha256" in meta

    # Search for term in narrative chunk
    results_text = search("cloud revenue", db_path=db_path, top_k=5)
    types_text = {r["chunk_type"] for r in results_text}
    assert "text" in types_text

    # Search for term in table chunk
    results_table = search("intelligent cloud segment", db_path=db_path, top_k=5)
    types_table = {r["chunk_type"] for r in results_table}
    assert "table" in types_table


def test_metadata_filtering(tmp_path: Path):
    """Filters for ticker, year, section, accession, and chunk_type work accurately."""
    chunks_path, _ = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"
    build_index(chunks_path=chunks_path, db_path=db_path)

    # Filter by ticker
    res_ticker = search("revenue", ticker="MSFT", db_path=db_path, top_k=10)
    assert len(res_ticker) > 0
    assert all(r["ticker"] == "MSFT" for r in res_ticker)

    # Filter by year
    res_year = search("revenue", year=2023, db_path=db_path, top_k=10)
    assert len(res_year) > 0
    assert all(r["fiscal_year"] == 2023 for r in res_year)

    # Filter by accession
    acc = "0000950170-24-087843"
    res_acc = search("cloud", accession=acc, db_path=db_path, top_k=10)
    assert len(res_acc) > 0
    assert all(r["accession"] == acc for r in res_acc)

    # Filter by section
    sec = "Item 8 - Financial Statements"
    res_sec = search("revenue", section=sec, db_path=db_path, top_k=10)
    assert len(res_sec) == 1
    assert res_sec[0]["section"] == sec

    # Filter by chunk_type
    res_type = search("revenue", chunk_type="table", db_path=db_path, top_k=10)
    assert len(res_type) == 1
    assert res_type[0]["chunk_type"] == "table"


def test_results_never_crossing_requested_metadata_filters(tmp_path: Path):
    """Search results must strictly stay within all requested metadata filters."""
    chunks_path, _ = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"
    build_index(chunks_path=chunks_path, db_path=db_path)

    # Broad query matching almost everything, but strictly filter to JPM 2024
    res = search("dollars revenue year income", ticker="JPM", year=2024, db_path=db_path, top_k=10)
    assert len(res) == 1
    assert res[0]["ticker"] == "JPM"
    assert res[0]["fiscal_year"] == 2024
    assert res[0]["chunk_id"] == "JPM_2024_10K_000001961724000273_00001"

    # Strict non-matching filter should return empty list, not cross boundaries
    res_none = search("revenue", ticker="WMT", year=2023, db_path=db_path, top_k=5)
    assert res_none == []


def test_stable_ranking_across_searches_and_rebuilds(tmp_path: Path):
    """Repeated searches and rebuilds from identical source chunks produce identical rankings."""
    chunks_path, _ = _make_synthetic_corpus(tmp_path)
    db_path1 = tmp_path / "index1.sqlite"
    db_path2 = tmp_path / "index2.sqlite"

    build_index(chunks_path=chunks_path, db_path=db_path1)
    build_index(chunks_path=chunks_path, db_path=db_path2)

    query = "Microsoft Cloud revenue dollars"
    res1_a = search(query, db_path=db_path1, top_k=5)
    res1_b = search(query, db_path=db_path1, top_k=5)
    res2 = search(query, db_path=db_path2, top_k=5)

    assert len(res1_a) == len(res1_b) == len(res2)
    ids1_a = [r["chunk_id"] for r in res1_a]
    ids1_b = [r["chunk_id"] for r in res1_b]
    ids2 = [r["chunk_id"] for r in res2]
    scores1_a = [r["score"] for r in res1_a]
    scores2 = [r["score"] for r in res2]

    assert ids1_a == ids1_b == ids2
    assert scores1_a == scores2


def test_correct_source_metadata_and_unchanged_text_offsets(tmp_path: Path):
    """Search results contain all original chunk metadata fields and exact character offsets."""
    chunks_path, orig_chunks = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"
    build_index(chunks_path=chunks_path, db_path=db_path)

    results = search("semiconductor shortages iPhone", ticker="AAPL", db_path=db_path, top_k=1)
    assert len(results) == 1
    r = results[0]
    orig = orig_chunks[3]  # AAPL chunk

    assert r["chunk_id"] == orig["chunk_id"]
    assert r["ticker"] == orig["ticker"]
    assert r["company"] == orig["ticker"]
    assert r["cik"] == orig["cik"]
    assert r["fiscal_year"] == orig["fiscal_year"]
    assert r["year"] == orig["fiscal_year"]
    assert r["accession"] == orig["accession"]
    assert r["section"] == orig["section"]
    assert r["chunk_type"] == orig["chunk_type"]
    assert r["text"] == orig["text"]
    assert r["chunk_text"] == orig["text"]
    assert r["char_start"] == orig["char_start"]
    assert r["char_end"] == orig["char_end"]
    assert r["source_url"] == orig["source_url"]
    assert r["source_file_sha256"] == orig["source_file_sha256"]
    assert r["source_hash"] == orig["source_file_sha256"]
    assert r["canonical_text_path"] == orig["canonical_text_path"]
    assert isinstance(r["score"], float)
    assert r["relevance_score"] == r["score"]


def test_queries_with_punctuation_and_special_characters(tmp_path: Path):
    """Punctuation, symbols, quotes, and operator keywords do not break FTS5 MATCH."""
    chunks_path, _ = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"
    build_index(chunks_path=chunks_path, db_path=db_path)

    tricky_queries = [
        "Microsoft's cloud revenue (FY24)?",
        "10-K",
        "Item 1A: risk factors; supply chain!",
        "OR AND NOT",
        'quote: "azure" & cloud*',
        "Revenue: +22% / -10%",
        "What was Walmart's wholesale warehouse business???",
    ]

    for q in tricky_queries:
        # Must execute without syntax errors
        results = search(q, db_path=db_path, top_k=5)
        assert isinstance(results, list)


def test_empty_results_and_invalid_inputs(tmp_path: Path):
    """Empty queries, punctuation-only queries, and invalid inputs are handled safely."""
    chunks_path, _ = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"
    build_index(chunks_path=chunks_path, db_path=db_path)

    # Empty and whitespace queries
    assert search("", db_path=db_path) == []
    assert search("   ", db_path=db_path) == []
    assert search("??? !!! @@@", db_path=db_path) == []

    # Non-matching query
    assert search("quantum teleportation crypto", db_path=db_path) == []

    # Invalid top_k
    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 20"):
        search("cloud", top_k=0, db_path=db_path)
    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 20"):
        search("cloud", top_k=25, db_path=db_path)

    # Invalid query type
    with pytest.raises(ValueError, match="Query must be a string"):
        search(None, db_path=db_path)  # type: ignore

    # Missing database
    with pytest.raises(FileNotFoundError, match="Retrieval database not found"):
        search("cloud", db_path=tmp_path / "nonexistent.sqlite")


def test_failed_index_rebuild_preserves_existing_valid_index(tmp_path: Path, monkeypatch):
    """An indexing failure during rebuild preserves the previous working index untouched."""
    chunks_path, _ = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "protected_retrieval.sqlite"

    # Build initial valid index
    build_index(chunks_path=chunks_path, db_path=db_path)
    initial_res = search("cloud", db_path=db_path, top_k=5)
    assert len(initial_res) > 0
    orig_db_bytes = db_path.read_bytes()

    # Simulate failure during a rebuild attempt (e.g. error rebuilding FTS5)
    real_sha256 = hashlib.sha256
    def fail_after_start(*args, **kwargs):
        raise RuntimeError("Simulated failure mid-index rebuild")

    # Monkeypatch compute_file_sha256 or something in the build process
    monkeypatch.setattr("fyntrace.retrieval.compute_file_sha256", fail_after_start)

    with pytest.raises(RuntimeError, match="Simulated failure mid-index rebuild"):
        build_index(chunks_path=chunks_path, db_path=db_path)

    # The existing database file must be completely preserved
    assert db_path.read_bytes() == orig_db_bytes
    post_res = search("cloud", db_path=db_path, top_k=5)
    assert len(post_res) == len(initial_res)

    # Ensure no temporary files remained
    assert not (tmp_path / f"{db_path.name}.tmp").exists()


def test_no_network_requests(tmp_path: Path, monkeypatch):
    """Retrieval index building and searching must never perform network requests."""
    chunks_path, _ = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "network_test.sqlite"

    def forbid_network(*args, **kwargs):
        raise RuntimeError("Unexpected network access detected during retrieval operation.")

    monkeypatch.setattr(socket, "socket", forbid_network)

    build_index(chunks_path=chunks_path, db_path=db_path)
    results = search("Microsoft Cloud revenue", ticker="MSFT", year=2024, db_path=db_path)
    assert len(results) > 0
