"""Automated unit tests for source-aware BM25 evidence retrieval."""

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from src.retrieval import (
    DEFAULT_STOPWORDS,
    build_index,
    load_chunks,
    main,
    parse_args,
    search_evidence,
    tokenize,
)


def make_valid_chunk(
    chunk_id: str = "TEST_2024_10K_000000000024000001_0001",
    ticker: str = "TEST",
    fiscal_year: int = 2024,
    text: str = "Sample financial evidence text.",
    char_start: int = 0,
    section: str = "Item 1 - Business",
    section_confidence: str = "high",
    accession: str = "0000000000-24-000001",
    cik: str = "0000000001",
    form: str = "10-K",
    source_url: str = "https://www.sec.gov/Archives/edgar/data/1/000000000024000001/test-20241231.htm",
    file_sha256: str = "a" * 64,
) -> dict:
    """Helper to create a schema-compliant mock filing chunk."""
    return {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "company_name": f"{ticker} CORP",
        "cik": cik,
        "fiscal_year": fiscal_year,
        "form": form,
        "accession": accession,
        "report_date": f"{fiscal_year}-12-31",
        "section": section,
        "section_confidence": section_confidence,
        "text": text,
        "char_start": char_start,
        "char_end": char_start + len(text),
        "source_url": source_url,
        "file_sha256": file_sha256,
    }


def write_chunks_jsonl(path: Path, chunks: list[dict]) -> None:
    """Helper to write chunks to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def test_1_loading_valid_chunks(tmp_path: Path) -> None:
    """1. Loading valid chunks."""
    c1 = make_valid_chunk("C1", text="Cloud computing solutions.")
    c2 = make_valid_chunk("C2", text="Retail grocery operations.")
    p = tmp_path / "chunks.jsonl"
    write_chunks_jsonl(p, [c1, c2])

    loaded = load_chunks(p)
    assert len(loaded) == 2
    assert loaded[0]["chunk_id"] == "C1"
    assert loaded[1]["chunk_id"] == "C2"
    assert loaded[0]["text"] == "Cloud computing solutions."


def test_2_malformed_json_handling(tmp_path: Path) -> None:
    """2. Malformed JSON handling."""
    p = tmp_path / "bad.jsonl"
    p.write_text("{\"chunk_id\": \"C1\", invalid_json\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "Malformed JSON" in str(exc.value)


def test_3_duplicate_chunk_id_rejection(tmp_path: Path) -> None:
    """3. Duplicate chunk ID rejection."""
    c1 = make_valid_chunk("DUP_01", text="First chunk.")
    c2 = make_valid_chunk("DUP_01", text="Second chunk with identical ID.")
    p = tmp_path / "dups.jsonl"
    write_chunks_jsonl(p, [c1, c2])

    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "Duplicate chunk ID" in str(exc.value)


def test_4_invalid_metadata_and_offset_rejection(tmp_path: Path) -> None:
    """4. Invalid metadata and offset rejection."""
    p = tmp_path / "invalid.jsonl"

    # Empty text
    c = make_valid_chunk("C_EMPTY", text="   ")
    write_chunks_jsonl(p, [c])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "empty or invalid 'text'" in str(exc.value)

    # Negative offset
    c = make_valid_chunk("C_NEG")
    c["char_start"] = -1
    write_chunks_jsonl(p, [c])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "negative or invalid 'char_start'" in str(exc.value)

    # Reversed offsets
    c = make_valid_chunk("C_REV")
    c["char_start"] = 100
    c["char_end"] = 50
    write_chunks_jsonl(p, [c])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "reversed or invalid 'char_end'" in str(exc.value)

    # Offset span does not match text length
    c = make_valid_chunk("C_SPAN")
    c["char_end"] = c["char_start"] + len(c["text"]) + 5
    write_chunks_jsonl(p, [c])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "does not equal text length" in str(exc.value)

    # Non-SEC source URL hostname
    c = make_valid_chunk("C_URL")
    c["source_url"] = "https://www.example.com/sec/filing.htm"
    write_chunks_jsonl(p, [c])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "must be an official SEC URL" in str(exc.value)

    # Invalid SHA-256
    c = make_valid_chunk("C_SHA")
    c["file_sha256"] = "short_hash"
    write_chunks_jsonl(p, [c])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "invalid 'file_sha256'" in str(exc.value)

    # Invalid section_confidence
    c = make_valid_chunk("C_CONF")
    c["section_confidence"] = "very_high"
    write_chunks_jsonl(p, [c])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "invalid 'section_confidence'" in str(exc.value)

    # Conflicting file_sha256 for a single accession
    c_hash1 = make_valid_chunk("C_H1", accession="0000950170-24-087843", file_sha256="1" * 64)
    c_hash2 = make_valid_chunk("C_H2", accession="0000950170-24-087843", file_sha256="2" * 64)
    write_chunks_jsonl(p, [c_hash1, c_hash2])
    with pytest.raises(ValueError) as exc:
        load_chunks(p)
    assert "Conflicting file_sha256 for accession" in str(exc.value)


def test_5_deterministic_tokenization() -> None:
    """5. Deterministic tokenization."""
    text = "Cloud-based AI infrastructure, revenue of $100M (15.2% margin) in FY2024!"
    t1 = tokenize(text)
    t2 = tokenize(text)
    assert t1 == t2
    # Verify case-insensitivity and hyphen handling
    assert "cloud-based" in t1
    assert "cloud" in t1
    assert "based" in t1
    assert "infrastructure" in t1
    assert "revenue" in t1
    assert "margin" in t1
    assert "fy2024" in t1
    # Stopword removed
    assert "in" not in t1
    assert "of" not in t1


def test_6_correct_bm25_ranking_for_known_toy_example() -> None:
    """6. Correct BM25 ranking for a known toy example."""
    # Doc 1: "apple banana"
    # Doc 2: "banana cherry"
    # Query: "apple"
    c1 = make_valid_chunk("DOC1", ticker="TOY", fiscal_year=2024, text="apple banana")
    c2 = make_valid_chunk("DOC2", ticker="TOY", fiscal_year=2024, text="banana cherry")
    idx = build_index([c1, c2])

    # Hand calculation:
    # N = 2, df(apple) = 1
    # IDF(apple) = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(1 + 1.5/1.5) = ln(2) ~= 0.693147
    # For DOC1: tf=1, doc_len=2, avg_doc_len=2.0 -> (doc_len/avg_doc_len) = 1.0
    # k1 = 1.5, b = 0.75 -> 1 - b + b*(1.0) = 1.0
    # den = 1 + 1.5 * 1.0 = 2.5
    # num = 1 * (1.5 + 1.0) = 2.5
    # BM25 = ln(2) * 2.5 / 2.5 = ln(2) ~= 0.6931
    expected_score = round(math.log(2.0), 4)

    res = search_evidence("apple", ["TOY"], [2024], idx, top_k=5)
    assert res["status"] == "success"
    assert len(res["results"]) == 1
    assert res["results"][0]["chunk_id"] == "DOC1"
    assert res["results"][0]["score"] == expected_score


def test_7_identical_queries_giving_identical_results() -> None:
    """7. Identical queries giving identical results."""
    c1 = make_valid_chunk("C1", ticker="AAA", fiscal_year=2024, text="operating income growth")
    c2 = make_valid_chunk("C2", ticker="AAA", fiscal_year=2024, text="credit loss allowance")
    idx = build_index([c1, c2])

    res1 = search_evidence("income growth", ["AAA"], [2024], idx)
    res2 = search_evidence("income growth", ["AAA"], [2024], idx)
    assert res1 == res2


def test_8_strict_ticker_filtering() -> None:
    """8. Strict ticker filtering."""
    c_msft = make_valid_chunk("C_MSFT", ticker="MSFT", fiscal_year=2024, text="cloud computing revenue")
    c_jpm = make_valid_chunk("C_JPM", ticker="JPM", fiscal_year=2024, text="cloud computing revenue")
    idx = build_index([c_msft, c_jpm])

    res = search_evidence("cloud computing", ["MSFT"], [2024], idx)
    assert res["status"] == "success"
    assert len(res["results"]) == 1
    assert res["results"][0]["ticker"] == "MSFT"
    assert res["results"][0]["chunk_id"] == "C_MSFT"


def test_9_strict_fiscal_year_filtering() -> None:
    """9. Strict fiscal-year filtering."""
    c_2023 = make_valid_chunk("C_23", ticker="MSFT", fiscal_year=2023, text="cloud revenue growth")
    c_2024 = make_valid_chunk("C_24", ticker="MSFT", fiscal_year=2024, text="cloud revenue growth")
    idx = build_index([c_2023, c_2024])

    res = search_evidence("cloud revenue", ["MSFT"], [2024], idx)
    assert res["status"] == "success"
    assert len(res["results"]) == 1
    assert res["results"][0]["fiscal_year"] == 2024
    assert res["results"][0]["chunk_id"] == "C_24"


def test_10_multiple_company_search() -> None:
    """10. Multiple-company search."""
    c_msft = make_valid_chunk("C_MSFT", ticker="MSFT", fiscal_year=2024, text="enterprise software")
    c_jpm = make_valid_chunk("C_JPM", ticker="JPM", fiscal_year=2024, text="enterprise banking software")
    c_wmt = make_valid_chunk("C_WMT", ticker="WMT", fiscal_year=2024, text="enterprise retail software")
    idx = build_index([c_msft, c_jpm, c_wmt])

    res = search_evidence("enterprise software", ["MSFT", "JPM"], [2024], idx, top_k=5)
    assert res["status"] == "success"
    returned_tickers = {r["ticker"] for r in res["results"]}
    assert returned_tickers == {"MSFT", "JPM"}
    assert "WMT" not in returned_tickers


def test_11_no_data_leakage_between_companies_or_years() -> None:
    """11. No data leakage between companies or years."""
    # Term statistics of unselected company must not affect score of selected company
    c_target1 = make_valid_chunk("T1", ticker="TARGET", fiscal_year=2024, text="risk assessment")
    c_target2 = make_valid_chunk("T2", ticker="TARGET", fiscal_year=2024, text="capital structure")
    idx_isolated = build_index([c_target1, c_target2])
    score_isolated = search_evidence("risk", ["TARGET"], [2024], idx_isolated)["results"][0]["score"]

    # Now add unrelated company chunks with many mentions of 'risk'
    unrelated = [
        make_valid_chunk(f"U{i}", ticker="OTHER", fiscal_year=2024, text="risk risk risk")
        for i in range(10)
    ]
    idx_combined = build_index([c_target1, c_target2] + unrelated)
    score_combined = search_evidence("risk", ["TARGET"], [2024], idx_combined)["results"][0]["score"]

    # Scores must be identical because statistics are calculated strictly on the eligible subset!
    assert score_isolated == score_combined


def test_12_missing_company_year_returning_no_results() -> None:
    """12. Missing company-year returning no_results."""
    c = make_valid_chunk("C1", ticker="MSFT", fiscal_year=2024, text="some filing text")
    idx = build_index([c])

    res_missing_ticker = search_evidence("filing text", ["UNKNOWN"], [2024], idx)
    assert res_missing_ticker["status"] == "no_results"
    assert len(res_missing_ticker["results"]) == 0
    assert "No indexed filings found" in res_missing_ticker["message"]

    res_missing_year = search_evidence("filing text", ["MSFT"], [2020], idx)
    assert res_missing_year["status"] == "no_results"
    assert len(res_missing_year["results"]) == 0


def test_13_stopword_only_and_empty_query_behavior() -> None:
    """13. Stopword-only and empty query behavior."""
    c = make_valid_chunk("C1", ticker="MSFT", fiscal_year=2024, text="some filing text")
    idx = build_index([c])

    # Empty string query
    res_empty = search_evidence("", ["MSFT"], [2024], idx)
    assert res_empty["status"] == "no_results"
    assert res_empty["message"] == "Query is empty."

    # Stopword-only query
    res_stops = search_evidence("and the of in", ["MSFT"], [2024], idx)
    assert res_stops["status"] == "no_results"
    assert "no searchable terms" in res_stops["message"]


def test_14_stable_tie_breaking() -> None:
    """14. Stable tie-breaking."""
    # Two identical chunks with same text will receive identical BM25 scores
    c_b = make_valid_chunk("CHUNK_B", ticker="TEST", fiscal_year=2024, text="identical score text")
    c_a = make_valid_chunk("CHUNK_A", ticker="TEST", fiscal_year=2024, text="identical score text")
    idx = build_index([c_b, c_a])

    res = search_evidence("identical score", ["TEST"], [2024], idx, top_k=2)
    assert res["status"] == "success"
    assert len(res["results"]) == 2
    assert res["results"][0]["score"] == res["results"][1]["score"]
    # Deterministic alphabetical ordering by chunk_id on tie
    assert res["results"][0]["chunk_id"] == "CHUNK_A"
    assert res["results"][1]["chunk_id"] == "CHUNK_B"


def test_15_top_k_validation() -> None:
    """15. top_k validation."""
    c = make_valid_chunk("C1", ticker="TEST", fiscal_year=2024, text="test text")
    idx = build_index([c])

    with pytest.raises(ValueError):
        search_evidence("test", ["TEST"], [2024], idx, top_k=0)

    with pytest.raises(ValueError):
        search_evidence("test", ["TEST"], [2024], idx, top_k=21)

    with pytest.raises(ValueError):
        search_evidence("test", ["TEST"], [2024], idx, top_k="five")  # type: ignore


def test_16_exact_source_url_and_accession_preservation() -> None:
    """16. Exact source URL and accession preservation."""
    url = "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm"
    accn = "0000950170-24-087843"
    c = make_valid_chunk("C_MSFT", ticker="MSFT", fiscal_year=2024, text="cloud revenue", accession=accn, source_url=url)
    idx = build_index([c])

    res = search_evidence("cloud", ["MSFT"], [2024], idx)
    top = res["results"][0]
    assert top["source_url"] == url
    assert top["accession"] == accn
    assert top["char_start"] == c["char_start"]
    assert top["char_end"] == c["char_end"]
    assert top["file_sha256"] == c["file_sha256"]


def test_17_section_unknown_remains_searchable() -> None:
    """17. Section unknown remains searchable."""
    c = make_valid_chunk("C_UNK", ticker="TEST", fiscal_year=2024, text="preamble executive remarks", section="unknown", section_confidence="unknown")
    idx = build_index([c])

    res = search_evidence("preamble remarks", ["TEST"], [2024], idx)
    assert res["status"] == "success"
    top = res["results"][0]
    assert top["chunk_id"] == "C_UNK"
    assert top["section"] == "unknown"
    assert top["section_confidence"] == "unknown"


def test_18_correct_text_returned_without_rewriting() -> None:
    """18. Correct text returned without rewriting."""
    exact_text = "Exact text with punctuation: $1,234.56, 15.2% margin & special formatting.\nNew line preserved."
    c = make_valid_chunk("C_EXACT", ticker="TEST", fiscal_year=2024, text=exact_text)
    idx = build_index([c])

    res = search_evidence("margin formatting", ["TEST"], [2024], idx)
    assert res["status"] == "success"
    assert res["results"][0]["text"] == exact_text


def test_19_filtering_before_scoring_trap_test() -> None:
    """19. Filtering before scoring, not merely filtering final ranked results.

    Trap: create many highly relevant chunks from another company and one relevant
    chunk from the requested company. The requested company's chunk must still be returned.
    """
    other_chunks = [
        make_valid_chunk(
            f"OTHER_{i}",
            ticker="OTHER",
            fiscal_year=2024,
            text="cybersecurity risk cloud services security incident threat breach mitigation " * 5,
        )
        for i in range(50)
    ]
    target_chunk = make_valid_chunk(
        "TARGET_01",
        ticker="TARGET",
        fiscal_year=2024,
        text="Target company cybersecurity risk and cloud services evaluation.",
    )
    idx = build_index(other_chunks + [target_chunk])

    # Search with filter for TARGET only: must return TARGET_01 despite OTHER having 50 hyper-relevant chunks
    res = search_evidence("cybersecurity risk cloud services", ["TARGET"], [2024], idx, top_k=5)
    assert res["status"] == "success"
    assert len(res["results"]) == 1
    assert res["results"][0]["chunk_id"] == "TARGET_01"
    assert res["results"][0]["ticker"] == "TARGET"


def test_20_cli_argument_parsing_and_execution(tmp_path: Path) -> None:
    """20. CLI argument parsing and execution."""
    args = parse_args(["--query", "cloud risks", "--tickers", "MSFT", "--years", "2024", "--top-k", "3"])
    assert args.query == "cloud risks"
    assert args.tickers == ["MSFT"]
    assert args.years == [2024]
    assert args.top_k == 3

    # Test main execution with a mock chunks file
    chunks_file = tmp_path / "test_chunks.jsonl"
    c = make_valid_chunk("CLI_01", ticker="MSFT", fiscal_year=2024, text="cloud risks overview")
    write_chunks_jsonl(chunks_file, [c])

    with patch("sys.argv", ["retrieval.py", "--query", "cloud risks", "--tickers", "MSFT", "--years", "2024", "--chunks-path", str(chunks_file)]):
        main()


def test_21_full_precision_ranking_before_display_rounding() -> None:
    """Full-precision BM25 score is used for ranking; display rounding to 4 decimals does not distort ranking."""
    # Chunk A has slightly fewer total words than Chunk B, so its doc_len is smaller -> slightly higher BM25 score.
    # We craft two chunks with same term frequency of query word 'alpha', but Chunk B has an extra padding word.
    c_a = make_valid_chunk("CHUNK_A", ticker="MSFT", fiscal_year=2024, text="alpha " + "beta " * 100)
    c_b = make_valid_chunk("CHUNK_B", ticker="MSFT", fiscal_year=2024, text="alpha " + "beta " * 101)

    idx = build_index([c_b, c_a])
    res = search_evidence("alpha", ["MSFT"], [2024], idx, top_k=2)

    assert res["status"] == "success"
    assert len(res["results"]) == 2
    # CHUNK_A has shorter length, so its BM25 score is strictly higher at unrounded float precision
    assert res["results"][0]["chunk_id"] == "CHUNK_A"
    assert res["results"][1]["chunk_id"] == "CHUNK_B"


def test_22_accession_isolation_filtering() -> None:
    """Search evidence respects accession filter applied before scoring."""
    c1 = make_valid_chunk("C_ACC1", ticker="MSFT", fiscal_year=2024, accession="0000950170-24-000001", text="operating revenue streams")
    c2 = make_valid_chunk("C_ACC2", ticker="MSFT", fiscal_year=2024, accession="0000950170-24-000002", text="operating revenue streams")

    idx = build_index([c1, c2])
    res = search_evidence("revenue", ["MSFT"], [2024], idx, accessions=["0000950170-24-000001"])
    assert res["status"] == "success"
    assert len(res["results"]) == 1
    assert res["results"][0]["chunk_id"] == "C_ACC1"
    assert res["results"][0]["accession"] == "0000950170-24-000001"


def test_23_form_isolation_filtering() -> None:
    """Search evidence respects form filter applied before scoring."""
    c1 = make_valid_chunk("C_10K", ticker="MSFT", fiscal_year=2024, form="10-K", text="annual risk factors")
    c2 = make_valid_chunk("C_10Q", ticker="MSFT", fiscal_year=2024, form="10-Q", text="quarterly risk factors")

    idx = build_index([c1, c2])
    res = search_evidence("risk", ["MSFT"], [2024], idx, forms=["10-K"])
    assert res["status"] == "success"
    assert len(res["results"]) == 1
    assert res["results"][0]["chunk_id"] == "C_10K"


def test_24_unrelated_companies_do_not_distort_scores() -> None:
    """Adding unrelated companies to the index does not change the BM25 score of chunks within an isolated query."""
    c_target = make_valid_chunk("TARGET_1", ticker="MSFT", fiscal_year=2024, text="unique keyword query")
    c_other = make_valid_chunk("OTHER_1", ticker="AAPL", fiscal_year=2024, text="unique keyword query extra words")

    idx_isolated = build_index([c_target])
    res_isolated = search_evidence("unique keyword", ["MSFT"], [2024], idx_isolated)

    idx_combined = build_index([c_target, c_other])
    res_combined = search_evidence("unique keyword", ["MSFT"], [2024], idx_combined)

    assert res_isolated["status"] == "success"
    assert res_combined["status"] == "success"
    # The score of TARGET_1 must be mathematically identical because eligible subset statistics are used
    assert res_isolated["results"][0]["score"] == res_combined["results"][0]["score"]

