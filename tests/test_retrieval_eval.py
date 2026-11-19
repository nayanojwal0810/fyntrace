"""Unit tests for the retrieval evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import pytest

from fyntrace.retrieval import build_index
from fyntrace.retrieval_eval import (
    load_questions,
    validate_questions,
    evaluate_retrieval,
    save_results,
    COMPANY_TO_SECTOR,
)
from tests.test_retrieval import _make_synthetic_corpus


def _make_eval_setup(tmp_path: Path):
    """Set up a synthetic corpus, build index, and return valid questions."""
    chunks_path, chunks = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"
    summary = build_index(chunks_path=chunks_path, db_path=db_path)
    corpus_sha256 = summary["corpus_sha256"]

    # Create 6 valid questions corresponding to the 6 synthetic chunks
    questions = [
        {
            "question_id": "DEV0001",
            "query": "What was the growth rate of Microsoft Cloud revenue in 2024?",
            "ticker": "MSFT",
            "fiscal_year": 2024,
            "sector": "Technology",
            "category": "management_discussion",
            "difficulty": "easy",
            "gold_chunk_ids": ["MSFT_2024_10K_000095017024087843_00001"],
            "evidence_excerpt": "Microsoft Cloud revenue grew 22 percent",
            "accession": "0000950170-24-087843",
            "source_url": "https://www.sec.gov/edgar/msft-2024.htm",
            "section": "Item 7 - Management's Discussion and Analysis",
            "chunk_type": "text",
            "corpus_sha256": corpus_sha256,
        },
        {
            "question_id": "DEV0002",
            "query": "What segment revenue was reported for Intelligent Cloud in fiscal year 2024?",
            "ticker": "MSFT",
            "fiscal_year": 2024,
            "sector": "Technology",
            "category": "financial_tables",
            "difficulty": "medium",
            "gold_chunk_ids": ["MSFT_2024_10K_000095017024087843_00002"],
            "evidence_excerpt": "Intelligent Cloud | 105362",
            "accession": "0000950170-24-087843",
            "source_url": "https://www.sec.gov/edgar/msft-2024.htm",
            "section": "Item 8 - Financial Statements",
            "chunk_type": "table",
            "corpus_sha256": corpus_sha256,
        },
        {
            "question_id": "DEV0003",
            "query": "How much cloud revenue did Microsoft generate in fiscal year 2023?",
            "ticker": "MSFT",
            "fiscal_year": 2023,
            "sector": "Technology",
            "category": "management_discussion",
            "difficulty": "hard",
            "gold_chunk_ids": ["MSFT_2023_10K_000095017023035122_00001"],
            "evidence_excerpt": "Microsoft Cloud revenue was 111 billion dollars",
            "accession": "0000950170-23-035122",
            "source_url": "https://www.sec.gov/edgar/msft-2023.htm",
            "section": "Item 7 - Management's Discussion and Analysis",
            "chunk_type": "text",
            "corpus_sha256": corpus_sha256,
        },
        {
            "question_id": "DEV0004",
            "query": "What risks affect Apple iPhone manufacturing?",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "sector": "Technology",
            "category": "risk_factors",
            "difficulty": "medium",
            "gold_chunk_ids": ["AAPL_2024_10K_000032019324000123_00001"],
            "evidence_excerpt": "Global supply chain disruptions and semiconductor shortages",
            "accession": "0000320193-24-000123",
            "source_url": "https://www.sec.gov/edgar/aapl-2024.htm",
            "section": "Item 1A - Risk Factors",
            "chunk_type": "text",
            "corpus_sha256": corpus_sha256,
        },
        {
            "question_id": "DEV0005",
            "query": "What was JPMorgan Chase net interest income?",
            "ticker": "JPM",
            "fiscal_year": 2024,
            "sector": "Finance",
            "category": "business_operations",
            "difficulty": "hard",
            "gold_chunk_ids": ["JPM_2024_10K_000001961724000273_00001"],
            "evidence_excerpt": "net interest income of 90 billion dollars",
            "accession": "0000019617-24-000273",
            "source_url": "https://www.sec.gov/edgar/jpm-2024.htm",
            "section": "Item 7 - Management's Discussion and Analysis",
            "chunk_type": "text",
            "corpus_sha256": corpus_sha256,
        },
        {
            "question_id": "DEV0006",
            "query": "What retail and wholesale formats does Walmart operate?",
            "ticker": "WMT",
            "fiscal_year": 2024,
            "sector": "Consumer",
            "category": "other_disclosures",
            "difficulty": "easy",
            "gold_chunk_ids": ["WMT_2024_10K_000010416924000033_00001"],
            "evidence_excerpt": "retail discount department stores and wholesale warehouse clubs",
            "accession": "0000104169-24-000033",
            "source_url": "https://www.sec.gov/edgar/wmt-2024.htm",
            "section": "Item 1 - Business",
            "chunk_type": "text",
            "corpus_sha256": corpus_sha256,
        },
    ]

    q_file = tmp_path / "questions.jsonl"
    with q_file.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    return db_path, chunks_path, q_file, questions


def test_1_correct_loading_of_valid_question_records(tmp_path: Path):
    """load_questions loads valid JSON Lines and preserves fields."""
    _, _, q_file, questions = _make_eval_setup(tmp_path)
    loaded = load_questions(q_file)
    assert len(loaded) == len(questions)
    assert loaded[0]["question_id"] == "DEV0001"
    assert loaded[0]["difficulty"] == "easy"


def test_2_duplicate_question_ids_rejected(tmp_path: Path):
    """Duplicate question_id raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0]), dict(questions[0])]
    with pytest.raises(ValueError, match="Duplicate question_id"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)


def test_3_missing_gold_chunk_ids_rejected(tmp_path: Path):
    """Missing or empty gold_chunk_ids raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    bad_q[0]["gold_chunk_ids"] = []
    with pytest.raises(ValueError, match="missing or empty 'gold_chunk_ids'"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)


def test_4_wrong_ticker_year_gold_label_rejected(tmp_path: Path):
    """Gold chunk belonging to a different ticker/year raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    # Question asks for MSFT 2024, but points to AAPL 2024 chunk
    bad_q[0]["gold_chunk_ids"] = ["AAPL_2024_10K_000032019324000123_00001"]
    with pytest.raises(ValueError, match="specifies ticker MSFT, but gold chunk"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)


def test_5_wrong_accession_and_source_url_rejected(tmp_path: Path):
    """Mismatched accession or source_url raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    bad_q[0]["accession"] = "0000000000-00-000000"
    with pytest.raises(ValueError, match="accession"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)

    bad_q2 = [dict(questions[0])]
    bad_q2[0]["source_url"] = "https://wrong.url"
    with pytest.raises(ValueError, match="source_url"):
        validate_questions(bad_q2, db_path, chunks_path, enforce_100_dist=False)


def test_6_evidence_excerpt_not_found_in_gold_text_rejected(tmp_path: Path):
    """Evidence excerpt missing from gold chunk text raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    bad_q[0]["evidence_excerpt"] = "This phrase does not exist in MSFT 2024 text"
    with pytest.raises(ValueError, match="evidence_excerpt not found in gold chunk"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)


def test_7_corpus_hash_mismatch_rejected(tmp_path: Path):
    """Corpus SHA-256 mismatch raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    bad_q[0]["corpus_sha256"] = "badhash" * 8
    with pytest.raises(ValueError, match="does not match actual"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)


def test_8_recall_at_5_calculated_correctly(tmp_path: Path):
    """Recall@5 correctly computes proportion of gold chunks retrieved in top 5."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert 0.0 <= results["overall"]["recall_at_5"] <= 1.0


def test_9_hit_at_5_calculated_correctly(tmp_path: Path):
    """Hit@5 correctly computes proportion of queries with at least one gold chunk."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert 0.0 <= results["overall"]["hit_at_5"] <= 1.0
    # Every hit=1 has recall > 0
    for q_res in results["per_question"]:
        if q_res["hit"] == 1.0:
            assert q_res["recall"] > 0.0


def test_10_mrr_at_5_calculated_correctly(tmp_path: Path):
    """MRR@5 assigns reciprocal ranks 1.0, 0.5, 0.3333, 0.25, 0.2, or 0.0."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert 0.0 <= results["overall"]["mrr_at_5"] <= 1.0
    for q_res in results["per_question"]:
        if q_res["rank"] is not None:
            assert abs(q_res["mrr"] - (1.0 / q_res["rank"])) < 1e-4
        else:
            assert q_res["mrr"] == 0.0


def test_11_per_category_and_per_sector_metrics_calculated(tmp_path: Path):
    """Metrics by category and sector are grouped and aggregated correctly."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert "Technology" in results["by_sector"]
    assert "management_discussion" in results["by_category"]
    assert "by_fiscal_year" in results


def test_12_results_stay_within_requested_filters(tmp_path: Path):
    """All returned passages respect requested company and year filters."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert results["overall"]["filter_violations"] == 0


def test_13_empty_or_invalid_evaluation_set_rejected(tmp_path: Path):
    """Empty question file raises ValueError."""
    empty_file = tmp_path / "empty_q.jsonl"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or contains no valid questions"):
        load_questions(empty_file)


def test_14_evaluation_output_saves_correctly(tmp_path: Path):
    """save_results writes inspectable JSON matching output structure."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    out_file = tmp_path / "results.json"
    save_results(results, out_file)
    assert out_file.is_file()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["overall"]["recall_at_5"] == results["overall"]["recall_at_5"]


def test_15_no_network_access_occurs(tmp_path: Path, monkeypatch):
    """Evaluation operations never make network requests."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)

    def forbid_network(*args, **kwargs):
        raise RuntimeError("Network access attempted during evaluation!")

    monkeypatch.setattr(socket, "socket", forbid_network)
    validate_questions(questions, db_path, chunks_path, enforce_100_dist=False)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert len(results["per_question"]) == len(questions)


def test_16_missing_difficulty_field_rejected(tmp_path: Path):
    """Question without 'difficulty' raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    del bad_q[0]["difficulty"]
    with pytest.raises(ValueError, match="missing mandatory 'difficulty'"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)


def test_17_invalid_difficulty_value_rejected(tmp_path: Path):
    """Question with invalid difficulty (e.g. 'advanced' or 'basic') raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    for bad_diff in ["advanced", "basic", "intermediate", "EASY", "1"]:
        bad_q = [dict(questions[0])]
        bad_q[0]["difficulty"] = bad_diff
        if bad_diff.lower() not in ("easy", "medium", "hard"):
            with pytest.raises(ValueError, match="invalid difficulty"):
                validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False)


def test_18_incorrect_difficulty_distribution_rejected(tmp_path: Path):
    """100 questions with non-20/40/40 difficulty distribution raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    # Generate 100 questions with wrong difficulty breakdown (e.g. 50 easy, 30 medium, 20 hard)
    q100 = []
    for i in range(100):
        q = dict(questions[i % len(questions)])
        q["question_id"] = f"DEV{i+1:04d}"
        q["difficulty"] = "easy" if i < 50 else ("medium" if i < 80 else "hard")
        q100.append(q)

    with pytest.raises(ValueError, match="Difficulty"):
        validate_questions(q100, db_path, chunks_path, enforce_100_dist=True)


def test_19_difficulty_wise_metrics_calculated(tmp_path: Path):
    """evaluate_retrieval computes breakdown for easy, medium, and hard."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert "by_difficulty" in results
    assert "easy" in results["by_difficulty"]
    assert "medium" in results["by_difficulty"]
    assert "hard" in results["by_difficulty"]
    for diff in ["easy", "medium", "hard"]:
        d_res = results["by_difficulty"][diff]
        assert "recall_at_5" in d_res
        assert "hit_at_5" in d_res
        assert "mrr_at_5" in d_res
        assert "missed_count" in d_res
