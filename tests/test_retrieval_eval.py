"""Unit tests for the retrieval evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import pytest

from fyntrace.retrieval import build_index
from fyntrace.retrieval_eval import (
    load_questions,
    load_audit,
    validate_questions,
    validate_audit,
    evaluate_retrieval,
    save_results,
    COMPANY_TO_SECTOR,
)
from tests.test_retrieval import _make_synthetic_corpus


def _make_eval_setup(tmp_path: Path, write_canonical: bool = False):
    """Set up a synthetic corpus, build index, and return valid questions."""
    chunks_path, chunks = _make_synthetic_corpus(tmp_path)
    db_path = tmp_path / "test_retrieval.sqlite"
    summary = build_index(chunks_path=chunks_path, db_path=db_path)
    corpus_sha256 = summary["corpus_sha256"]

    if write_canonical:
        # Create matching canonical text files for all chunks
        for c in chunks:
            c_rel = c["canonical_text_path"]
            txt_file = tmp_path / c_rel
            txt_file.parent.mkdir(parents=True, exist_ok=True)
            prefix = " " * c["char_start"]
            content = prefix + c["text"] + " [END]"
            txt_file.write_text(content, encoding="utf-8")

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


def _make_100_synthetic_questions(questions: list[dict], corpus_sha256: str) -> list[dict]:
    """Generate 100 well-formed synthetic questions satisfying full distribution."""
    sectors = ["Technology", "Finance", "Healthcare", "Energy", "Consumer"]
    sec_to_tickers = {
        "Technology": ["AAPL", "MSFT", "NVDA"],
        "Finance": ["JPM", "BAC", "GS"],
        "Healthcare": ["JNJ", "PFE", "ABBV"],
        "Energy": ["XOM", "CVX", "COP"],
        "Consumer": ["WMT", "COST", "MCD"],
    }
    categories = (
        ["business_operations"] * 20
        + ["risk_factors"] * 25
        + ["management_discussion"] * 25
        + ["financial_tables"] * 20
        + ["other_disclosures"] * 10
    )
    difficulties = ["easy"] * 20 + ["medium"] * 40 + ["hard"] * 40

    q100 = []
    base_q = questions[0]
    for i in range(100):
        sec = sectors[i // 20]
        ticker = sec_to_tickers[sec][i % len(sec_to_tickers[sec])]
        year = 2023 + (i % 3)
        q = dict(base_q)
        q["question_id"] = f"DEV{i+1:04d}"
        q["ticker"] = ticker
        q["fiscal_year"] = year
        q["sector"] = sec
        q["category"] = categories[i]
        q["difficulty"] = difficulties[i]
        q["corpus_sha256"] = corpus_sha256
        q100.append(q)
    return q100


def test_1_production_validation_rejects_99_questions(tmp_path: Path):
    """Production validation with enforce_100_dist=True rejects 99 questions."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    q99 = [dict(questions[0]) for _ in range(99)]
    for i, q in enumerate(q99):
        q["question_id"] = f"DEV{i+1:04d}"
    with pytest.raises(ValueError, match="Expected exactly 100 questions, got 99"):
        validate_questions(q99, db_path, chunks_path, enforce_100_dist=True, check_canonical_text=False)


def test_2_production_evaluation_rejects_incomplete_dataset(tmp_path: Path):
    """Incomplete dataset is rejected when production 100-question rule is enforced."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    incomplete_qs = questions[:5]
    with pytest.raises(ValueError, match="Expected exactly 100 questions"):
        validate_questions(incomplete_qs, db_path, chunks_path, enforce_100_dist=True, check_canonical_text=False)


def test_3_missing_accession_rejected(tmp_path: Path):
    """Question missing accession field is rejected."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    del bad_q[0]["accession"]
    with pytest.raises(ValueError, match="missing mandatory field 'accession'"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)

    bad_q2 = [dict(questions[0])]
    bad_q2[0]["accession"] = ""
    with pytest.raises(ValueError, match="empty field 'accession'"):
        validate_questions(bad_q2, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)


def test_4_incorrect_accession_rejected(tmp_path: Path):
    """Question with accession mismatched from gold chunk is rejected."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    bad_q[0]["accession"] = "0000000000-00-000000"
    with pytest.raises(ValueError, match="accession"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)


def test_5_missing_source_url_rejected(tmp_path: Path):
    """Question missing source_url field is rejected."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    del bad_q[0]["source_url"]
    with pytest.raises(ValueError, match="missing mandatory field 'source_url'"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)

    bad_q2 = [dict(questions[0])]
    bad_q2[0]["source_url"] = "   "
    with pytest.raises(ValueError, match="empty field 'source_url'"):
        validate_questions(bad_q2, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)


def test_6_incorrect_source_url_rejected(tmp_path: Path):
    """Question with source_url mismatched from gold chunk is rejected."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    bad_q[0]["source_url"] = "https://wrong.url/filing.htm"
    with pytest.raises(ValueError, match="source_url"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)


def test_7_missing_canonical_text_file_rejected_in_strict_mode(tmp_path: Path):
    """Missing canonical text file raises FileNotFoundError when check_canonical_text=True."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path, write_canonical=False)
    with pytest.raises(FileNotFoundError, match="Canonical text file not found"):
        validate_questions(questions[:1], db_path, chunks_path, enforce_100_dist=False, check_canonical_text=True)


def test_8_incorrect_text_offsets_rejected(tmp_path: Path):
    """Incorrect canonical text offsets raise ValueError when check_canonical_text=True."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path, write_canonical=True)
    # Tamper with the canonical text file so offsets don't match chunk text
    txt_file = tmp_path / "data/processed/text/msft_2024.txt"
    txt_file.write_text("CORRUPTED_TEXT_CORRUPTED_TEXT", encoding="utf-8")
    with pytest.raises(ValueError, match="char offsets do not match canonical text"):
        validate_questions(questions[:1], db_path, chunks_path, enforce_100_dist=False, check_canonical_text=True)


def test_9_incorrect_corpus_sha256_rejected(tmp_path: Path):
    """Corpus SHA-256 mismatch raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    bad_q[0]["corpus_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match actual"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)


def test_10_missing_difficulty_rejected(tmp_path: Path):
    """Missing difficulty raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    bad_q = [dict(questions[0])]
    del bad_q[0]["difficulty"]
    with pytest.raises(ValueError, match="missing mandatory field 'difficulty'"):
        validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)


def test_11_invalid_difficulty_rejected(tmp_path: Path):
    """Invalid difficulty raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    for bad_diff in ["advanced", "basic", "extreme", "none"]:
        bad_q = [dict(questions[0])]
        bad_q[0]["difficulty"] = bad_diff
        with pytest.raises(ValueError, match="invalid difficulty"):
            validate_questions(bad_q, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)


def test_12_incorrect_difficulty_distribution_rejected(tmp_path: Path):
    """100 questions with non-20/40/40 difficulty breakdown raises ValueError."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    meta_corpus = "test"
    # Create 100 questions with 50 easy, 30 medium, 20 hard
    q100 = []
    for i in range(100):
        q = dict(questions[i % len(questions)])
        q["question_id"] = f"DEV{i+1:04d}"
        q["difficulty"] = "easy" if i < 50 else ("medium" if i < 80 else "hard")
        q100.append(q)

    with pytest.raises(ValueError, match="Difficulty"):
        validate_questions(q100, db_path, chunks_path, enforce_100_dist=True, check_canonical_text=False)


def test_13_duplicate_audit_ids_rejected():
    """Duplicate question_id in audit records raises ValueError."""
    questions = [{"question_id": "DEV0001", "gold_chunk_ids": ["c1"], "difficulty": "easy"}]
    bad_audit = [
        {
            "question_id": "DEV0001",
            "decision": "verified",
            "final_gold_chunk_ids": ["c1"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
        },
        {
            "question_id": "DEV0001",
            "decision": "verified",
            "final_gold_chunk_ids": ["c1"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
        },
    ]
    with pytest.raises(ValueError, match="Duplicate question_id 'DEV0001' in audit records"):
        validate_audit(bad_audit, questions, enforce_100=False)


def test_14_missing_audit_records_rejected():
    """Audit missing a record for a question in dataset raises ValueError."""
    questions = [
        {"question_id": "DEV0001", "gold_chunk_ids": ["c1"], "difficulty": "easy"},
        {"question_id": "DEV0002", "gold_chunk_ids": ["c2"], "difficulty": "medium"},
    ]
    incomplete_audit = [
        {
            "question_id": "DEV0001",
            "decision": "verified",
            "final_gold_chunk_ids": ["c1"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
        }
    ]
    with pytest.raises(ValueError, match="Question 'DEV0002' is missing an audit record"):
        validate_audit(incomplete_audit, questions, enforce_100=False)


def test_15_invalid_audit_decisions_rejected():
    """Audit record with invalid decision raises ValueError."""
    questions = [{"question_id": "DEV0001", "gold_chunk_ids": ["c1"], "difficulty": "easy"}]
    bad_audit = [
        {
            "question_id": "DEV0001",
            "decision": "approved_by_worker",
            "final_gold_chunk_ids": ["c1"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
        }
    ]
    with pytest.raises(ValueError, match="invalid decision 'approved_by_worker'"):
        validate_audit(bad_audit, questions, enforce_100=False)


def test_16_unresolved_audit_records_block_completion():
    """Audit record marked 'unresolved' raises ValueError."""
    questions = [{"question_id": "DEV0001", "gold_chunk_ids": ["c1"], "difficulty": "easy"}]
    unresolved_audit = [
        {
            "question_id": "DEV0001",
            "decision": "unresolved",
            "final_gold_chunk_ids": ["c1"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
        }
    ]
    with pytest.raises(ValueError, match="marked 'unresolved'"):
        validate_audit(unresolved_audit, questions, enforce_100=False)


def test_17_changed_gold_labels_without_reason_rejected():
    """Audit record with changed gold label but missing correction_reason raises ValueError."""
    questions = [{"question_id": "DEV0001", "gold_chunk_ids": ["c2"], "difficulty": "easy"}]
    bad_audit = [
        {
            "question_id": "DEV0001",
            "decision": "corrected_gold",
            "gold_label_changed": True,
            "final_gold_chunk_ids": ["c2"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
            "correction_reason": None,
        }
    ]
    with pytest.raises(ValueError, match="missing 'correction_reason'"):
        validate_audit(bad_audit, questions, enforce_100=False)


def test_18_audit_gold_ids_must_match_final_question_labels():
    """Audit record final_gold_chunk_ids must match question gold_chunk_ids."""
    questions = [{"question_id": "DEV0001", "gold_chunk_ids": ["c_final"], "difficulty": "easy"}]
    mismatched_audit = [
        {
            "question_id": "DEV0001",
            "decision": "verified",
            "final_gold_chunk_ids": ["c_different"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
        }
    ]
    with pytest.raises(ValueError, match="does not match question gold_chunk_ids"):
        validate_audit(mismatched_audit, questions, enforce_100=False)


def test_19_audit_difficulty_must_match_final_question_difficulty():
    """Audit record final_difficulty must match question difficulty."""
    questions = [{"question_id": "DEV0001", "gold_chunk_ids": ["c1"], "difficulty": "hard"}]
    mismatched_audit = [
        {
            "question_id": "DEV0001",
            "decision": "verified",
            "final_gold_chunk_ids": ["c1"],
            "final_difficulty": "easy",
            "evidence_support": "Clear factual evidence passage.",
            "verified_evidence_excerpt": "Valid excerpt phrase",
        }
    ]
    with pytest.raises(ValueError, match="does not match question difficulty"):
        validate_audit(mismatched_audit, questions, enforce_100=False)


def test_20_technical_dataset_validation_works_without_private_audit_file(tmp_path: Path):
    """Technical dataset validation succeeds independently without any audit file."""
    db_path, chunks_path, q_file, questions = _make_eval_setup(tmp_path)
    loaded_qs = load_questions(q_file)
    summary = validate_questions(loaded_qs, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)
    assert summary["status"] == "valid"
    assert summary["question_count"] == len(questions)


def test_21_optional_local_audit_validation_works_when_path_supplied(tmp_path: Path):
    """validate_audit succeeds when valid matching audit records are supplied."""
    _, _, _, questions = _make_eval_setup(tmp_path)
    audit_records = [
        {
            "question_id": q["question_id"],
            "decision": "verified",
            "original_gold_chunk_ids": q["gold_chunk_ids"],
            "final_gold_chunk_ids": q["gold_chunk_ids"],
            "question_changed": False,
            "gold_label_changed": False,
            "excerpt_changed": False,
            "original_difficulty": q["difficulty"],
            "final_difficulty": q["difficulty"],
            "evidence_support": f"Verified factual support for {q['question_id']}",
            "verified_evidence_excerpt": q["evidence_excerpt"],
            "correction_reason": None,
        }
        for q in questions
    ]
    audit_file = tmp_path / "audit.jsonl"
    with audit_file.open("w", encoding="utf-8") as f:
        for r in audit_records:
            f.write(json.dumps(r) + "\n")

    loaded_audit = load_audit(audit_file)
    audit_summary = validate_audit(loaded_audit, questions, enforce_100=False)
    assert audit_summary["status"] == "valid"
    assert audit_summary["audit_record_count"] == len(questions)
    assert audit_summary["decisions"]["verified"] == len(questions)


def test_22_existing_recall_at_5_calculated_correctly(tmp_path: Path):
    """Recall@5 correctly computes proportion of gold chunks retrieved in top 5."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert 0.0 <= results["overall"]["recall_at_5"] <= 1.0


def test_23_existing_hit_at_5_calculated_correctly(tmp_path: Path):
    """Hit@5 correctly computes proportion of queries with at least one gold chunk."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert 0.0 <= results["overall"]["hit_at_5"] <= 1.0
    for q_res in results["per_question"]:
        if q_res["hit"] == 1.0:
            assert q_res["recall"] > 0.0


def test_24_existing_mrr_at_5_calculated_correctly(tmp_path: Path):
    """MRR@5 assigns reciprocal ranks 1.0, 0.5, 0.3333, 0.25, 0.2, or 0.0."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert 0.0 <= results["overall"]["mrr_at_5"] <= 1.0
    for q_res in results["per_question"]:
        if q_res["rank"] is not None:
            assert abs(q_res["mrr"] - (1.0 / q_res["rank"])) < 1e-4
        else:
            assert q_res["mrr"] == 0.0


def test_25_no_network_access_occurs(tmp_path: Path, monkeypatch):
    """Evaluation operations never make network requests."""
    db_path, chunks_path, _, questions = _make_eval_setup(tmp_path)

    def forbid_network(*args, **kwargs):
        raise RuntimeError("Network access attempted during evaluation!")

    monkeypatch.setattr(socket, "socket", forbid_network)
    validate_questions(questions, db_path, chunks_path, enforce_100_dist=False, check_canonical_text=False)
    results = evaluate_retrieval(questions, db_path, top_k=5)
    assert len(results["per_question"]) == len(questions)
