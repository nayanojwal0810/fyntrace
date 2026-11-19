"""Offline automated tests for the retrieval evaluation and validation system."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import pytest
import sys
from unittest.mock import patch

from evaluation.evaluate import (
    calculate_metrics,
    compute_file_sha256,
    evaluate_retrieval,
    main,
    parse_question_record,
    validate_question_record_schema,
    validate_retrieval_questions,
    EXPECTED_COMPANY_ALLOCATIONS,
    EXPECTED_SECTOR_DIFFICULTY,
    EXPECTED_SECTOR_TOPIC,
)


def create_synthetic_benchmark(tmp_path: Path):
    """Generate a fully valid synthetic 25-question benchmark with matching chunks and canonical files."""
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    canonical_dir = tmp_path / "filing_text"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    sectors = [
        ("Technology", [("MSFT", 2024), ("AAPL", 2025)]),
        ("Financial services", [("JPM", 2024), ("WFC", 2023)]),
        ("Healthcare", [("PFE", 2023), ("JNJ", 2025)]),
        ("Energy and utilities", [("XOM", 2024), ("DUK", 2025)]),
        ("Consumer", [("WMT", 2024), ("MCD", 2023)]),
    ]

    records = []
    chunks = []
    canonical_files = {}

    qid_counter = 1
    for sector, companies in sectors:
        # Sector allocations: 5 questions total
        diffs = (
            ["easy", "medium", "hard", "easy", "medium"]
            if sector in ("Technology", "Healthcare", "Consumer")
            else ["easy", "medium", "medium", "medium", "hard"]
        )
        topics = [
            "Risk Factors",
            "Business / MD&A",
            "Financial-table or filing-specific evidence",
            "Risk Factors",
            "Business / MD&A",
        ]

        # Allocate between the two companies
        co1, fy1 = companies[0]
        co2, fy2 = companies[1]
        co1_count = EXPECTED_COMPANY_ALLOCATIONS[(co1, fy1)]
        co2_count = EXPECTED_COMPANY_ALLOCATIONS[(co2, fy2)]
        comp_plan = [(co1, fy1)] * co1_count + [(co2, fy2)] * co2_count

        for i in range(5):
            qid = f"RETR_{qid_counter:03d}"
            qid_counter += 1
            co, fy = comp_plan[i]
            diff = diffs[i]
            top = topics[i]
            acc = f"0000000001-{str(fy)[2:]}-{qid_counter:06d}"
            chunk_id = f"{co}_{fy}_10K_{acc.replace('-', '')}_0001"
            evidence_str = f"Synthetic evidence sentence for {qid} in {co} {fy}."
            chunk_text = f"Header. {evidence_str} Trailing text."
            dummy_sha = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

            # Add canonical text
            if acc not in canonical_files:
                canonical_files[acc] = ""
            char_start = len(canonical_files[acc])
            char_end = char_start + len(chunk_text)
            canonical_files[acc] += chunk_text + "\n\n"

            chunk = {
                "chunk_id": chunk_id,
                "ticker": co,
                "company_name": f"{co} Corp",
                "cik": "0000000001",
                "fiscal_year": fy,
                "form": "10-K",
                "accession": acc,
                "report_date": f"{fy}-12-31",
                "section": "Item 1 - Business",
                "section_confidence": "high",
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
                "source_url": f"https://www.sec.gov/Archives/edgar/data/1/{acc.replace('-', '')}/doc.htm",
                "file_sha256": dummy_sha,
            }
            chunks.append(chunk)

            record = {
                "id": qid,
                "category": "RETR",
                "difficulty": diff,
                "question": f"What is the synthetic finding for {qid} in {co}?",
                "companies": [co],
                "fiscal_years": [fy],
                "source_filing_types": ["10-K"],
                "expected_behavior": "answer",
                "gold_answer_numeric": None,
                "gold_unit": None,
                "gold_answer_text": f"Synthetic expected answer for {qid}.",
                "gold_inputs": None,
                "formula": None,
                "gold_source_accessions": [acc],
                "gold_evidence_text": evidence_str,
                "tolerance_abs": None,
                "tolerance_rel": None,
                "verification_note": f"Synthetic note for {qid}.",
                "attack_fixture": None,
                "topic": top,
                "sector": sector,
                "gold_chunk_ids": [chunk_id],
                "gold_source_url": f"https://www.sec.gov/Archives/edgar/data/1/{acc.replace('-', '')}/doc.htm",
                "gold_file_sha256": dummy_sha,
                "gold_mapping_status": "mapped",
                "human_review_status": "human_review_pending",
            }
            records.append(record)

    # Write questions
    with open(questions_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Write chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    # Write canonical texts
    for acc, txt in canonical_files.items():
        with open(canonical_dir / f"{acc}.txt", "w", encoding="utf-8") as f:
            f.write(txt)

    return questions_path, chunks_path, canonical_dir, records


def test_01_valid_benchmark_record_parsing(tmp_path):
    """Test 1: Valid benchmark record parsing."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    rec = parse_question_record(records[0])
    assert rec["id"] == "RETR_001"
    assert rec["category"] == "RETR"
    errs = validate_question_record_schema(rec)
    assert errs == []


def test_02_malformed_json(tmp_path):
    """Test 2: Malformed JSON detection."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    with open(q_path, "a", encoding="utf-8") as f:
        f.write("{malformed json line\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Malformed JSON" in e for e in errors)


def test_03_duplicate_question_ids(tmp_path):
    """Test 3: Duplicate question IDs detection."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[1]["id"] = records[0]["id"]
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Duplicate question ID" in e for e in errors)


def test_04_missing_required_fields(tmp_path):
    """Test 4: Missing required fields detection."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    del records[0]["gold_chunk_ids"]
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Missing required fields" in e for e in errors)


def test_05_wrong_question_count(tmp_path):
    """Test 5: Wrong question count detection."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Write only 24 questions
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records[:-1]:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Expected exactly 25 questions" in e for e in errors)


def test_06_wrong_difficulty_distribution(tmp_path):
    """Test 6: Wrong easy/medium/hard distribution."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Flip an easy to hard
    for r in records:
        if r["difficulty"] == "easy":
            r["difficulty"] = "hard"
            break
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Expected 8 easy questions" in e for e in errors)


def test_07_wrong_topic_distribution(tmp_path):
    """Test 7: Wrong topic distribution."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    for r in records:
        if r["topic"] == "Risk Factors":
            r["topic"] = "Business / MD&A"
            break
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Expected 10 Risk Factors questions" in e for e in errors)


def test_08_wrong_sector_distribution(tmp_path):
    """Test 8: Wrong sector distribution."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["sector"] = "Consumer"  # change Tech to Consumer
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Sector 'Technology': expected 5 questions" in e for e in errors)


def test_09_wrong_fiscal_year_distribution(tmp_path):
    """Test 9: Wrong fiscal-year distribution."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    for r in records:
        if r["fiscal_years"] == [2023]:
            r["fiscal_years"] = [2024]
            break
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Fiscal year 2023: expected 8 questions" in e for e in errors)


def test_10_wrong_company_allocation(tmp_path):
    """Test 10: Wrong company allocation."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Change MSFT to AAPL
    for r in records:
        if r["companies"] == ["MSFT"]:
            r["companies"] = ["AAPL"]
            break
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Company MSFT (2024): expected 2 questions" in e for e in errors)


def test_11_invalid_accession(tmp_path):
    """Test 11: Invalid accession format detection."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_source_accessions"] = ["invalid-accession-format"]
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Invalid accession format" in e for e in errors)


def test_12_invalid_sec_url(tmp_path):
    """Test 12: Invalid SEC URL detection."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_source_url"] = "http://insecure-host.com/filing.htm"
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Invalid gold_source_url" in e for e in errors)


def test_13_missing_gold_evidence(tmp_path):
    """Test 13: Missing gold evidence text when mapped."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_evidence_text"] = "   "
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("gold_evidence_text' must be a non-empty string" in e for e in errors)


def test_14_gold_chunk_incorrect_company_or_year(tmp_path):
    """Test 14: Gold chunk with incorrect company or fiscal year."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Point gold_chunk_ids of MSFT question to an AAPL chunk
    aapl_chunk_id = next(r["gold_chunk_ids"][0] for r in records if r["companies"] == ["AAPL"])
    records[0]["gold_chunk_ids"] = [aapl_chunk_id]
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("company/year" in e for e in errors)


def test_15_gold_chunk_fails_source_text_matching(tmp_path):
    """Test 15: Gold chunk that fails source-text matching."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_evidence_text"] = "This text does not exist anywhere in the chunk!"
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Gold evidence text not found in chunk" in e for e in errors)


def test_16_missing_supporting_chunk(tmp_path):
    """Test 16: Gold chunk ID not found in chunks file."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_chunk_ids"] = ["NONEXISTENT_CHUNK_ID_0001"]
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("not found in chunks file" in e for e in errors)


def test_17_correct_recall_calculation():
    """Test 17: Correct Recall@5 calculation."""
    eval_results = [
        {"hit": True, "reciprocal_rank": 1.0, "difficulty": "easy", "topic": "Risk Factors", "sector": "Technology"},
        {"hit": True, "reciprocal_rank": 0.5, "difficulty": "medium", "topic": "Business / MD&A", "sector": "Technology"},
        {"hit": False, "reciprocal_rank": 0.0, "difficulty": "hard", "topic": "Financial-table or filing-specific evidence", "sector": "Technology"},
        {"hit": True, "reciprocal_rank": 0.25, "difficulty": "easy", "topic": "Risk Factors", "sector": "Financial services"},
    ]
    metrics = calculate_metrics(eval_results)
    assert metrics["total_questions"] == 4
    assert metrics["hits"] == 3
    assert metrics["misses"] == 1
    assert metrics["recall"] == 0.75


def test_18_correct_mrr_calculation():
    """Test 18: Correct MRR calculation."""
    # 4 items with ranks: 1, 2, 4, and miss (0)
    # MRR = (1.0 + 0.5 + 0.25 + 0.0) / 4 = 1.75 / 4 = 0.4375
    eval_results = [
        {"hit": True, "reciprocal_rank": 1.0, "difficulty": "easy", "topic": "Risk Factors", "sector": "Technology"},
        {"hit": True, "reciprocal_rank": 0.5, "difficulty": "medium", "topic": "Business / MD&A", "sector": "Technology"},
        {"hit": True, "reciprocal_rank": 0.25, "difficulty": "hard", "topic": "Financial-table or filing-specific evidence", "sector": "Technology"},
        {"hit": False, "reciprocal_rank": 0.0, "difficulty": "easy", "topic": "Risk Factors", "sector": "Financial services"},
    ]
    metrics = calculate_metrics(eval_results)
    assert metrics["mrr"] == 0.4375


def test_19_correct_handling_ranks_1_through_5():
    """Test 19: Correct handling of rank 1 through rank 5."""
    ranks = [1, 2, 3, 4, 5]
    expected_rr = [1.0, 0.5, 1.0 / 3.0, 0.25, 0.2]
    eval_results = [
        {"hit": True, "rank": r, "reciprocal_rank": 1.0 / r, "difficulty": "easy", "topic": "Risk Factors", "sector": "Technology"}
        for r in ranks
    ]
    metrics = calculate_metrics(eval_results)
    expected_mrr = sum(expected_rr) / 5
    assert abs(metrics["mrr"] - expected_mrr) < 1e-9


def test_20_misses_contribute_zero():
    """Test 20: Misses contribute zero to MRR and Recall without shrinking denominator."""
    eval_results = [
        {"hit": True, "reciprocal_rank": 1.0, "difficulty": "easy", "topic": "Risk Factors", "sector": "Technology"},
        {"hit": False, "reciprocal_rank": 0.0, "difficulty": "easy", "topic": "Risk Factors", "sector": "Technology"},
    ]
    metrics = calculate_metrics(eval_results)
    assert metrics["total_questions"] == 2
    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5


def test_21_errors_included_in_denominator():
    """Test 21: Runtime errors are counted as misses and remain in denominator."""
    eval_results = [
        {"hit": True, "reciprocal_rank": 1.0, "error": None, "difficulty": "easy", "topic": "Risk Factors", "sector": "Technology"},
        {"hit": False, "reciprocal_rank": 0.0, "error": "Simulated query failure", "difficulty": "easy", "topic": "Risk Factors", "sector": "Technology"},
    ]
    metrics = calculate_metrics(eval_results)
    assert metrics["total_questions"] == 2
    assert metrics["hits"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5


def test_22_correct_company_year_filtering(tmp_path):
    """Test 22: Correct company/year filtering in retrieval evaluation."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    eval_output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5)
    assert eval_output["wrong_company_year_count"] == 0


def test_23_no_gold_answer_leakage(tmp_path):
    """Test 23: No gold answer, gold evidence, or gold IDs leaked into search_evidence."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    
    with patch("evaluation.evaluate.search_evidence") as mock_search:
        mock_search.return_value = {"status": "success", "results": []}
        evaluate_retrieval(q_path, chunks_path=c_path, top_k=5)

        for call in mock_search.call_args_list:
            kwargs = call.kwargs
            query = kwargs.get("query")
            # Query must not contain the gold answer or gold evidence
            assert "Synthetic expected answer" not in query
            assert "Synthetic evidence sentence" not in query
            # search_evidence must not receive gold_chunk_ids or notes
            assert "gold_chunk_ids" not in kwargs
            assert "gold_evidence_text" not in kwargs
            assert "verification_note" not in kwargs


def test_24_stable_repeated_results(tmp_path):
    """Test 24: Stable repeated results across multiple runs."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    run1 = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5)
    run2 = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5)
    assert run1["metrics"]["recall"] == run2["metrics"]["recall"]
    assert run1["metrics"]["mrr"] == run2["metrics"]["mrr"]
    assert [r["rank"] for r in run1["results"]] == [r["rank"] for r in run2["results"]]


def test_25_correct_question_file_sha256_reporting(tmp_path):
    """Test 25: Correct question-file SHA-256 reporting."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    expected_sha = hashlib.sha256(q_path.read_bytes()).hexdigest()
    eval_output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5)
    assert eval_output["question_file_sha256"] == expected_sha


def test_26_nonzero_exit_on_invalid_benchmark(tmp_path):
    """Test 26: Correct nonzero exit on invalid benchmark CLI invocation."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Break question count
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records[:10]:
            f.write(json.dumps(r) + "\n")

    exit_code = main(["validate-retrieval", "--questions", str(q_path)])
    assert exit_code != 0


def test_27_clean_execution_without_env_or_sec_files(tmp_path):
    """Test 27: Clean execution with synthetic files without real SEC files or .env."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    is_valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert is_valid
    assert errors == []
    output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5)
    assert output["metrics"]["total_questions"] == 25


def test_28_correct_handling_of_source_mapping_failure(tmp_path):
    """Test 28: Correct handling of source_present_chunk_missing status."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_mapping_status"] = "source_present_chunk_missing"
    records[0]["gold_chunk_ids"] = []
    records[0]["verification_note"] = "Evidence is present in 10-K Item 1A but was omitted during chunk extraction."
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert is_valid
    assert errors == []

    output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5)
    res0 = output["results"][0]
    assert res0["hit"] is False
    assert res0["rank"] is None
    assert res0["reciprocal_rank"] == 0.0
