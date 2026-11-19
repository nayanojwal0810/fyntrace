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
    generate_public_results,
    main,
    parse_question_record,
    validate_question_record_schema,
    validate_retrieval_questions,
    EXPECTED_COMPANY_ALLOCATIONS,
    EXPECTED_SECTOR_DIFFICULTY,
    EXPECTED_SECTOR_TOPIC,
)


def create_synthetic_benchmark(tmp_path: Path):
    """Generate a fully valid synthetic 25-question benchmark with matching chunks, canonical, raw, and manifest files."""
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    canonical_dir = tmp_path / "filing_text"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw_filings"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "filing_manifest.json"

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
    manifest_entries = []
    seen_manifest_pairs = set()

    qid_counter = 1
    for sector, companies in sectors:
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
            dummy_raw_html = f"<html><body><p>{chunk_text}</p></body></html>"
            dummy_sha = hashlib.sha256(dummy_raw_html.encode("utf-8")).hexdigest()

            # Add raw HTML
            raw_file = raw_dir / f"{acc}.html"
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(dummy_raw_html)

            # Add manifest entry once per company-year
            if (co, fy) not in seen_manifest_pairs:
                seen_manifest_pairs.add((co, fy))
                manifest_entries.append({
                    "ticker": co,
                    "company_name": f"{co} Corp",
                    "cik": "0000000001",
                    "sector": sector,
                    "target_fiscal_year": fy,
                    "fiscal_year_status": "confirmed",
                    "form": "10-K",
                    "accession_number": acc,
                    "filing_date": f"{fy}-12-31",
                    "report_date": f"{fy}-12-31",
                    "primary_document": "doc.htm",
                    "source_url": f"https://www.sec.gov/Archives/edgar/data/1/{acc.replace('-', '')}/doc.htm",
                })

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

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

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
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records[:-1]:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Expected exactly 25 questions" in e for e in errors)


def test_06_wrong_difficulty_distribution(tmp_path):
    """Test 6: Wrong easy/medium/hard distribution."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
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
    records[0]["sector"] = "Consumer"
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
    eval_output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)
    assert eval_output["wrong_company_year_count"] == 0


def test_23_no_gold_answer_leakage(tmp_path):
    """Test 23: No gold answer, gold evidence, or gold IDs leaked into search_evidence."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)

    with patch("evaluation.evaluate.search_evidence") as mock_search:
        mock_search.return_value = {"status": "success", "results": []}
        evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)

        for call in mock_search.call_args_list:
            kwargs = call.kwargs
            query = kwargs.get("query")
            assert "Synthetic expected answer" not in query
            assert "Synthetic evidence sentence" not in query
            assert "gold_chunk_ids" not in kwargs
            assert "gold_evidence_text" not in kwargs
            assert "verification_note" not in kwargs


def test_24_stable_repeated_results(tmp_path):
    """Test 24: Stable repeated results across multiple runs."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    run1 = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)
    run2 = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)
    assert run1["metrics"]["recall"] == run2["metrics"]["recall"]
    assert run1["metrics"]["mrr"] == run2["metrics"]["mrr"]
    assert [r["rank"] for r in run1["results"]] == [r["rank"] for r in run2["results"]]


def test_25_correct_question_file_sha256_reporting(tmp_path):
    """Test 25: Correct question-file SHA-256 reporting."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    expected_sha = hashlib.sha256(q_path.read_bytes()).hexdigest()
    eval_output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)
    assert eval_output["question_file_sha256"] == expected_sha


def test_26_nonzero_exit_on_invalid_benchmark(tmp_path):
    """Test 26: Correct nonzero exit on invalid benchmark CLI invocation."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
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
    output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)
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

    output = evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)
    res0 = output["results"][0]
    assert res0["hit"] is False
    assert res0["rank"] is None
    assert res0["reciprocal_rank"] == 0.0


def test_29_wrong_source_sha256(tmp_path):
    """Test 29: Detection of invalid or mismatched source SHA-256."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Mismatch with chunk's file_sha256
    records[0]["gold_file_sha256"] = "a" * 64
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("file_sha256" in e for e in errors)


def test_30_missing_required_source_hash(tmp_path):
    """Test 30: Detection of missing gold_file_sha256."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_file_sha256"] = None
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("Missing required gold_file_sha256" in e for e in errors)


def test_31_wrong_filing_form(tmp_path):
    """Test 31: Detection of non-10-K filing form."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["source_filing_types"] = ["10-Q"]
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("source_filing_types" in e for e in errors)


def test_32_incomplete_chunk_mapping(tmp_path):
    """Test 32: Detection of empty gold_chunk_ids when mapping status is mapped."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    records[0]["gold_chunk_ids"] = []
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("cannot be empty when gold_mapping_status is 'mapped'" in e for e in errors)


def test_33_missing_original_html_in_strict_source_mode(tmp_path):
    """Test 33: Detection of missing original SEC HTML in strict source verification mode."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    raw_dir = tmp_path / "raw_filings"
    manifest_path = tmp_path / "filing_manifest.json"

    # Delete one raw HTML file
    acc = records[0]["gold_source_accessions"][0]
    raw_file = raw_dir / f"{acc}.html"
    if raw_file.is_file():
        raw_file.unlink()

    valid, errors = validate_retrieval_questions(
        q_path,
        chunks_path=c_path,
        canonical_dir=can_dir,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        verify_sources=True,
    )
    assert not valid
    assert any("Original SEC HTML file not found" in e for e in errors)


def test_34_incorrect_original_html_sha256_in_strict_source_mode(tmp_path):
    """Test 34: Detection of HTML file SHA-256 mismatch in strict source verification mode."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    raw_dir = tmp_path / "raw_filings"
    manifest_path = tmp_path / "filing_manifest.json"

    # Corrupt one raw HTML file content
    acc = records[0]["gold_source_accessions"][0]
    raw_file = raw_dir / f"{acc}.html"
    with open(raw_file, "a", encoding="utf-8") as f:
        f.write("<!-- corrupted byte -->")

    valid, errors = validate_retrieval_questions(
        q_path,
        chunks_path=c_path,
        canonical_dir=can_dir,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        verify_sources=True,
    )
    assert not valid
    assert any("Original HTML SHA-256" in e for e in errors)


def test_35_incorrect_canonical_text_offset(tmp_path):
    """Test 35: Detection of canonical-text character offset mismatch."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Corrupt canonical file
    acc = records[0]["gold_source_accessions"][0]
    can_file = can_dir / f"{acc}.txt"
    with open(can_file, "w", encoding="utf-8") as f:
        f.write("Shifted prefix text " + can_file.read_text(encoding="utf-8"))

    valid, errors = validate_retrieval_questions(q_path, c_path, can_dir)
    assert not valid
    assert any("offset span mismatch" in e for e in errors)


def test_36_invalid_question_file_rejected_before_scoring(tmp_path):
    """Test 36: Evaluate rejects invalid questions file before scoring without running retrieval."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    # Truncate to 10 questions
    with open(q_path, "w", encoding="utf-8") as f:
        for r in records[:10]:
            f.write(json.dumps(r) + "\n")

    with pytest.raises(ValueError, match="Pre-scoring validation failed"):
        evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)


def test_37_dataset_freeze_hash_verified_after_scoring(tmp_path):
    """Test 37: Evaluate raises error if question file hash is altered during scoring."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)

    # Patch search_evidence to tamper with the question file mid-evaluation
    from src import retrieval as ret_mod
    orig_search = ret_mod.search_evidence

    def tamper_search(*args, **kwargs):
        with open(q_path, "a", encoding="utf-8") as f:
            f.write("\n")
        return orig_search(*args, **kwargs)

    with patch("evaluation.evaluate.search_evidence", side_effect=tamper_search):
        with pytest.raises(RuntimeError, match="FATAL: Questions file SHA-256 changed during evaluation"):
            evaluate_retrieval(q_path, chunks_path=c_path, top_k=5, canonical_dir=can_dir)


def test_38_correct_public_summary_generation(tmp_path):
    """Test 38: generate_public_results produces valid summary file."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    out_json = tmp_path / "public_results.json"
    eval_output = evaluate_retrieval(
        q_path,
        chunks_path=c_path,
        top_k=5,
        canonical_dir=can_dir,
        publish_results_path=out_json,
    )
    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["evaluation_type"] == "retrieval"
    assert data["evaluation_status"] == "provisional_human_review_pending"
    assert data["evaluation_question_count"] == 25
    assert data["top_k"] == 5
    assert "recall_at_top_k" in data
    assert "mrr" in data
    assert "results_by_difficulty" in data
    assert "results_by_topic" in data
    assert "results_by_sector" in data
    assert "missing_questions" in data
    assert "notes" in data


def test_39_public_results_match_actual_run(tmp_path):
    """Test 39: Public results match the actual evaluation run metrics."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    out_json = tmp_path / "public_results.json"
    eval_output = evaluate_retrieval(
        q_path,
        chunks_path=c_path,
        top_k=5,
        canonical_dir=can_dir,
        publish_results_path=out_json,
    )
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["recall_at_top_k"] == round(eval_output["metrics"]["recall"], 4)
    assert data["mrr"] == round(eval_output["metrics"]["mrr"], 4)
    assert data["hits"] == eval_output["metrics"]["hits"]
    assert data["misses"] == eval_output["metrics"]["misses"]


def test_40_no_historical_scores_in_public_results(tmp_path):
    """Test 40: Public results summary contains no historical runs or comparisons."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    out_json = tmp_path / "public_results.json"
    evaluate_retrieval(
        q_path,
        chunks_path=c_path,
        top_k=5,
        canonical_dir=can_dir,
        publish_results_path=out_json,
    )
    with open(out_json, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_keys = ["previous", "prior", "baseline_score", "history", "old_recall", "run1", "run2"]
    for k in forbidden_keys:
        assert k not in content.lower()


def test_41_no_public_version_labels(tmp_path):
    """Test 41: Public results summary contains no public version labels."""
    q_path, c_path, can_dir, records = create_synthetic_benchmark(tmp_path)
    out_json = tmp_path / "public_results.json"
    evaluate_retrieval(
        q_path,
        chunks_path=c_path,
        top_k=5,
        canonical_dir=can_dir,
        publish_results_path=out_json,
    )
    with open(out_json, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_patterns = [r"\bv1\b", r"\bv2\b", r"\bv3\b", r"benchmark version", r"version 1", r"version 2"]
    import re
    for pat in forbidden_patterns:
        assert not re.search(pat, content, re.IGNORECASE), f"Found forbidden version pattern: {pat}"
