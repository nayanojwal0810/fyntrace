"""Evaluation framework for Fyntrace BM25 filing retrieval system."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from fyntrace.retrieval import compute_file_sha256, get_index_metadata, search


COMPANY_TO_SECTOR: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "JPM": "Finance",
    "BAC": "Finance",
    "GS": "Finance",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "WMT": "Consumer",
    "COST": "Consumer",
    "MCD": "Consumer",
}

VALID_CATEGORIES: set[str] = {
    "business_operations",
    "risk_factors",
    "management_discussion",
    "financial_tables",
    "other_disclosures",
}

VALID_DIFFICULTIES: set[str] = {"easy", "medium", "hard"}

ALLOWED_AUDIT_DECISIONS: set[str] = {
    "verified",
    "corrected_gold",
    "corrected_question",
    "unresolved",
}

EXPECTED_SECTOR_COUNTS: dict[str, int] = {
    "Technology": 20,
    "Finance": 20,
    "Healthcare": 20,
    "Energy": 20,
    "Consumer": 20,
}

EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "business_operations": 20,
    "risk_factors": 25,
    "management_discussion": 25,
    "financial_tables": 20,
    "other_disclosures": 10,
}

EXPECTED_DIFFICULTY_COUNTS: dict[str, int] = {
    "easy": 20,
    "medium": 40,
    "hard": 40,
}

MANDATORY_QUESTION_FIELDS: tuple[str, ...] = (
    "question_id",
    "query",
    "ticker",
    "fiscal_year",
    "sector",
    "category",
    "difficulty",
    "gold_chunk_ids",
    "evidence_excerpt",
    "accession",
    "source_url",
    "section",
    "chunk_type",
    "corpus_sha256",
)


def load_questions(path: Path | str) -> list[dict[str, Any]]:
    """Load development questions from a JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Questions file not found: {p}")

    questions: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no} in {p} is malformed JSON: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {p} must be a JSON object")
            questions.append(obj)

    if not questions:
        raise ValueError(f"Questions file {p} is empty or contains no valid questions.")
    return questions


def load_audit(path: Path | str) -> list[dict[str, Any]]:
    """Load local audit records from a JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Audit file not found: {p}")

    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no} in audit file {p} is malformed JSON: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in audit file {p} must be a JSON object")
            records.append(obj)

    if not records:
        raise ValueError(f"Audit file {p} is empty or contains no valid records.")
    return records


def _normalize_category(cat: str) -> str:
    """Normalize category names to standard keys."""
    c = cat.strip().lower()
    if c in ("other_supported_disclosures", "other_disclosures", "other"):
        return "other_disclosures"
    return c


def validate_questions(
    questions: list[dict[str, Any]],
    db_path: Path | str,
    chunks_path: Path | str,
    text_dir: Path | str | None = None,
    enforce_100_dist: bool = True,
    check_canonical_text: bool = True,
) -> dict[str, Any]:
    """Validate question records against corpus, SQLite index, and specifications."""
    if not questions:
        raise ValueError("Questions list is empty.")

    db_file = Path(db_path)
    chunks_file = Path(chunks_path)

    if not db_file.is_file():
        raise FileNotFoundError(f"Retrieval database not found: {db_file}")
    if not chunks_file.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    # 1. Verify corpus and DB hashes
    actual_corpus_sha256 = compute_file_sha256(chunks_file)
    meta = get_index_metadata(db_file)
    db_corpus_sha256 = meta.get("corpus_sha256")
    if actual_corpus_sha256 != db_corpus_sha256:
        raise ValueError(
            f"Corpus hash mismatch: chunks.jsonl is {actual_corpus_sha256}, "
            f"but database recorded {db_corpus_sha256}"
        )

    # 2. Check 100-count requirements if production dev set
    if enforce_100_dist and len(questions) != 100:
        raise ValueError(f"Expected exactly 100 questions, got {len(questions)}")

    # 3. Check mandatory fields and duplicate IDs upfront
    seen_ids: set[str] = set()
    for idx, q in enumerate(questions, start=1):
        for field in MANDATORY_QUESTION_FIELDS:
            val = q.get(field)
            if val is None:
                raise ValueError(f"Question #{idx} missing mandatory field '{field}'")
            if isinstance(val, str) and not val.strip():
                raise ValueError(f"Question #{idx} has empty field '{field}'")
            if isinstance(val, list) and len(val) == 0:
                raise ValueError(f"Question #{idx} has empty list field '{field}'")

        qid = q["question_id"]
        if qid in seen_ids:
            raise ValueError(f"Duplicate question_id '{qid}' at #{idx}")
        seen_ids.add(qid)

    # 4. Connect to DB to load chunk metadata
    con = sqlite3.connect(str(db_file))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute("""
            SELECT chunk_id, ticker, fiscal_year, accession, section,
                   chunk_type, text, char_start, char_end, source_url,
                   source_file_sha256, canonical_text_path
            FROM chunks;
        """)
        db_chunks = {r["chunk_id"]: dict(r) for r in cur.fetchall()}
    finally:
        con.close()

    sector_counts: dict[str, int] = {k: 0 for k in EXPECTED_SECTOR_COUNTS}
    category_counts: dict[str, int] = {k: 0 for k in EXPECTED_CATEGORY_COUNTS}
    difficulty_counts: dict[str, int] = {k: 0 for k in EXPECTED_DIFFICULTY_COUNTS}
    pair_counts: dict[tuple[str, int], int] = {}
    text_cache: dict[str, str] = {}

    for q in questions:
        qid = q["question_id"]
        query = q["query"]
        ticker = q["ticker"]
        if ticker not in COMPANY_TO_SECTOR:
            raise ValueError(f"Question {qid} has invalid/unsupported ticker {ticker!r}")

        year = q["fiscal_year"]
        if year not in (2023, 2024, 2025):
            raise ValueError(f"Question {qid} has invalid fiscal_year {year!r}")

        sector = q["sector"]
        expected_sector = COMPANY_TO_SECTOR[ticker]
        if sector != expected_sector:
            raise ValueError(
                f"Question {qid} sector mismatch: ticker {ticker} is in {expected_sector}, got {sector!r}"
            )

        cat_raw = q["category"]
        cat = _normalize_category(cat_raw)
        if cat not in VALID_CATEGORIES:
            raise ValueError(f"Question {qid} has invalid category {cat_raw!r}")

        diff = q["difficulty"]
        diff_lower = diff.strip().lower()
        if diff_lower not in VALID_DIFFICULTIES:
            raise ValueError(f"Question {qid} has invalid difficulty {diff!r} (must be easy, medium, or hard)")

        corpus_hash = q["corpus_sha256"]
        if corpus_hash != actual_corpus_sha256:
            raise ValueError(
                f"Question {qid} corpus_sha256 '{corpus_hash}' does not match actual '{actual_corpus_sha256}'"
            )

        gold_ids = q["gold_chunk_ids"]
        excerpt = q["evidence_excerpt"]

        # Validate each gold chunk ID against DB
        for gid in gold_ids:
            if gid not in db_chunks:
                raise ValueError(f"Question {qid} gold_chunk_id '{gid}' not found in index")
            chunk = db_chunks[gid]

            if chunk["ticker"] != ticker:
                raise ValueError(
                    f"Question {qid} specifies ticker {ticker}, but gold chunk {gid} belongs to {chunk['ticker']}"
                )
            if chunk["fiscal_year"] != year:
                raise ValueError(
                    f"Question {qid} specifies year {year}, but gold chunk {gid} belongs to {chunk['fiscal_year']}"
                )
            if q["accession"] != chunk["accession"]:
                raise ValueError(
                    f"Question {qid} accession '{q['accession']}' != chunk '{chunk['accession']}'"
                )
            if q["source_url"] != chunk["source_url"]:
                raise ValueError(
                    f"Question {qid} source_url '{q['source_url']}' != chunk '{chunk['source_url']}'"
                )
            if q["section"] != chunk["section"]:
                raise ValueError(
                    f"Question {qid} section '{q['section']}' != chunk '{chunk['section']}'"
                )
            if q["chunk_type"] != chunk["chunk_type"]:
                raise ValueError(
                    f"Question {qid} chunk_type '{q['chunk_type']}' != chunk '{chunk['chunk_type']}'"
                )

            # Excerpt must be present in gold chunk text
            if excerpt not in chunk["text"]:
                raise ValueError(
                    f"Question {qid} evidence_excerpt not found in gold chunk {gid} text"
                )

            # Check character offsets against canonical text file
            if check_canonical_text:
                c_path_str = chunk.get("canonical_text_path")
                if not c_path_str:
                    raise ValueError(f"Question {qid} chunk {gid} missing canonical_text_path")
                txt_path = Path(c_path_str)
                if not txt_path.is_file() and text_dir:
                    txt_path = Path(text_dir) / txt_path.name
                if not txt_path.is_file() and chunks_file.parent:
                    candidate = chunks_file.parent / c_path_str
                    if candidate.is_file():
                        txt_path = candidate
                if not txt_path.is_file():
                    raise FileNotFoundError(
                        f"Canonical text file not found for Question {qid} chunk {gid}: {txt_path}"
                    )
                if str(txt_path) not in text_cache:
                    text_cache[str(txt_path)] = txt_path.read_text(encoding="utf-8")
                full_text = text_cache[str(txt_path)]
                start, end = chunk["char_start"], chunk["char_end"]
                if full_text[start:end] != chunk["text"]:
                    raise ValueError(
                        f"Question {qid} chunk {gid} char offsets do not match canonical text"
                    )

        # Track distributions
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        category_counts[cat] = category_counts.get(cat, 0) + 1
        difficulty_counts[diff_lower] = difficulty_counts.get(diff_lower, 0) + 1
        pair_counts[(ticker, year)] = pair_counts.get((ticker, year), 0) + 1

    # Check distributions if full 100-question set
    if enforce_100_dist:
        for d, expected in EXPECTED_DIFFICULTY_COUNTS.items():
            if difficulty_counts.get(d, 0) != expected:
                raise ValueError(f"Difficulty {d} has {difficulty_counts.get(d, 0)} questions; expected {expected}")

        for s, expected in EXPECTED_SECTOR_COUNTS.items():
            if sector_counts.get(s, 0) != expected:
                raise ValueError(f"Sector {s} has {sector_counts.get(s, 0)} questions; expected {expected}")

        for c, expected in EXPECTED_CATEGORY_COUNTS.items():
            if category_counts.get(c, 0) != expected:
                raise ValueError(f"Category {c} has {category_counts.get(c, 0)} questions; expected {expected}")

        for t in COMPANY_TO_SECTOR:
            for y in (2023, 2024, 2025):
                cnt = pair_counts.get((t, y), 0)
                if cnt < 2:
                    raise ValueError(f"Company/year pair ({t}, {y}) has {cnt} questions; minimum required is 2")

    return {
        "status": "valid",
        "question_count": len(questions),
        "corpus_sha256": actual_corpus_sha256,
        "sector_counts": sector_counts,
        "category_counts": category_counts,
        "difficulty_counts": difficulty_counts,
    }


def validate_audit(
    audit_records: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    enforce_100: bool = True,
) -> dict[str, Any]:
    """Validate local audit records against audited question dataset."""
    if not audit_records:
        raise ValueError("Audit records list is empty.")

    if enforce_100 and len(audit_records) != 100:
        raise ValueError(f"Expected exactly 100 audit records, got {len(audit_records)}")
    if enforce_100 and len(questions) != 100:
        raise ValueError(f"Expected exactly 100 questions, got {len(questions)}")

    q_map = {q["question_id"]: q for q in questions}
    seen_ids: set[str] = set()
    decision_counts: dict[str, int] = {}

    for idx, rec in enumerate(audit_records, start=1):
        qid = rec.get("question_id")
        if not qid or not isinstance(qid, str):
            raise ValueError(f"Audit record #{idx} missing valid 'question_id'")
        if qid in seen_ids:
            raise ValueError(f"Duplicate question_id '{qid}' in audit records")
        seen_ids.add(qid)

        if qid not in q_map:
            raise ValueError(f"Audit record '{qid}' does not match any question in dataset")

        q = q_map[qid]
        decision = rec.get("decision")
        if decision not in ALLOWED_AUDIT_DECISIONS:
            raise ValueError(f"Audit record '{qid}' has invalid decision {decision!r}")
        if decision == "unresolved":
            raise ValueError(
                f"Audit record '{qid}' is marked 'unresolved'; audit cannot be accepted with unresolved records"
            )

        decision_counts[decision] = decision_counts.get(decision, 0) + 1

        evidence_support = rec.get("evidence_support")
        if not evidence_support or not isinstance(evidence_support, str) or not evidence_support.strip():
            raise ValueError(f"Audit record '{qid}' missing required 'evidence_support' explanation")

        verified_excerpt = rec.get("verified_evidence_excerpt")
        if not verified_excerpt or not isinstance(verified_excerpt, str) or not verified_excerpt.strip():
            raise ValueError(f"Audit record '{qid}' missing 'verified_evidence_excerpt'")

        # If gold label was changed, require correction reason
        gold_changed = rec.get("gold_label_changed", False)
        if (gold_changed or decision == "corrected_gold") and not rec.get("correction_reason"):
            raise ValueError(f"Audit record '{qid}' has corrected gold label but missing 'correction_reason'")

        # If question was changed, require correction reason
        q_changed = rec.get("question_changed", False)
        if (q_changed or decision == "corrected_question") and not rec.get("correction_reason"):
            raise ValueError(f"Audit record '{qid}' has corrected question but missing 'correction_reason'")

        # Final gold chunk IDs must match question
        final_golds = rec.get("final_gold_chunk_ids")
        if final_golds != q.get("gold_chunk_ids"):
            raise ValueError(
                f"Audit record '{qid}' final_gold_chunk_ids {final_golds} does not match question gold_chunk_ids {q.get('gold_chunk_ids')}"
            )

        # Final difficulty must match question
        final_diff = rec.get("final_difficulty")
        if final_diff != q.get("difficulty"):
            raise ValueError(
                f"Audit record '{qid}' final_difficulty {final_diff!r} does not match question difficulty {q.get('difficulty')!r}"
            )

    # Check that every question in dataset has an audit entry
    for qid in q_map:
        if qid not in seen_ids:
            raise ValueError(f"Question '{qid}' is missing an audit record")

    return {
        "status": "valid",
        "audit_record_count": len(audit_records),
        "decisions": decision_counts,
    }


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Compute median, p95, and average latency in milliseconds."""
    if not latencies_ms:
        return {"median_ms": 0.0, "p95_ms": 0.0, "avg_ms": 0.0}
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    avg_l = sum(sorted_l) / n
    if n % 2 == 1:
        med_l = sorted_l[n // 2]
    else:
        med_l = (sorted_l[n // 2 - 1] + sorted_l[n // 2]) / 2.0
    p95_idx = min(int(round(0.95 * n + 0.5)) - 1, n - 1)
    p95_idx = max(0, p95_idx)
    p95_l = sorted_l[p95_idx]
    return {
        "median_ms": round(med_l, 2),
        "p95_ms": round(p95_l, 2),
        "avg_ms": round(avg_l, 2),
    }


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    db_path: Path | str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute BM25 evaluation over questions and compute standard metrics."""
    db_file = Path(db_path)
    if not db_file.is_file():
        raise FileNotFoundError(f"Retrieval database not found: {db_file}")

    meta = get_index_metadata(db_file)
    corpus_sha256 = meta.get("corpus_sha256", "")

    per_question_results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    filter_violations = 0
    total_results_checked = 0
    failures: list[dict[str, Any]] = []

    for q in questions:
        qid = q["question_id"]
        query = q["query"]
        ticker = q["ticker"]
        year = q["fiscal_year"]
        sector = q.get("sector", COMPANY_TO_SECTOR.get(ticker, "Unknown"))
        cat = _normalize_category(q.get("category", "unknown"))
        diff = q.get("difficulty", "medium").strip().lower()
        gold_ids = list(q["gold_chunk_ids"])

        t0 = time.perf_counter()
        results = search(
            query=query,
            ticker=ticker,
            year=year,
            top_k=top_k,
            db_path=db_file,
        )
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_elapsed_ms)

        retrieved_ids = [r["chunk_id"] for r in results]
        retrieved_sections = [r["section"] for r in results]

        # Verify filters
        for r in results:
            total_results_checked += 1
            if r["ticker"] != ticker or r["fiscal_year"] != year:
                filter_violations += 1

        # Calculate metrics for this question
        gold_set = set(gold_ids)
        retrieved_set = set(retrieved_ids)
        found_golds = gold_set & retrieved_set
        hit = 1.0 if len(found_golds) > 0 else 0.0
        recall = len(found_golds) / len(gold_set) if gold_set else 0.0

        # Find rank of first gold chunk (1-based)
        first_gold_rank: int | None = None
        for r_idx, cid in enumerate(retrieved_ids, start=1):
            if cid in gold_set:
                first_gold_rank = r_idx
                break

        if first_gold_rank is not None:
            mrr = 1.0 / first_gold_rank
        else:
            mrr = 0.0

        q_res = {
            "question_id": qid,
            "query": query,
            "ticker": ticker,
            "fiscal_year": year,
            "sector": sector,
            "category": cat,
            "difficulty": diff,
            "gold_chunk_ids": gold_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_sections": retrieved_sections,
            "hit": hit,
            "recall": recall,
            "mrr": mrr,
            "rank": first_gold_rank,
            "latency_ms": round(t_elapsed_ms, 2),
        }
        per_question_results.append(q_res)

        if hit == 0.0:
            # Diagnose heuristic category
            heuristic_type = "Lexical mismatch"
            if cat == "financial_tables":
                heuristic_type = "Table-heavy content"
            elif any("Item 15" in sec for sec in retrieved_sections):
                heuristic_type = "Section mismatch / Exhibit schedule"
            elif not retrieved_ids:
                heuristic_type = "No matches returned"

            failures.append({
                "question_id": qid,
                "query": query,
                "ticker": ticker,
                "fiscal_year": year,
                "sector": sector,
                "category": cat,
                "difficulty": diff,
                "gold_chunk_ids": gold_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "rank": None,
                "retrieved_sections": retrieved_sections,
                "evidence_excerpt": q.get("evidence_excerpt", ""),
                "source_url": q.get("source_url", ""),
                "observed_failure_type": "unreviewed",
                "heuristic_failure_mode": heuristic_type,
            })

    total_q = len(per_question_results)
    overall_recall = sum(r["recall"] for r in per_question_results) / total_q if total_q else 0.0
    overall_hit = sum(r["hit"] for r in per_question_results) / total_q if total_q else 0.0
    overall_mrr = sum(r["mrr"] for r in per_question_results) / total_q if total_q else 0.0
    latency_stats = compute_latency_stats(latencies_ms)

    def _group_metrics(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in per_question_results:
            val = str(r[key])
            groups.setdefault(val, []).append(r)
        breakdown = {}
        for val, items in groups.items():
            cnt = len(items)
            r_avg = sum(it["recall"] for it in items) / cnt
            h_avg = sum(it["hit"] for it in items) / cnt
            m_avg = sum(it["mrr"] for it in items) / cnt
            missed = sum(1 for it in items if it["hit"] == 0.0)
            breakdown[val] = {
                "question_count": cnt,
                "recall_at_5": round(r_avg, 4),
                "hit_at_5": round(h_avg, 4),
                "mrr_at_5": round(m_avg, 4),
                "missed_count": missed,
            }
        return breakdown

    return {
        "corpus_sha256": corpus_sha256,
        "evaluation_question_count": total_q,
        "top_k": top_k,
        "overall": {
            "recall_at_5": round(overall_recall, 4),
            "hit_at_5": round(overall_hit, 4),
            "mrr_at_5": round(overall_mrr, 4),
            "filter_violations": filter_violations,
            "total_results_checked": total_results_checked,
            "total_missed_questions": len(failures),
        },
        "latency_stats": latency_stats,
        "by_sector": _group_metrics("sector"),
        "by_category": _group_metrics("category"),
        "by_fiscal_year": _group_metrics("fiscal_year"),
        "by_difficulty": _group_metrics("difficulty"),
        "failures": failures,
        "per_question": per_question_results,
    }


def save_results(results: dict[str, Any], output_path: Path | str) -> None:
    """Save evaluation results as JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    """CLI entrypoint for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m fyntrace.retrieval_eval",
        description="Fyntrace BM25 retrieval evaluation tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: validate
    val_parser = subparsers.add_parser("validate", help="Validate evaluation questions dataset.")
    val_parser.add_argument(
        "--questions",
        required=True,
        help="Path to dev_questions.jsonl",
    )
    val_parser.add_argument(
        "--db-path",
        default="data/processed/retrieval.sqlite",
        help="Path to SQLite database",
    )
    val_parser.add_argument(
        "--chunks-path",
        default="data/processed/chunks.jsonl",
        help="Path to chunks.jsonl",
    )
    val_parser.add_argument(
        "--text-dir",
        default="data/processed/text",
        help="Path to canonical text directory",
    )
    val_parser.add_argument(
        "--audit-path",
        default=None,
        help="Optional path to local audit JSONL file (e.g. data/processed/retrieval_dev_audit.jsonl)",
    )

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run retrieval evaluation.")
    run_parser.add_argument(
        "--questions",
        required=True,
        help="Path to dev_questions.jsonl",
    )
    run_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval parameter (default 5)",
    )
    run_parser.add_argument(
        "--db-path",
        default="data/processed/retrieval.sqlite",
        help="Path to SQLite database",
    )
    run_parser.add_argument(
        "--chunks-path",
        default="data/processed/chunks.jsonl",
        help="Path to chunks.jsonl",
    )
    run_parser.add_argument(
        "--text-dir",
        default="data/processed/text",
        help="Path to canonical text directory",
    )
    run_parser.add_argument(
        "--output",
        default="data/processed/retrieval_dev_results.json",
        help="Path to output results file",
    )

    args = parser.parse_args()

    if args.command == "validate":
        questions = load_questions(args.questions)
        val_summary = validate_questions(
            questions=questions,
            db_path=args.db_path,
            chunks_path=args.chunks_path,
            text_dir=args.text_dir,
            enforce_100_dist=True,
            check_canonical_text=True,
        )
        print(f"Validation SUCCESS: {val_summary['question_count']} questions verified.")
        print(f"Corpus SHA-256: {val_summary['corpus_sha256']}")
        print(f"Sector counts: {val_summary['sector_counts']}")
        print(f"Category counts: {val_summary['category_counts']}")
        print(f"Difficulty counts: {val_summary['difficulty_counts']}")

        if args.audit_path:
            audit_records = load_audit(args.audit_path)
            audit_summary = validate_audit(
                audit_records=audit_records,
                questions=questions,
                enforce_100=True,
            )
            print(f"Audit validation SUCCESS: {audit_summary['audit_record_count']} audit records verified.")
            print(f"Audit decisions: {audit_summary['decisions']}")

    elif args.command == "run":
        questions = load_questions(args.questions)
        validate_questions(
            questions=questions,
            db_path=args.db_path,
            chunks_path=args.chunks_path,
            text_dir=args.text_dir,
            enforce_100_dist=True,
            check_canonical_text=True,
        )
        results = evaluate_retrieval(
            questions=questions,
            db_path=args.db_path,
            top_k=args.top_k,
        )
        save_results(results, args.output)

        ov = results["overall"]
        lat = results["latency_stats"]
        print("--- Retrieval Evaluation Summary ---")
        print(f"Questions: {results['evaluation_question_count']}")
        print(f"Recall@{args.top_k}: {ov['recall_at_5']:.4f}")
        print(f"Hit@{args.top_k}: {ov['hit_at_5']:.4f}")
        print(f"MRR@{args.top_k}: {ov['mrr_at_5']:.4f}")
        print(f"Filter violations: {ov['filter_violations']} / {ov['total_results_checked']}")
        print(f"Latency: median={lat['median_ms']}ms, p95={lat['p95_ms']}ms, avg={lat['avg_ms']}ms")
        print(f"Missed questions: {ov['total_missed_questions']}")
        print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
