"""Retrieval evaluation and validation for SEC filing evidence retrieval."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from src.retrieval import build_index, load_chunks, search_evidence

REQUIRED_GUIDEBOOK_FIELDS = [
    "id",
    "category",
    "difficulty",
    "question",
    "companies",
    "fiscal_years",
    "source_filing_types",
    "expected_behavior",
    "gold_answer_numeric",
    "gold_unit",
    "gold_answer_text",
    "gold_inputs",
    "formula",
    "gold_source_accessions",
    "gold_evidence_text",
    "tolerance_abs",
    "tolerance_rel",
    "verification_note",
    "attack_fixture",
]

REQUIRED_RETRIEVAL_FIELDS = [
    "topic",
    "sector",
    "gold_chunk_ids",
    "gold_source_url",
    "gold_file_sha256",
    "gold_mapping_status",
    "human_review_status",
]

ALL_REQUIRED_FIELDS = REQUIRED_GUIDEBOOK_FIELDS + REQUIRED_RETRIEVAL_FIELDS

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_TOPICS = {
    "Risk Factors",
    "Business / MD&A",
    "Financial-table or filing-specific evidence",
}
VALID_SECTORS = {
    "Technology",
    "Financial services",
    "Healthcare",
    "Energy and utilities",
    "Consumer",
}

EXPECTED_SECTOR_DIFFICULTY = {
    "Technology": {"easy": 2, "medium": 2, "hard": 1},
    "Financial services": {"easy": 1, "medium": 3, "hard": 1},
    "Healthcare": {"easy": 2, "medium": 2, "hard": 1},
    "Energy and utilities": {"easy": 1, "medium": 3, "hard": 1},
    "Consumer": {"easy": 2, "medium": 2, "hard": 1},
}

EXPECTED_SECTOR_TOPIC = {
    "Technology": {"Risk Factors": 2, "Business / MD&A": 2, "Financial-table or filing-specific evidence": 1},
    "Financial services": {"Risk Factors": 2, "Business / MD&A": 2, "Financial-table or filing-specific evidence": 1},
    "Healthcare": {"Risk Factors": 2, "Business / MD&A": 2, "Financial-table or filing-specific evidence": 1},
    "Energy and utilities": {"Risk Factors": 2, "Business / MD&A": 2, "Financial-table or filing-specific evidence": 1},
    "Consumer": {"Risk Factors": 2, "Business / MD&A": 2, "Financial-table or filing-specific evidence": 1},
}

EXPECTED_COMPANY_ALLOCATIONS = {
    ("MSFT", 2024): 2,
    ("AAPL", 2025): 3,
    ("JPM", 2024): 2,
    ("WFC", 2023): 3,
    ("PFE", 2023): 2,
    ("JNJ", 2025): 3,
    ("XOM", 2024): 3,
    ("DUK", 2025): 2,
    ("WMT", 2024): 2,
    ("MCD", 2023): 3,
}

EXPECTED_FY_COUNTS = {2023: 8, 2024: 9, 2025: 8}

ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
SEC_URL_PREFIX = "https://www.sec.gov/Archives/edgar/data/"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_file_sha256(path: Path | str) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_question_record(raw: Any, line_no: int = 1) -> dict[str, Any]:
    """Parse and check presence of required fields in a question record."""
    if not isinstance(raw, dict):
        raise ValueError(f"Line {line_no}: Expected JSON object, got {type(raw).__name__}")
    missing = [field for field in ALL_REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"Line {line_no} (ID: {raw.get('id', 'unknown')}): Missing required fields: {missing}")
    return raw


def validate_question_record_schema(q: dict[str, Any]) -> list[str]:
    """Validate schema, types, and values of an individual question record."""
    errors = []
    qid = str(q.get("id", ""))

    if not qid:
        errors.append("Question missing 'id'")

    if q.get("category") != "RETR":
        errors.append(f"[{qid}] Expected category 'RETR', got {q.get('category')!r}")

    if q.get("expected_behavior") != "answer":
        errors.append(f"[{qid}] Expected expected_behavior 'answer', got {q.get('expected_behavior')!r}")

    diff = q.get("difficulty")
    if diff not in VALID_DIFFICULTIES:
        errors.append(f"[{qid}] Invalid difficulty {diff!r}, expected one of {sorted(VALID_DIFFICULTIES)}")

    topic = q.get("topic")
    if topic not in VALID_TOPICS:
        errors.append(f"[{qid}] Invalid topic {topic!r}, expected one of {sorted(VALID_TOPICS)}")

    sector = q.get("sector")
    if sector not in VALID_SECTORS:
        errors.append(f"[{qid}] Invalid sector {sector!r}, expected one of {sorted(VALID_SECTORS)}")

    qtext = q.get("question")
    if not isinstance(qtext, str) or not qtext.strip():
        errors.append(f"[{qid}] 'question' must be a non-empty string")

    companies = q.get("companies")
    if not isinstance(companies, list) or len(companies) != 1 or not isinstance(companies[0], str) or not companies[0].strip():
        errors.append(f"[{qid}] 'companies' must be a list containing exactly one ticker string")

    fys = q.get("fiscal_years")
    if not isinstance(fys, list) or len(fys) != 1 or not isinstance(fys[0], int) or isinstance(fys[0], bool):
        errors.append(f"[{qid}] 'fiscal_years' must be a list containing exactly one integer fiscal year")

    sft = q.get("source_filing_types")
    if sft != ["10-K"]:
        errors.append(f"[{qid}] 'source_filing_types' must be ['10-K'], got {sft!r}")

    accs = q.get("gold_source_accessions")
    if not isinstance(accs, list) or not accs:
        errors.append(f"[{qid}] 'gold_source_accessions' must be a non-empty list of accessions")
    else:
        for acc in accs:
            if not isinstance(acc, str) or not ACCESSION_PATTERN.match(acc):
                errors.append(f"[{qid}] Invalid accession format: {acc!r}")

    url = q.get("gold_source_url")
    if not isinstance(url, str) or not url.startswith(SEC_URL_PREFIX):
        errors.append(f"[{qid}] Invalid gold_source_url: {url!r}")

    sha = q.get("gold_file_sha256")
    if sha is not None:
        if not isinstance(sha, str) or not SHA256_PATTERN.match(sha):
            errors.append(f"[{qid}] Invalid gold_file_sha256: {sha!r}")
    else:
        errors.append(f"[{qid}] Missing required gold_file_sha256")

    status = q.get("gold_mapping_status")
    if status not in {"mapped", "source_present_chunk_missing"}:
        errors.append(f"[{qid}] Invalid gold_mapping_status: {status!r}")

    hr_status = q.get("human_review_status")
    if hr_status != "human_review_pending":
        errors.append(f"[{qid}] Expected human_review_status 'human_review_pending', got {hr_status!r}")

    chunk_ids = q.get("gold_chunk_ids")
    if not isinstance(chunk_ids, list):
        errors.append(f"[{qid}] 'gold_chunk_ids' must be a list")

    ev_text = q.get("gold_evidence_text")
    if status == "mapped":
        if not chunk_ids:
            errors.append(f"[{qid}] 'gold_chunk_ids' cannot be empty when gold_mapping_status is 'mapped'")
        if not isinstance(ev_text, str) or not ev_text.strip():
            errors.append(f"[{qid}] 'gold_evidence_text' must be a non-empty string when mapped")
    elif status == "source_present_chunk_missing":
        note = q.get("verification_note")
        if not isinstance(note, str) or not note.strip():
            errors.append(f"[{qid}] 'verification_note' must explain the missing chunk when status is 'source_present_chunk_missing'")

    return errors


class ValidationReport(list):
    """Validation report that acts as a list of error strings while exposing report metadata."""

    def __init__(
        self,
        errors: list[str] | None = None,
        structure_validated: bool = True,
        chunk_mapping_validated: bool = True,
        source_verified: bool = False,
        human_review_status: str = "human_review_pending",
        question_count: int = 0,
        records: list[dict[str, Any]] | None = None,
    ):
        super().__init__(errors or [])
        self.structure_validated = structure_validated
        self.chunk_mapping_validated = chunk_mapping_validated
        self.source_verified = source_verified
        self.human_review_status = human_review_status
        self.question_count = question_count
        self.records = records or []

    @property
    def is_valid(self) -> bool:
        return len(self) == 0

    @property
    def errors(self) -> list[str]:
        return list(self)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "errors":
                return list(self)
            if item == "is_valid":
                return self.is_valid
            return getattr(self, item)
        return super().__getitem__(item)


def validate_retrieval_questions(
    questions_path: Path | str,
    chunks_path: Path | str | None = None,
    canonical_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
    strict_benchmark: bool = True,
    verify_sources: bool = False,
) -> tuple[bool, ValidationReport]:
    """Validate a retrieval questions JSONL file against structural, chunk-mapping, and source rules."""
    path = Path(questions_path)
    if not path.is_file():
        rep = ValidationReport(
            errors=[f"Questions file not found: {path}"],
            structure_validated=False,
            chunk_mapping_validated=False,
            source_verified=False,
            human_review_status="unknown",
            question_count=0,
            records=[],
        )
        return False, rep

    errors: list[str] = []
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # 1. Structural parse
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw_json = json.loads(line)
            except Exception as exc:
                errors.append(f"Line {line_no}: Malformed JSON - {exc}")
                continue

            try:
                record = parse_question_record(raw_json, line_no=line_no)
            except ValueError as ve:
                errors.append(str(ve))
                continue

            qid = record.get("id")
            if qid in seen_ids:
                errors.append(f"Line {line_no}: Duplicate question ID {qid!r}")
            seen_ids.add(qid)

            rec_errors = validate_question_record_schema(record)
            errors.extend(rec_errors)
            records.append(record)

    structure_ok = len(errors) == 0

    # Strict benchmark distribution checks (for 25-question benchmark)
    if strict_benchmark:
        if len(records) != 25:
            errors.append(f"Expected exactly 25 questions, got {len(records)}")

        expected_ids = {f"RETR_{i:03d}" for i in range(1, 26)}
        actual_ids = {r.get("id") for r in records}
        if actual_ids != expected_ids:
            missing_ids = expected_ids - actual_ids
            extra_ids = actual_ids - expected_ids
            if missing_ids:
                errors.append(f"Missing expected IDs: {sorted(missing_ids)}")
            if extra_ids:
                errors.append(f"Unexpected IDs: {sorted(extra_ids)}")

        diff_counts = Counter(r.get("difficulty") for r in records)
        if diff_counts.get("easy", 0) != 8:
            errors.append(f"Expected 8 easy questions, got {diff_counts.get('easy', 0)}")
        if diff_counts.get("medium", 0) != 12:
            errors.append(f"Expected 12 medium questions, got {diff_counts.get('medium', 0)}")
        if diff_counts.get("hard", 0) != 5:
            errors.append(f"Expected 5 hard questions, got {diff_counts.get('hard', 0)}")

        topic_counts = Counter(r.get("topic") for r in records)
        if topic_counts.get("Risk Factors", 0) != 10:
            errors.append(f"Expected 10 Risk Factors questions, got {topic_counts.get('Risk Factors', 0)}")
        if topic_counts.get("Business / MD&A", 0) != 10:
            errors.append(f"Expected 10 Business / MD&A questions, got {topic_counts.get('Business / MD&A', 0)}")
        if topic_counts.get("Financial-table or filing-specific evidence", 0) != 5:
            errors.append(f"Expected 5 Financial-table questions, got {topic_counts.get('Financial-table or filing-specific evidence', 0)}")

        sector_counts = Counter(r.get("sector") for r in records)
        for s in VALID_SECTORS:
            if sector_counts.get(s, 0) != 5:
                errors.append(f"Sector '{s}': expected 5 questions, got {sector_counts.get(s, 0)}")

        for s in VALID_SECTORS:
            s_records = [r for r in records if r.get("sector") == s]
            s_diffs = Counter(r.get("difficulty") for r in s_records)
            exp_diffs = EXPECTED_SECTOR_DIFFICULTY[s]
            for d, count in exp_diffs.items():
                if s_diffs.get(d, 0) != count:
                    errors.append(f"Sector '{s}' difficulty '{d}': expected {count}, got {s_diffs.get(d, 0)}")

            s_topics = Counter(r.get("topic") for r in s_records)
            exp_topics = EXPECTED_SECTOR_TOPIC[s]
            for t, count in exp_topics.items():
                if s_topics.get(t, 0) != count:
                    errors.append(f"Sector '{s}' topic '{t}': expected {count}, got {s_topics.get(t, 0)}")

        comp_fy_counts = Counter(
            (r["companies"][0], r["fiscal_years"][0])
            for r in records
            if r.get("companies") and r.get("fiscal_years")
        )
        for (co, fy), count in EXPECTED_COMPANY_ALLOCATIONS.items():
            if comp_fy_counts.get((co, fy), 0) != count:
                errors.append(f"Company {co} ({fy}): expected {count} questions, got {comp_fy_counts.get((co, fy), 0)}")

        fy_counts = Counter(r["fiscal_years"][0] for r in records if r.get("fiscal_years"))
        for fy, count in EXPECTED_FY_COUNTS.items():
            if fy_counts.get(fy, 0) != count:
                errors.append(f"Fiscal year {fy}: expected {count} questions, got {fy_counts.get(fy, 0)}")

    # 2. Chunk mapping validation
    chunk_mapping_ok = True
    chunks_by_id = {}
    if chunks_path and Path(chunks_path).is_file():
        try:
            loaded_chunks = load_chunks(chunks_path)
            chunks_by_id = {c["chunk_id"]: c for c in loaded_chunks}
            for r in records:
                qid = r.get("id")
                co = r["companies"][0]
                fy = r["fiscal_years"][0]
                accs = r.get("gold_source_accessions", [])
                acc = accs[0] if accs else ""
                url = r.get("gold_source_url")
                sha = r.get("gold_file_sha256")
                ev = r.get("gold_evidence_text")
                status = r.get("gold_mapping_status")
                cids = r.get("gold_chunk_ids", [])

                if status == "mapped":
                    for cid in cids:
                        if cid not in chunks_by_id:
                            errors.append(f"[{qid}] Gold chunk ID '{cid}' not found in chunks file")
                            chunk_mapping_ok = False
                        else:
                            c = chunks_by_id[cid]
                            if c["ticker"] != co or c["fiscal_year"] != fy:
                                errors.append(f"[{qid}] Gold chunk '{cid}' has company/year ({c['ticker']}, {c['fiscal_year']}) != question ({co}, {fy})")
                                chunk_mapping_ok = False
                            if c["accession"] != acc:
                                errors.append(f"[{qid}] Gold chunk '{cid}' accession ({c['accession']}) != question accession ({acc})")
                                chunk_mapping_ok = False
                            if c.get("form") != "10-K":
                                errors.append(f"[{qid}] Gold chunk '{cid}' form ({c.get('form')}) != '10-K'")
                                chunk_mapping_ok = False
                            if url and c.get("source_url") != url:
                                errors.append(f"[{qid}] Gold chunk '{cid}' source_url ({c.get('source_url')}) != question source_url ({url})")
                                chunk_mapping_ok = False
                            if sha and c.get("file_sha256") != sha:
                                errors.append(f"[{qid}] Gold chunk '{cid}' file_sha256 ({c.get('file_sha256')}) != question file_sha256 ({sha})")
                                chunk_mapping_ok = False
                            if ev and ev not in c["text"]:
                                errors.append(f"[{qid}] Gold evidence text not found in chunk '{cid}' text")
                                chunk_mapping_ok = False
        except Exception as exc:
            errors.append(f"Error validating against chunks file {chunks_path}: {exc}")
            chunk_mapping_ok = False

    # Canonical text check
    if canonical_dir and Path(canonical_dir).is_dir():
        for r in records:
            qid = r.get("id")
            ev = r.get("gold_evidence_text")
            accs = r.get("gold_source_accessions", [])
            cids = r.get("gold_chunk_ids", [])
            status = r.get("gold_mapping_status")
            if status == "mapped" and ev and accs:
                acc = accs[0]
                can_file = Path(canonical_dir) / f"{acc}.txt"
                if can_file.is_file():
                    with open(can_file, "r", encoding="utf-8") as f:
                        can_txt = f.read()
                    if ev not in can_txt:
                        errors.append(f"[{qid}] Gold evidence text not found in canonical text {can_file.name}")
                        chunk_mapping_ok = False
                    for cid in cids:
                        if cid in chunks_by_id:
                            c = chunks_by_id[cid]
                            span_txt = can_txt[c["char_start"]:c["char_end"]]
                            if span_txt != c["text"]:
                                errors.append(f"[{qid}] Canonical character offset span mismatch for chunk '{cid}'")
                                chunk_mapping_ok = False
                elif verify_sources:
                    errors.append(f"[{qid}] Canonical text file missing in strict source mode: {can_file}")
                    chunk_mapping_ok = False

    # 3. Source verification (raw SEC HTML & manifest provenance)
    source_ok = True
    if verify_sources:
        if not raw_dir or not Path(raw_dir).is_dir():
            errors.append(f"Raw filings directory missing in strict source mode: {raw_dir}")
            source_ok = False
        if not manifest_path or not Path(manifest_path).is_file():
            errors.append(f"Filing manifest file missing in strict source mode: {manifest_path}")
            source_ok = False

        manifest_map = {}
        if manifest_path and Path(manifest_path).is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                manifest_map = {
                    (m["ticker"], m["target_fiscal_year"]): m
                    for m in manifest_data
                }
            except Exception as exc:
                errors.append(f"Error reading manifest {manifest_path}: {exc}")
                source_ok = False

        if raw_dir and Path(raw_dir).is_dir() and manifest_map:
            for r in records:
                qid = r.get("id")
                co = r["companies"][0]
                fy = r["fiscal_years"][0]
                accs = r.get("gold_source_accessions", [])
                acc = accs[0] if accs else ""
                expected_sha = r.get("gold_file_sha256")

                m = manifest_map.get((co, fy))
                if not m:
                    errors.append(f"[{qid}] Provenance mismatch: manifest has no record for {co} ({fy})")
                    source_ok = False
                else:
                    if m.get("accession_number") != acc:
                        errors.append(f"[{qid}] Manifest accession ({m.get('accession_number')}) != question accession ({acc})")
                        source_ok = False
                    if m.get("source_url") != r.get("gold_source_url"):
                        errors.append(f"[{qid}] Manifest source_url != question gold_source_url")
                        source_ok = False

                raw_candidate = Path(raw_dir) / f"{acc}.html"
                if not raw_candidate.is_file():
                    raw_candidate = Path(raw_dir) / f"{acc}.htm"

                if not raw_candidate.is_file():
                    errors.append(f"[{qid}] Original SEC HTML file not found for accession {acc}")
                    source_ok = False
                else:
                    actual_sha = compute_file_sha256(raw_candidate)
                    if actual_sha != expected_sha:
                        errors.append(f"[{qid}] Original HTML SHA-256 ({actual_sha}) != gold record SHA-256 ({expected_sha})")
                        source_ok = False
    else:
        # When not running strict source verification, source_ok reflects whether files exist if provided
        source_ok = False

    is_valid = len(errors) == 0
    report = ValidationReport(
        errors=errors,
        structure_validated=structure_ok,
        chunk_mapping_validated=chunk_mapping_ok and (bool(chunks_by_id) or not chunks_path),
        source_verified=source_ok,
        human_review_status="human_review_pending",
        question_count=len(records),
        records=records,
    )
    return is_valid, report


def calculate_metrics(eval_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate Recall@k, MRR, and breakdown metrics from evaluation results.

    All questions are kept in the denominator. Misses and errors contribute 0.0.
    """
    total = len(eval_results)
    if total == 0:
        return {
            "total_questions": 0,
            "hits": 0,
            "misses": 0,
            "recall": 0.0,
            "mrr": 0.0,
            "by_difficulty": {},
            "by_topic": {},
            "by_sector": {},
        }

    hits = sum(1 for r in eval_results if r.get("hit"))
    misses = total - hits
    recall = hits / total
    mrr = sum(r.get("reciprocal_rank", 0.0) for r in eval_results) / total

    def compute_subset_metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        sub_total = len(subset)
        if sub_total == 0:
            return {"total": 0, "hits": 0, "recall": 0.0, "mrr": 0.0}
        sub_hits = sum(1 for r in subset if r.get("hit"))
        sub_rec = sub_hits / sub_total
        sub_mrr = sum(r.get("reciprocal_rank", 0.0) for r in subset) / sub_total
        return {"total": sub_total, "hits": sub_hits, "recall": sub_rec, "mrr": sub_mrr}

    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in eval_results if r.get("difficulty") == diff]
        by_difficulty[diff] = compute_subset_metrics(sub)

    by_topic = {}
    for topic in ["Risk Factors", "Business / MD&A", "Financial-table or filing-specific evidence"]:
        sub = [r for r in eval_results if r.get("topic") == topic]
        by_topic[topic] = compute_subset_metrics(sub)

    by_sector = {}
    for sector in ["Technology", "Financial services", "Healthcare", "Energy and utilities", "Consumer"]:
        sub = [r for r in eval_results if r.get("sector") == sector]
        by_sector[sector] = compute_subset_metrics(sub)

    return {
        "total_questions": total,
        "hits": hits,
        "misses": misses,
        "recall": recall,
        "mrr": mrr,
        "by_difficulty": by_difficulty,
        "by_topic": by_topic,
        "by_sector": by_sector,
    }


def generate_public_results(eval_output: dict[str, Any], output_path: Path | str | None = None) -> dict[str, Any]:
    """Generate the public results.json summary containing only current results.

    Contains no historical runs, no score comparisons, and no public version labels.
    """
    metrics = eval_output["metrics"]
    missed_records = []
    for r in eval_output.get("results", []):
        if not r.get("hit"):
            missed_records.append({
                "id": r["id"],
                "ticker": r["ticker"],
                "fiscal_year": r["fiscal_year"],
                "difficulty": r["difficulty"],
                "topic": r["topic"],
                "technical_reason": r.get("miss_reason", "Ranked outside top-k cutoff"),
            })

    public_summary = {
        "evaluation_type": "retrieval",
        "evaluation_status": "provisional_human_review_pending",
        "evaluation_question_count": metrics["total_questions"],
        "sec_filing_count": eval_output.get("sec_filing_count", 10),
        "indexed_chunk_count": eval_output.get("total_chunks", 0),
        "question_file_sha256": eval_output["question_file_sha256"],
        "corpus_file_sha256": eval_output["chunks_file_sha256"],
        "top_k": eval_output.get("top_k", 5),
        "recall_at_top_k": round(metrics["recall"], 4),
        "mrr": round(metrics["mrr"], 4),
        "hits": metrics["hits"],
        "misses": metrics["misses"],
        "results_by_difficulty": {
            d: {
                "total": m["total"],
                "hits": m["hits"],
                "recall": round(m["recall"], 4),
                "mrr": round(m["mrr"], 4),
            }
            for d, m in metrics["by_difficulty"].items()
        },
        "results_by_topic": {
            t: {
                "total": m["total"],
                "hits": m["hits"],
                "recall": round(m["recall"], 4),
                "mrr": round(m["mrr"], 4),
            }
            for t, m in metrics["by_topic"].items()
        },
        "results_by_sector": {
            s: {
                "total": m["total"],
                "hits": m["hits"],
                "recall": round(m["recall"], 4),
                "mrr": round(m["mrr"], 4),
            }
            for s, m in metrics["by_sector"].items()
        },
        "index_build_time_sec": eval_output.get("index_build_time_sec", 0.0),
        "avg_search_time_sec": eval_output.get("avg_search_time_sec", 0.0),
        "missing_questions": missed_records,
        "notes": {
            "scope": "Evaluates BM25 evidence retrieval over SEC Form 10-K filings; does not measure LLM generation, arithmetic correctness, or multi-hop reasoning.",
            "human_review": "Candidate gold labels verified by author; independent human review remains pending.",
        },
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(public_summary, f, indent=2)

    return public_summary


def evaluate_retrieval(
    questions_path: Path | str,
    chunks_path: Path | str | None = None,
    canonical_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
    top_k: int = 5,
    output_path: Path | str | None = None,
    publish_results_path: Path | str | None = None,
    strict_benchmark: bool = True,
    verify_sources: bool = False,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Execute BM25 retrieval evaluation against the questions dataset with mandatory pre-scoring validation."""
    q_path = Path(questions_path)
    if not q_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    # 1. Mandatory validation before scoring
    if not skip_validation:
        is_valid, report = validate_retrieval_questions(
            questions_path=q_path,
            chunks_path=chunks_path,
            canonical_dir=canonical_dir,
            raw_dir=raw_dir,
            manifest_path=manifest_path,
            strict_benchmark=strict_benchmark,
            verify_sources=verify_sources,
        )
        if not is_valid:
            err_msg = "; ".join(report["errors"][:5])
            raise ValueError(f"Pre-scoring validation failed with {len(report['errors'])} error(s): {err_msg}")

    # 2. Record SHA-256 before evaluation
    sha_before = compute_file_sha256(q_path)

    # 3. Load questions
    questions = []
    with open(q_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    # 4. Load chunks and build BM25 index
    resolved_chunks_path = Path(chunks_path) if chunks_path else Path("data/processed/chunks.jsonl")
    if not resolved_chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {resolved_chunks_path}")

    raw_chunks = load_chunks(resolved_chunks_path)
    t0_idx = time.perf_counter()
    index = build_index(raw_chunks)
    t1_idx = time.perf_counter()
    index_build_time = t1_idx - t0_idx

    search_times: list[float] = []
    eval_results: list[dict[str, Any]] = []
    wrong_company_year_count = 0
    sec_filings_seen = set()

    for q in questions:
        qid = q["id"]
        ticker = q["companies"][0]
        fy = q["fiscal_years"][0]
        sec_filings_seen.add((ticker, fy))
        query_text = q["question"]

        gold_set = set(q.get("gold_chunk_ids") or [])
        status = q.get("gold_mapping_status")

        t0_s = time.perf_counter()
        try:
            # STRICT PRIVACY: pass ONLY query_text, ticker, fy to search_evidence.
            # Never pass answer keys, evidence text, or notes.
            search_res = search_evidence(
                query=query_text,
                tickers=[ticker],
                fiscal_years=[fy],
                index=index,
                top_k=top_k,
            )
            err = None
        except Exception as exc:
            search_res = {"status": "error", "results": []}
            err = str(exc)
        t1_s = time.perf_counter()
        search_times.append(t1_s - t0_s)

        retrieved_chunks = search_res.get("results", [])
        retrieved_ids = [r["chunk_id"] for r in retrieved_chunks]
        retrieved_accs = [r["accession"] for r in retrieved_chunks]

        for r in retrieved_chunks:
            if r["ticker"] != ticker or r["fiscal_year"] != fy:
                wrong_company_year_count += 1

        hit = False
        first_rank = None
        reciprocal_rank = 0.0
        miss_reason = None

        if status == "mapped" and gold_set:
            for rank_idx, cid in enumerate(retrieved_ids, start=1):
                if cid in gold_set:
                    hit = True
                    first_rank = rank_idx
                    reciprocal_rank = 1.0 / rank_idx
                    break
            if not hit:
                miss_reason = "Relevant passage present in corpus but ranked below top-k cutoff"
        else:
            miss_reason = "Source passage present in SEC filing but omitted from extracted corpus"

        eval_results.append({
            "id": qid,
            "ticker": ticker,
            "fiscal_year": fy,
            "topic": q.get("topic"),
            "difficulty": q.get("difficulty"),
            "sector": q.get("sector"),
            "hit": hit,
            "rank": first_rank,
            "reciprocal_rank": reciprocal_rank,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_accessions": retrieved_accs,
            "gold_chunk_ids": list(gold_set),
            "gold_mapping_status": status,
            "miss_reason": miss_reason if not hit else None,
            "error": err,
        })

    # 5. Record SHA-256 after evaluation and verify freeze
    sha_after = compute_file_sha256(q_path)
    if sha_before != sha_after:
        raise RuntimeError(f"FATAL: Questions file SHA-256 changed during evaluation! ({sha_before} != {sha_after})")

    metrics = calculate_metrics(eval_results)
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0.0

    full_output = {
        "question_file_sha256": sha_before,
        "chunks_file_sha256": compute_file_sha256(resolved_chunks_path),
        "total_chunks": len(raw_chunks),
        "sec_filing_count": len(sec_filings_seen),
        "vocab_size": index.get("vocab_size", 0),
        "top_k": top_k,
        "index_build_time_sec": round(index_build_time, 4),
        "avg_search_time_sec": round(avg_search_time, 4),
        "wrong_company_year_count": wrong_company_year_count,
        "metrics": metrics,
        "results": eval_results,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2)

    if publish_results_path:
        generate_public_results(full_output, output_path=publish_results_path)

    return full_output


def run_cli_validate(args: argparse.Namespace) -> int:
    """CLI handler for validate-retrieval."""
    questions_path = args.questions or "evaluation/questions.jsonl"
    chunks_path = args.chunks_path
    if not chunks_path and os.path.isfile("data/processed/chunks.jsonl"):
        chunks_path = "data/processed/chunks.jsonl"
    canonical_dir = args.canonical_dir
    if not canonical_dir and os.path.isdir("data/processed/filing_text"):
        canonical_dir = "data/processed/filing_text"
    raw_dir = args.raw_dir
    if not raw_dir and os.path.isdir("data/raw/filings"):
        raw_dir = "data/raw/filings"
    manifest_path = args.manifest_path
    if not manifest_path and os.path.isfile("data/processed/filing_manifest.json"):
        manifest_path = "data/processed/filing_manifest.json"

    print(f"Validating retrieval questions from: {questions_path}")
    if chunks_path:
        print(f"Using chunks file: {chunks_path}")
    if canonical_dir:
        print(f"Using canonical texts dir: {canonical_dir}")
    if args.verify_sources:
        print("Strict source verification enabled (validating raw HTML and filing manifest)")

    is_valid, report = validate_retrieval_questions(
        questions_path=questions_path,
        chunks_path=chunks_path,
        canonical_dir=canonical_dir,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        strict_benchmark=not args.no_strict,
        verify_sources=args.verify_sources,
    )

    print("\nValidation Status Breakdown:")
    print(f"  Structure validated       : {'PASS' if report['structure_validated'] else 'FAIL'}")
    print(f"  Chunk mapping validated   : {'PASS' if report['chunk_mapping_validated'] else 'FAIL'}")
    print(f"  Original source verified  : {'PASS' if report['source_verified'] else ('SKIPPED' if not args.verify_sources else 'FAIL')}")
    print(f"  Human review status       : {report['human_review_status']}")

    if not is_valid:
        print(f"\n[FAIL] Validation failed with {len(report['errors'])} error(s):")
        for err in report["errors"]:
            print(f"  - {err}")
        return 1

    sha = compute_file_sha256(questions_path)
    print(f"\n[PASS] All validation checks passed successfully!")
    print(f"Question file SHA-256: {sha}")
    return 0


def run_cli_retrieval(args: argparse.Namespace) -> int:
    """CLI handler for retrieval evaluation."""
    questions_path = args.questions or "evaluation/questions.jsonl"
    chunks_path = args.chunks_path or "data/processed/chunks.jsonl"
    canonical_dir = args.canonical_dir
    if not canonical_dir and os.path.isdir("data/processed/filing_text"):
        canonical_dir = "data/processed/filing_text"
    raw_dir = args.raw_dir
    if not raw_dir and os.path.isdir("data/raw/filings"):
        raw_dir = "data/raw/filings"
    manifest_path = args.manifest_path
    if not manifest_path and os.path.isfile("data/processed/filing_manifest.json"):
        manifest_path = "data/processed/filing_manifest.json"

    top_k = args.top_k or 5
    output_path = args.output
    publish_path = args.publish_results or "evaluation/results.json"

    print(f"Executing retrieval evaluation:")
    print(f"  Questions: {questions_path}")
    print(f"  Chunks: {chunks_path}")
    print(f"  Top-k: {top_k}")
    if publish_path:
        print(f"  Publish results: {publish_path}")

    res = evaluate_retrieval(
        questions_path=questions_path,
        chunks_path=chunks_path,
        canonical_dir=canonical_dir,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        top_k=top_k,
        output_path=output_path,
        publish_results_path=publish_path,
        strict_benchmark=not args.no_strict,
        verify_sources=args.verify_sources,
        skip_validation=args.skip_validation,
    )

    metrics = res["metrics"]
    print(f"\n========================================================")
    print(f"RETRIEVAL EVALUATION RESULTS (Top-{top_k})")
    print(f"========================================================")
    print(f"Question file SHA-256: {res['question_file_sha256']}")
    print(f"Corpus file SHA-256: {res['chunks_file_sha256']}")
    print(f"Total chunks indexed: {res['total_chunks']}")
    print(f"Total filings: {res['sec_filing_count']}")
    print(f"Vocabulary size: {res['vocab_size']}")
    print(f"Index build time: {res['index_build_time_sec']:.4f} s")
    print(f"Average search time: {res['avg_search_time_sec']:.4f} s")
    print(f"Cross-company/year errors: {res['wrong_company_year_count']}")
    print(f"--------------------------------------------------------")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Hits: {metrics['hits']} / {metrics['total_questions']}")
    print(f"Recall@{top_k}: {metrics['recall']:.2%}")
    print(f"Mean Reciprocal Rank (MRR): {metrics['mrr']:.4f}")
    print(f"--------------------------------------------------------")

    print("\nDetailed Question Results:")
    print(f"{'ID':<10} {'Ticker':<6} {'FY':<6} {'Topic':<35} {'Diff':<8} {'Rank':<6} {'Status':<6}")
    print("-" * 77)
    for r in res["results"]:
        rank_str = str(r["rank"]) if r["rank"] is not None else "miss"
        status_str = "HIT" if r["hit"] else "MISS"
        print(f"{r['id']:<10} {r['ticker']:<6} {r['fiscal_year']:<6} {r['topic'][:33]:<35} {r['difficulty']:<8} {rank_str:<6} {status_str:<6}")

    print("\nMetrics by Difficulty:")
    for diff, m in metrics["by_difficulty"].items():
        print(f"  {diff.capitalize():<8}: Total={m['total']}, Hits={m['hits']}, Recall@{top_k}={m['recall']:.2%}, MRR={m['mrr']:.4f}")

    print("\nMetrics by Topic:")
    for topic, m in metrics["by_topic"].items():
        print(f"  {topic:<40}: Total={m['total']}, Hits={m['hits']}, Recall@{top_k}={m['recall']:.2%}, MRR={m['mrr']:.4f}")

    print("\nMetrics by Sector:")
    for sector, m in metrics["by_sector"].items():
        print(f"  {sector:<25}: Total={m['total']}, Hits={m['hits']}, Recall@{top_k}={m['recall']:.2%}, MRR={m['mrr']:.4f}")

    missed = [r for r in res["results"] if not r["hit"]]
    if missed:
        print(f"\nMissed Questions ({len(missed)}):")
        for m in missed:
            print(f"  - {m['id']} ({m['ticker']} {m['fiscal_year']}, {m['difficulty']}, {m['topic']})")
    else:
        print("\nNo missed questions!")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Fyntrace Retrieval Evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: validate-retrieval
    val_parser = subparsers.add_parser("validate-retrieval", help="Validate retrieval benchmark dataset")
    val_parser.add_argument("--questions", default="evaluation/questions.jsonl", help="Path to questions JSONL")
    val_parser.add_argument("--chunks-path", default=None, help="Path to chunks JSONL")
    val_parser.add_argument("--canonical-dir", default=None, help="Path to canonical text dir")
    val_parser.add_argument("--raw-dir", default=None, help="Path to raw HTML filings dir")
    val_parser.add_argument("--manifest-path", default=None, help="Path to filing manifest JSON")
    val_parser.add_argument("--verify-sources", action="store_true", help="Enable strict source verification")
    val_parser.add_argument("--no-strict", action="store_true", help="Disable strict 25-count benchmark rules")

    # Subcommand: retrieval
    eval_parser = subparsers.add_parser("retrieval", help="Evaluate BM25 retrieval against benchmark")
    eval_parser.add_argument("--questions", default="evaluation/questions.jsonl", help="Path to questions JSONL")
    eval_parser.add_argument("--chunks-path", default="data/processed/chunks.jsonl", help="Path to chunks JSONL")
    eval_parser.add_argument("--canonical-dir", default=None, help="Path to canonical text dir")
    eval_parser.add_argument("--raw-dir", default=None, help="Path to raw HTML filings dir")
    eval_parser.add_argument("--manifest-path", default=None, help="Path to filing manifest JSON")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Top-K cutoff (default: 5)")
    eval_parser.add_argument("--output", default=None, help="Optional output JSON path for detailed results")
    eval_parser.add_argument("--publish-results", default="evaluation/results.json", help="Path to write public results.json")
    eval_parser.add_argument("--verify-sources", action="store_true", help="Enable strict source verification during pre-scoring validation")
    eval_parser.add_argument("--skip-validation", action="store_true", help="Skip pre-scoring validation")
    eval_parser.add_argument("--no-strict", action="store_true", help="Disable strict 25-count benchmark rules")

    args = parser.parse_args(argv)
    if args.command == "validate-retrieval":
        return run_cli_validate(args)
    elif args.command == "retrieval":
        return run_cli_retrieval(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
