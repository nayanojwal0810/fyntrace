"""Source-aware BM25 evidence retrieval over SEC filing chunks."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"

# Fixed list of common English stopwords; keeps financial terms (revenue, liabilities, credit, risk, margin, interest)
DEFAULT_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
})


def tokenize(text: str, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> list[str]:
    """Deterministic, case-insensitive tokenizer for filing chunks and queries.

    Preserves alphanumeric characters and numbers, handles hyphenated words,
    and strips punctuation and stopwords.
    """
    if not text or not isinstance(text, str):
        return []

    # Find words with alphanumeric characters and internal hyphens
    raw_tokens = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text.lower())
    tokens: list[str] = []
    for tok in raw_tokens:
        if "-" in tok:
            if tok not in stopwords:
                tokens.append(tok)
            # Also index constituent sub-words of hyphenated compounds
            for part in tok.split("-"):
                if part and part not in stopwords:
                    tokens.append(part)
        else:
            if tok not in stopwords:
                tokens.append(tok)
    return tokens


def load_chunks(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load and strictly validate SEC filing chunks from JSONL.

    Rejects malformed JSON, duplicate chunk IDs, missing/empty text or metadata,
    invalid character offsets, invalid SHA-256 hashes, and non-SEC hostnames.
    """
    chunks_path = Path(path) if path is not None else DEFAULT_CHUNKS_PATH
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")

    chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Malformed JSON on line {line_no} in {chunks_path}: {exc}") from exc

            if not isinstance(chunk, dict):
                raise ValueError(f"Invalid record on line {line_no} in {chunks_path}: expected JSON object.")

            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"Missing or invalid 'chunk_id' on line {line_no} in {chunks_path}.")

            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk ID '{chunk_id}' on line {line_no} in {chunks_path}.")
            seen_chunk_ids.add(chunk_id)

            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk '{chunk_id}' has empty or invalid 'text'.")

            ticker = chunk.get("ticker")
            if not isinstance(ticker, str) or not ticker.strip():
                raise ValueError(f"Chunk '{chunk_id}' has missing or invalid 'ticker'.")

            fy = chunk.get("fiscal_year")
            if not isinstance(fy, int) or isinstance(fy, bool):
                raise ValueError(f"Chunk '{chunk_id}' has missing or invalid 'fiscal_year'.")

            cik = chunk.get("cik")
            if not isinstance(cik, str) or not cik.strip():
                raise ValueError(f"Chunk '{chunk_id}' has missing or invalid 'cik'.")

            accession = chunk.get("accession")
            if not isinstance(accession, str) or not accession.strip():
                raise ValueError(f"Chunk '{chunk_id}' has missing or invalid 'accession'.")

            source_url = chunk.get("source_url")
            if not isinstance(source_url, str) or not source_url.strip():
                raise ValueError(f"Chunk '{chunk_id}' has missing 'source_url'.")

            parsed_url = urlparse(source_url)
            if parsed_url.scheme.lower() not in ("http", "https") or parsed_url.netloc.lower() not in ("www.sec.gov", "sec.gov"):
                raise ValueError(f"Chunk '{chunk_id}' source_url must be an official SEC URL on www.sec.gov or sec.gov, got {source_url!r}")

            char_start = chunk.get("char_start")
            char_end = chunk.get("char_end")
            if not isinstance(char_start, int) or isinstance(char_start, bool) or char_start < 0:
                raise ValueError(f"Chunk '{chunk_id}' has negative or invalid 'char_start': {char_start!r}")

            if not isinstance(char_end, int) or isinstance(char_end, bool) or char_end < char_start:
                raise ValueError(f"Chunk '{chunk_id}' has reversed or invalid 'char_end': {char_end!r} < {char_start!r}")

            if (char_end - char_start) != len(text):
                raise ValueError(
                    f"Chunk '{chunk_id}' character offset span ({char_end - char_start}) "
                    f"does not equal text length ({len(text)})."
                )

            sha = chunk.get("file_sha256")
            if (
                not isinstance(sha, str)
                or len(sha) != 64
                or not all(c in "0123456789abcdefABCDEF" for c in sha)
            ):
                raise ValueError(f"Chunk '{chunk_id}' has invalid 'file_sha256': {sha!r}")

            confidence = chunk.get("section_confidence")
            if confidence not in ("high", "low", "unknown"):
                raise ValueError(f"Chunk '{chunk_id}' has invalid 'section_confidence': {confidence!r}")

            chunks.append(chunk)

    return chunks


def build_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build in-memory BM25 index with inverted index and frequency tables.

    Indexing is deterministic. Chunks are stored by chunk ID and grouped
    by (ticker, fiscal_year) for strict subset filtering.
    """
    chunks_by_id: dict[str, dict[str, Any]] = {}
    doc_lens: dict[str, int] = {}
    term_freqs: dict[str, dict[str, int]] = {}
    inverted_index: dict[str, list[str]] = {}
    company_year_chunks: dict[tuple[str, int], list[str]] = {}

    # Sort chunks by chunk_id for deterministic indexing
    sorted_chunks = sorted(chunks, key=lambda c: c["chunk_id"])

    for chunk in sorted_chunks:
        cid = chunk["chunk_id"]
        chunks_by_id[cid] = chunk

        tokens = tokenize(chunk["text"])
        doc_lens[cid] = len(tokens)

        tf = dict(Counter(tokens))
        term_freqs[cid] = tf

        ticker_norm = chunk["ticker"].strip().upper()
        fy = chunk["fiscal_year"]
        pair = (ticker_norm, fy)
        company_year_chunks.setdefault(pair, []).append(cid)

        for term in tf:
            inverted_index.setdefault(term, []).append(cid)

    return {
        "chunks_by_id": chunks_by_id,
        "doc_lens": doc_lens,
        "term_freqs": term_freqs,
        "inverted_index": inverted_index,
        "company_year_chunks": company_year_chunks,
        "total_chunks": len(sorted_chunks),
        "vocab_size": len(inverted_index),
    }


def search_evidence(
    query: str,
    tickers: list[str],
    fiscal_years: list[int],
    index: dict[str, Any],
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, Any]:
    """Search filing evidence using BM25 with strict company and fiscal-year filtering.

    Calculates BM25 using statistics derived strictly from the eligible subset,
    preventing distortion from unrelated companies or fiscal years.
    """
    if not tickers or not isinstance(tickers, (list, tuple, set)):
        raise ValueError("tickers must be a non-empty list of ticker strings.")
    norm_tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    if not norm_tickers:
        raise ValueError("tickers must contain at least one non-empty string.")

    if not fiscal_years or not isinstance(fiscal_years, (list, tuple, set)):
        raise ValueError("fiscal_years must be a non-empty list of integers.")
    norm_years = [y for y in fiscal_years if isinstance(y, int) and not isinstance(y, bool)]
    if not norm_years:
        raise ValueError("fiscal_years must contain at least one valid integer.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > 20:
        raise ValueError(f"top_k must be an integer between 1 and 20, got {top_k!r}")

    filters_dict = {
        "tickers": norm_tickers,
        "fiscal_years": norm_years,
    }

    if not isinstance(query, str) or not query.strip():
        return {
            "status": "no_results",
            "query": query if isinstance(query, str) else "",
            "filters": filters_dict,
            "results": [],
            "message": "Query is empty.",
        }

    q_terms = tokenize(query)
    if not q_terms:
        return {
            "status": "no_results",
            "query": query,
            "filters": filters_dict,
            "results": [],
            "message": "Query produced no searchable terms after tokenization.",
        }

    # 1. Filter eligible chunk IDs before scoring
    target_pairs = {(t, y) for t in norm_tickers for y in norm_years}
    eligible_chunk_ids: list[str] = []
    for pair in sorted(target_pairs):
        eligible_chunk_ids.extend(index.get("company_year_chunks", {}).get(pair, []))

    if not eligible_chunk_ids:
        return {
            "status": "no_results",
            "query": query,
            "filters": filters_dict,
            "results": [],
            "message": f"No indexed filings found for requested tickers/years: {norm_tickers} {norm_years}",
        }

    # 2. Derive BM25 corpus statistics from the eligible subset only
    N = len(eligible_chunk_ids)
    eligible_set = set(eligible_chunk_ids)
    total_len = sum(index["doc_lens"][cid] for cid in eligible_chunk_ids)
    avg_doc_len = total_len / N if N > 0 else 1.0
    if avg_doc_len == 0:
        avg_doc_len = 1.0

    # 3. Calculate BM25 scores using inverted index on eligible candidates
    scores: dict[str, float] = {}
    q_counts = Counter(q_terms)

    for term, qtf in q_counts.items():
        doc_list = index["inverted_index"].get(term, ())
        eligible_docs = [cid for cid in doc_list if cid in eligible_set]
        df = len(eligible_docs)
        if df == 0:
            continue

        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

        for cid in eligible_docs:
            tf = index["term_freqs"][cid][term]
            doc_len = index["doc_lens"][cid]
            den = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            term_score = idf * (tf * (k1 + 1.0) / den) * qtf
            scores[cid] = scores.get(cid, 0.0) + term_score

    if not scores:
        return {
            "status": "no_results",
            "query": query,
            "filters": filters_dict,
            "results": [],
            "message": "No matching evidence found for query in requested filings.",
        }

    # 4. Assemble and rank results descending by score, deterministic tie-breaking on chunk_id
    results_list: list[dict[str, Any]] = []
    for cid, score in scores.items():
        chunk = index["chunks_by_id"][cid]
        results_list.append({
            "chunk_id": chunk["chunk_id"],
            "ticker": chunk["ticker"],
            "company_name": chunk["company_name"],
            "cik": chunk["cik"],
            "fiscal_year": chunk["fiscal_year"],
            "accession": chunk["accession"],
            "section": chunk["section"],
            "section_confidence": chunk["section_confidence"],
            "text": chunk["text"],
            "score": round(score, 4),
            "source_url": chunk["source_url"],
            "char_start": chunk["char_start"],
            "char_end": chunk["char_end"],
            "file_sha256": chunk["file_sha256"],
        })

    results_list.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    top_results = results_list[:top_k]

    return {
        "status": "success",
        "query": query,
        "filters": filters_dict,
        "results": top_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evidence retrieval."""
    parser = argparse.ArgumentParser(
        description="Source-aware BM25 evidence retrieval over SEC filing chunks."
    )
    parser.add_argument("--query", required=True, help="Search query string.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Company tickers (e.g. MSFT JPM).")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Fiscal years (e.g. 2024).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (1-20, default 5).")
    parser.add_argument("--chunks-path", type=str, default=None, help="Optional path to chunks.jsonl.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for SEC filing retrieval CLI."""
    args = parse_args(argv)

    chunks = load_chunks(args.chunks_path)
    index = build_index(chunks)

    search_result = search_evidence(
        query=args.query,
        tickers=args.tickers,
        fiscal_years=args.years,
        index=index,
        top_k=args.top_k,
    )

    print(f"Query: \"{search_result['query']}\"")
    print(f"Filters: tickers={search_result['filters']['tickers']}, fiscal_years={search_result['filters']['fiscal_years']}")
    print(f"Status: {search_result['status']}")
    print(f"Results returned: {len(search_result['results'])}")

    if search_result["status"] == "no_results":
        print(f"Message: {search_result.get('message', 'No results.')}")
        return

    print("\nEvidence:")
    for idx, item in enumerate(search_result["results"], start=1):
        excerpt = item["text"][:200].replace("\n", " ").strip()
        if len(item["text"]) > 200:
            excerpt += "..."
        print(f"\n{idx}. Score: {item['score']} | Ticker: {item['ticker']} | FY: {item['fiscal_year']}")
        print(f"   Chunk ID: {item['chunk_id']}")
        print(f"   Section: {item['section']} (confidence: {item['section_confidence']})")
        print(f"   URL: {item['source_url']}")
        print(f"   Excerpt: \"{excerpt}\"")


if __name__ == "__main__":
    main()
