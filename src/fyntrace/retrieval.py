"""BM25 retrieval engine using SQLite FTS5 for SEC filing chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file in blocks."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def tokenize_query(query: str) -> list[str]:
    """Extract alphanumeric tokens from query string, lowercased.

    Special characters, punctuation, and symbols are treated as token separators.
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query).__name__}")
    return re.findall(r"[A-Za-z0-9]+", query.lower())


def build_match_expression(tokens: list[str]) -> str:
    """Build a safe FTS5 OR match expression from tokens.

    Each token is wrapped in double quotes to prevent FTS5 syntax errors
    and operator collision (e.g. AND, OR, NOT, punctuation).
    Deduplicating tokens preserves query term fairness.
    """
    unique_tokens = list(dict.fromkeys(tokens))
    if not unique_tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in unique_tokens)


def init_schema(con: sqlite3.Connection) -> None:
    """Initialize SQLite tables and FTS5 virtual table."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            ticker TEXT NOT NULL,
            cik TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            form TEXT NOT NULL,
            accession TEXT NOT NULL,
            section TEXT NOT NULL,
            section_confidence TEXT,
            chunk_type TEXT NOT NULL,
            text TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            source_file_sha256 TEXT NOT NULL,
            canonical_text_path TEXT NOT NULL,
            filing_date TEXT,
            period_end_date TEXT,
            table_title TEXT
        );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ticker ON chunks(ticker);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(fiscal_year);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ticker_year ON chunks(ticker, fiscal_year);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_accession ON chunks(accession);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type);")
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id'
        );
    """)


def _validate_year(val: Any, param_name: str) -> int:
    """Validate and parse a fiscal year parameter."""
    if isinstance(val, bool):
        raise ValueError(f"{param_name} must be an integer, got boolean {val!r}")
    try:
        y = int(val)
    except (ValueError, TypeError):
        raise ValueError(f"{param_name} must be an integer, got {val!r}")
    if y <= 0:
        raise ValueError(f"{param_name} must be a positive integer, got {y}")
    return y


def build_index(
    chunks_path: Path | str | None = None,
    db_path: Path | str | None = None,
    batch_size: int = 5000,
) -> dict[str, Any]:
    """Build persistent SQLite FTS5 BM25 index from chunks.jsonl safely.

    Uses a temporary SQLite database, verifies chunk counts, closes connections,
    and atomically replaces any existing database file on completion.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    if chunks_path is None:
        chunks_file = Path("data/processed/chunks.jsonl")
    else:
        chunks_file = Path(chunks_path)

    if db_path is None:
        sqlite_file = Path("data/processed/retrieval.sqlite")
    else:
        sqlite_file = Path(db_path)

    if not chunks_file.is_file():
        raise FileNotFoundError(f"Corpus chunks file not found: {chunks_file}")

    sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_db_path = sqlite_file.with_name(f"{sqlite_file.name}.tmp")

    if tmp_db_path.exists():
        try:
            tmp_db_path.unlink()
        except OSError:
            pass

    t0 = time.perf_counter()
    corpus_sha256 = compute_file_sha256(chunks_file)

    con: sqlite3.Connection | None = None
    try:
        con = sqlite3.connect(str(tmp_db_path))
        con.execute("PRAGMA synchronous = OFF;")
        con.execute("PRAGMA journal_mode = DELETE;")

        init_schema(con)

        chunk_count = 0
        batch: list[tuple[Any, ...]] = []

        with chunks_file.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                c = json.loads(line_str)
                chunk_count += 1
                batch.append((
                    c["chunk_id"],
                    c["ticker"],
                    c["cik"],
                    int(c["fiscal_year"]),
                    c["form"],
                    c["accession"],
                    c["section"],
                    c.get("section_confidence"),
                    c["chunk_type"],
                    c["text"],
                    int(c["char_start"]),
                    int(c["char_end"]),
                    c["source_url"],
                    c["source_file_sha256"],
                    c["canonical_text_path"],
                    c.get("filing_date"),
                    c.get("period_end_date"),
                    c.get("table_title"),
                ))
                if len(batch) >= batch_size:
                    con.executemany("""
                        INSERT INTO chunks (
                            chunk_id, ticker, cik, fiscal_year, form, accession,
                            section, section_confidence, chunk_type, text,
                            char_start, char_end, source_url, source_file_sha256,
                            canonical_text_path, filing_date, period_end_date, table_title
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    batch = []

        if chunk_count == 0:
            raise ValueError(
                f"Cannot build index: corpus file {chunks_file} contains no valid chunks."
            )

        if batch:
            con.executemany("""
                INSERT INTO chunks (
                    chunk_id, ticker, cik, fiscal_year, form, accession,
                    section, section_confidence, chunk_type, text,
                    char_start, char_end, source_url, source_file_sha256,
                    canonical_text_path, filing_date, period_end_date, table_title
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)

        # Rebuild FTS5 inverted index from populated chunks table
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")

        # Run FTS5 external content integrity check
        con.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1);")

        # Verify indexed counts
        chunks_count_db = con.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
        fts_count_db = con.execute("SELECT COUNT(*) FROM chunks_fts;").fetchone()[0]

        if chunks_count_db != chunk_count or fts_count_db != chunk_count:
            raise ValueError(
                f"Index chunk count verification mismatch: read {chunk_count}, "
                f"chunks table has {chunks_count_db}, fts table has {fts_count_db}"
            )

        # Store index metadata
        con.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('corpus_sha256', ?);",
            (corpus_sha256,),
        )
        con.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('chunk_count', ?);",
            (str(chunk_count),),
        )
        con.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('indexed_at', ?);",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),),
        )

        con.commit()
        con.close()
        con = None

        # Atomic replacement of final database file
        tmp_db_path.replace(sqlite_file)

    except Exception:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        if tmp_db_path.exists():
            try:
                tmp_db_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if tmp_db_path.exists():
            try:
                tmp_db_path.unlink()
            except OSError:
                pass

    elapsed = time.perf_counter() - t0
    return {
        "indexed_chunks": chunk_count,
        "corpus_sha256": corpus_sha256,
        "db_path": str(sqlite_file),
        "elapsed_seconds": round(elapsed, 3),
    }


def get_index_metadata(db_path: Path | str | None = None) -> dict[str, str]:
    """Retrieve metadata stored in the index database."""
    if db_path is None:
        sqlite_file = Path("data/processed/retrieval.sqlite")
    else:
        sqlite_file = Path(db_path)

    if not sqlite_file.is_file():
        raise FileNotFoundError(f"Retrieval database not found: {sqlite_file}")

    con = sqlite3.connect(str(sqlite_file))
    try:
        cur = con.execute("SELECT key, value FROM index_metadata;")
        return dict(cur.fetchall())
    finally:
        con.close()


def search(
    query: str,
    ticker: str | None = None,
    year: int | None = None,
    fiscal_year: int | None = None,
    accession: str | None = None,
    section: str | None = None,
    chunk_type: str | None = None,
    top_k: int = 5,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Execute BM25 search over the indexed chunks with metadata filtering.

    Args:
        query: User search query string.
        ticker: Optional company ticker filter (e.g. MSFT).
        year: Optional fiscal year filter (e.g. 2024).
        fiscal_year: Alias for year.
        accession: Optional SEC filing accession number.
        section: Optional exact stored section name.
        chunk_type: Optional chunk type filter ('text' or 'table').
        top_k: Number of results to return (1-20, default 5).
        db_path: Path to SQLite retrieval database.

    Returns:
        List of result dictionaries containing all chunk fields, scores,
        and standard aliases. Returns [] if no matching documents.
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query).__name__}")

    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        raise ValueError(f"top_k must be an integer between 1 and 20, got {top_k}")

    effective_year: int | None = None
    if year is not None and fiscal_year is not None:
        y1 = _validate_year(year, "year")
        y2 = _validate_year(fiscal_year, "fiscal_year")
        if y1 != y2:
            raise ValueError(
                f"Conflicting fiscal year filters: year={year!r} and fiscal_year={fiscal_year!r}"
            )
        effective_year = y1
    elif year is not None:
        effective_year = _validate_year(year, "year")
    elif fiscal_year is not None:
        effective_year = _validate_year(fiscal_year, "fiscal_year")

    if db_path is None:
        sqlite_file = Path("data/processed/retrieval.sqlite")
    else:
        sqlite_file = Path(db_path)

    if not sqlite_file.is_file():
        raise FileNotFoundError(f"Retrieval database not found: {sqlite_file}")

    tokens = tokenize_query(query)
    if not tokens:
        return []

    match_expr = build_match_expression(tokens)
    if not match_expr:
        return []

    conditions = ["chunks_fts MATCH ?"]
    params: list[Any] = [match_expr]

    if ticker is not None and ticker.strip():
        conditions.append("c.ticker = ?")
        params.append(ticker.strip().upper())

    if effective_year is not None:
        conditions.append("c.fiscal_year = ?")
        params.append(effective_year)

    if accession is not None and accession.strip():
        conditions.append("c.accession = ?")
        params.append(accession.strip())

    if section is not None and section.strip():
        conditions.append("c.section = ?")
        params.append(section.strip())

    if chunk_type is not None and chunk_type.strip():
        conditions.append("c.chunk_type = ?")
        params.append(chunk_type.strip().lower())

    where_clause = " AND ".join(conditions)
    params.append(top_k)

    sql = f"""
        SELECT c.chunk_id,
               c.ticker,
               c.cik,
               c.fiscal_year,
               c.form,
               c.accession,
               c.section,
               c.section_confidence,
               c.chunk_type,
               c.text,
               c.char_start,
               c.char_end,
               c.source_url,
               c.source_file_sha256,
               c.canonical_text_path,
               c.filing_date,
               c.period_end_date,
               c.table_title,
               bm25(chunks_fts) AS bm25_score
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE {where_clause}
        ORDER BY bm25_score ASC, c.chunk_id ASC
        LIMIT ?
    """

    con = sqlite3.connect(str(sqlite_file))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            score = float(r["bm25_score"])
            results.append({
                "chunk_id": r["chunk_id"],
                "score": score,
                "relevance_score": score,
                "ticker": r["ticker"],
                "company": r["ticker"],
                "cik": r["cik"],
                "fiscal_year": int(r["fiscal_year"]),
                "year": int(r["fiscal_year"]),
                "form": r["form"],
                "accession": r["accession"],
                "section": r["section"],
                "section_confidence": r["section_confidence"],
                "chunk_type": r["chunk_type"],
                "text": r["text"],
                "chunk_text": r["text"],
                "char_start": int(r["char_start"]),
                "char_end": int(r["char_end"]),
                "source_url": r["source_url"],
                "source_file_sha256": r["source_file_sha256"],
                "source_hash": r["source_file_sha256"],
                "canonical_text_path": r["canonical_text_path"],
                "filing_date": r["filing_date"],
                "period_end_date": r["period_end_date"],
                "table_title": r["table_title"],
            })
        return results
    finally:
        con.close()


def main() -> None:
    """CLI entrypoint for building and searching the retrieval index."""
    parser = argparse.ArgumentParser(
        prog="python -m fyntrace.retrieval",
        description="Fyntrace BM25 filing retrieval system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: build
    build_parser = subparsers.add_parser("build", help="Build SQLite FTS5 BM25 index.")
    build_parser.add_argument(
        "--chunks-path",
        default=None,
        help="Path to chunks.jsonl (default: data/processed/chunks.jsonl)",
    )
    build_parser.add_argument(
        "--db-path",
        default=None,
        help="Path to output SQLite database (default: data/processed/retrieval.sqlite)",
    )

    # Subcommand: search
    search_parser = subparsers.add_parser("search", help="Search the retrieval index.")
    search_parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="User search query string.",
    )
    search_parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker filter (e.g. MSFT, AAPL).",
    )
    search_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Fiscal year filter (e.g. 2024).",
    )
    search_parser.add_argument(
        "--accession",
        default=None,
        help="SEC filing accession number filter.",
    )
    search_parser.add_argument(
        "--section",
        default=None,
        help="Exact stored section name filter.",
    )
    search_parser.add_argument(
        "--chunk-type",
        dest="chunk_type",
        default=None,
        help="Chunk type filter ('text' or 'table').",
    )
    search_parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=5,
        help="Number of results to return (1-20, default 5).",
    )
    search_parser.add_argument(
        "--db-path",
        default=None,
        help="Path to SQLite database (default: data/processed/retrieval.sqlite)",
    )

    args = parser.parse_args()

    if args.command == "build":
        summary = build_index(
            chunks_path=args.chunks_path,
            db_path=args.db_path,
        )
        print("Indexed chunks:", summary["indexed_chunks"])
        print("Corpus SHA-256:", summary["corpus_sha256"])
        print("Database path:", summary["db_path"])
        print("Indexing time:", f"{summary['elapsed_seconds']}s")

    elif args.command == "search":
        results = search(
            query=args.query,
            ticker=args.ticker,
            year=args.year,
            accession=args.accession,
            section=args.section,
            chunk_type=args.chunk_type,
            top_k=args.top_k,
            db_path=args.db_path,
        )

        if not results:
            print("No matching chunks found.")
            return

        print(f"Found {len(results)} matching chunk(s):\n")
        for idx, res in enumerate(results, start=1):
            text_snippet = res["text"].strip().replace("\n", " ")
            if len(text_snippet) > 160:
                text_snippet = text_snippet[:157] + "..."
            print(f"[{idx}] {res['chunk_id']} | Score: {res['score']:.4f}")
            print(f"    Company: {res['ticker']} | Year: {res['fiscal_year']} | Type: {res['chunk_type']}")
            print(f"    Section: {res['section']}")
            print(f"    Excerpt: {text_snippet}")
            print(f"    URL: {res['source_url']}\n")


if __name__ == "__main__":
    main()
