"""SEC filing text extraction and source-aware evidence chunking module."""

import argparse
import hashlib
import json
import os
import re
import uuid
import warnings
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment, Tag, XMLParsedAsHTMLWarning
import requests

from src.sec_data import DEFAULT_TIMEOUT, load_filing_manifest, make_sec_session

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "processed" / "filing_manifest.json"
DEFAULT_RAW_FILINGS_DIR = BASE_DIR / "data" / "raw" / "filings"
DEFAULT_PROCESSED_TEXT_DIR = BASE_DIR / "data" / "processed" / "filing_text"
DEFAULT_CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"

SECTIONS_CONFIG = [
    {
        "id": "item1",
        "name": "Item 1 - Business",
        "pattern": re.compile(
            r'(?:^|\n)\s*(?:PART\s+I[.:\s\u2013\u2014\-|]*)?\s*ITEM\s+1[.:\s\u2013\u2014\-|]*\s*BUSINESS\b',
            re.IGNORECASE,
        ),
    },
    {
        "id": "item1a",
        "name": "Item 1A - Risk Factors",
        "pattern": re.compile(
            r'(?:^|\n)\s*(?:PART\s+I[.:\s\u2013\u2014\-|]*)?\s*ITEM\s+1A[.:\s\u2013\u2014\-|]*\s*RISK\s+FACTORS\b',
            re.IGNORECASE,
        ),
    },
    {
        "id": "item7",
        "name": "Item 7 - Management's Discussion and Analysis",
        "pattern": re.compile(
            r'(?:^|\n)\s*(?:PART\s+II[.:\s\u2013\u2014\-|]*)?\s*ITEM\s+7[.:\s\u2013\u2014\-|]*\s*MANAGEMENT(?:\'S|\u2019S)?\s+DISCUSSION\b',
            re.IGNORECASE,
        ),
    },
    {
        "id": "item7a",
        "name": "Item 7A - Market Risk",
        "pattern": re.compile(
            r'(?:^|\n)\s*(?:PART\s+II[.:\s\u2013\u2014\-|]*)?\s*ITEM\s+7A[.:\s\u2013\u2014\-|]*\s*(?:QUANTITATIVE|MARKET\s+RISK)\b',
            re.IGNORECASE,
        ),
    },
    {
        "id": "item8",
        "name": "Item 8 - Financial Statements",
        "pattern": re.compile(
            r'(?:^|\n)\s*(?:PART\s+II[.:\s\u2013\u2014\-|]*)?\s*ITEM\s+8[.:\s\u2013\u2014\-|]*\s*FINANCIAL\s+STATEMENTS\b',
            re.IGNORECASE,
        ),
    },
]

ANY_ITEM_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:PART\s+[I|V|X]+[.:\s\u2013\u2014\-|]*)?\s*ITEM\s+(?:1A|1B|1C|1|2|3|4|5|6|7A|7|8|9A|9B|9|10|11|12|13|14|15)[.:\s\u2013\u2014\-|]',
    re.IGNORECASE,
)


def is_valid_html_content(content: bytes) -> bool:
    """Check whether downloaded bytes represent valid SEC HTML filing content."""
    if not content or len(content.strip()) < 100:
        return False

    prefix = content[:4096].lower()
    html_markers = (b"<html", b"<!doctype html", b"<xml", b"<?xml", b"<document", b"<body")
    has_marker = any(marker in prefix for marker in html_markers)
    if not has_marker:
        return False

    denial_signatures = (
        b"your request has been blocked",
        b"access denied",
        b"sec.gov/developer",
        b"403 forbidden",
    )
    if any(sig in prefix for sig in denial_signatures):
        return False

    return True


def find_filing(
    ticker: str,
    fiscal_year: int,
    manifest: list[dict[str, Any]] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Select exactly one verified Form 10-K filing record by ticker and fiscal year.

    Preserves exact CIK (including historical XOM CIK 0000034088), accession,
    report date, primary document, and SEC source URL from the manifest.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"Invalid ticker: {ticker!r}")
    ticker_clean = ticker.strip().upper()

    if not isinstance(fiscal_year, int) or isinstance(fiscal_year, bool):
        raise ValueError(f"Invalid fiscal year: {fiscal_year!r}")

    records = manifest if manifest is not None else load_filing_manifest(manifest_path)

    matches = [
        r
        for r in records
        if r.get("ticker", "").strip().upper() == ticker_clean
        and r.get("target_fiscal_year") == fiscal_year
    ]

    if not matches:
        raise KeyError(f"No filing found for company '{ticker_clean}' fiscal year {fiscal_year}.")

    if len(matches) > 1:
        raise ValueError(f"Duplicate filing records found for company '{ticker_clean}' FY{fiscal_year}.")

    rec = matches[0]

    if rec.get("fiscal_year_status") != "confirmed":
        raise ValueError(
            f"Filing for {ticker_clean} FY{fiscal_year} status is {rec.get('fiscal_year_status')!r}, expected 'confirmed'."
        )

    form = rec.get("form", "").strip()
    if form != "10-K":
        raise ValueError(f"Filing for {ticker_clean} FY{fiscal_year} has form {form!r}, expected '10-K'.")

    required_fields = (
        "ticker",
        "company_name",
        "cik",
        "target_fiscal_year",
        "accession_number",
        "filing_date",
        "report_date",
        "primary_document",
        "source_url",
    )
    for field in required_fields:
        if not rec.get(field):
            raise ValueError(f"Filing record for {ticker_clean} FY{fiscal_year} missing required field '{field}'.")

    return rec


def download_filing(
    filing_record: dict[str, Any],
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """Download and cache SEC filing HTML document.

    Reuses valid cached HTML files without downloading again.
    Returns:
        tuple[Path, str]: (path_to_cached_html_file, sha256_hexdigest)
    """
    raw_filings_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_RAW_FILINGS_DIR
    raw_filings_dir.mkdir(parents=True, exist_ok=True)

    accession = filing_record["accession_number"]
    target_file = raw_filings_dir / f"{accession}.html"

    # Reuse valid cache
    if target_file.is_file():
        try:
            cached_bytes = target_file.read_bytes()
            if is_valid_html_content(cached_bytes):
                sha256_hash = hashlib.sha256(cached_bytes).hexdigest()
                return target_file, sha256_hash
        except Exception:
            pass

    source_url = filing_record.get("source_url")
    if not source_url:
        raise ValueError(f"Filing record missing source_url for {accession}.")

    sec_session = session if session is not None else make_sec_session()

    try:
        response = sec_session.get(
            source_url,
            headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error fetching {source_url}: {exc}") from exc

    if response.status_code == 403:
        raise PermissionError(f"SEC API access denied (403 Forbidden) for {source_url}. Check User-Agent.")
    elif response.status_code == 404:
        raise FileNotFoundError(f"SEC filing document not found (404) at {source_url}.")
    elif response.status_code != 200:
        raise RuntimeError(f"HTTP error {response.status_code} fetching {source_url}.")

    content_bytes = response.content
    if not is_valid_html_content(content_bytes):
        raise ValueError(f"Downloaded content from {source_url} is not valid SEC HTML document.")

    sha256_hash = hashlib.sha256(content_bytes).hexdigest()
    tmp_file = raw_filings_dir / f"{accession}.html.tmp_{uuid.uuid4().hex}"
    try:
        tmp_file.write_bytes(content_bytes)
        tmp_file.replace(target_file)
    except Exception as exc:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to save filing HTML to {target_file}: {exc}") from exc

    return target_file, sha256_hash


def extract_table_text(table_tag: Tag) -> str:
    """Format an HTML table tag into a readable string preserving rows and cells.

    Preserves numbers, negative signs, decimals, commas, percentages, currency symbols.
    Cells in a row are separated by ' | '.
    """
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            text = " ".join(cell.get_text(" ", strip=True).split())
            if text:
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_visible_text(html_content: str | bytes) -> str:
    """Extract readable, normalized canonical text from filing HTML.

    Preserves real report content, tables, financial values, and numbers.
    Removes scripts, styles, comments, navigation, and hidden elements.
    """
    if isinstance(html_content, bytes):
        raw_bytes = html_content
    elif isinstance(html_content, str):
        raw_bytes = html_content.encode("utf-8", errors="replace")
    else:
        raise TypeError(f"Expected str or bytes for html_content, got {type(html_content)}")

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ValueError("HTML content is empty.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(raw_bytes, "lxml")

    # 1. Remove non-visible and irrelevant tags
    for tag in soup(["script", "style", "noscript", "nav", "svg", "head"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.lower() in ("ix:header", "ix:hidden")):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    for tag in list(soup.find_all(True)):
        if hasattr(tag, "attrs") and isinstance(tag.attrs, dict):
            s = tag.attrs.get("style", "").lower().replace(" ", "")
            if "display:none" in s or "visibility:hidden" in s:
                tag.decompose()

    # 2. Convert tables from innermost to outermost
    while True:
        tables = soup.find_all("table")
        if not tables:
            break
        innermost = [t for t in tables if not t.find("table")]
        for t in innermost:
            tbl_txt = extract_table_text(t)
            new_tag = soup.new_tag("div")
            new_tag.string = f"\n\n{tbl_txt}\n\n"
            t.replace_with(new_tag)

    # 3. Add newlines around block tags
    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "section", "article"}
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    for br in soup.find_all("br"):
        br.replace_with("\n")

    # 4. Extract text and normalize whitespace
    raw_text = soup.get_text()
    raw_text = (
        raw_text.replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\u200e", "")
        .replace("\u200f", "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )

    lines = [" ".join(l.split()) for l in raw_text.splitlines()]
    cleaned_lines = []
    for l in lines:
        if l:
            cleaned_lines.append(l)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    canonical_text = "\n".join(cleaned_lines).strip()
    if len(canonical_text) < 50:
        raise ValueError("Extracted text is too short or empty; likely malformed HTML.")

    return canonical_text


def extract_sections(canonical_text: str) -> list[dict[str, Any]]:
    """Identify key 10-K report sections with confidence levels.

    Detects:
    - Item 1 - Business
    - Item 1A - Risk Factors
    - Item 7 - Management's Discussion and Analysis
    - Item 7A - Market Risk
    - Item 8 - Financial Statements

    Distinguishes Table of Contents headings from actual report body sections.
    Assigns confidence: 'high', 'low', or 'unknown'.
    """
    all_item_matches = [m.start() for m in ANY_ITEM_PATTERN.finditer(canonical_text)]

    detected_starts: list[tuple[int, str]] = []

    for cfg in SECTIONS_CONFIG:
        matches = list(cfg["pattern"].finditer(canonical_text))
        if not matches:
            continue

        body_match = None
        for m in matches:
            pos = m.start()
            l_end = canonical_text.find("\n", pos)
            line = canonical_text[pos : l_end if l_end != -1 else pos + 150]

            # Check if TOC heading:
            # - Ends with pipe and page number or dots (e.g. '| 15' or '... 15')
            # - Or if multiple matches exist, the earlier matches are TOC
            is_toc = False
            if re.search(r"(?:\|\s*\d+|\.{2,}\s*\d+)\s*$", line):
                is_toc = True
            elif len(matches) > 1 and m != matches[-1]:
                is_toc = True

            if not is_toc:
                body_match = m
                break

        if body_match is not None:
            detected_starts.append((body_match.start(), cfg["name"]))

    detected_starts.sort(key=lambda x: x[0])

    sections: list[dict[str, Any]] = []
    for idx, (start_pos, sec_name) in enumerate(detected_starts):
        if idx + 1 < len(detected_starts):
            end_pos = detected_starts[idx + 1][0]
        else:
            subsequent = [p for p in all_item_matches if p > start_pos + 100]
            end_pos = subsequent[0] if subsequent else len(canonical_text)

        length = end_pos - start_pos
        confidence = "high" if length >= 2000 else "low"
        sections.append({
            "name": sec_name,
            "start": start_pos,
            "end": end_pos,
            "confidence": confidence,
            "length": length,
        })

    return sections


def make_chunks(
    canonical_text: str,
    filing_record: dict[str, Any],
    file_sha256: str,
    sections: list[dict[str, Any]] | None = None,
    target_words: int = 400,
    overlap_words: int = 50,
) -> list[dict[str, Any]]:
    """Generate source-aware evidence chunks from canonical filing text.

    Targets ~300-500 words per chunk with ~50 words overlap on paragraph boundaries.
    Preserves exact character offsets such that canonical_text[char_start:char_end] == chunk['text'].
    """
    if sections is None:
        sections = extract_sections(canonical_text)

    blocks: list[tuple[int, int, str, int]] = []
    for m in re.finditer(r'(?:\S[^\n]*\n?)+', canonical_text):
        text = m.group().strip()
        if text:
            start = m.start()
            exact_start = canonical_text.find(text, start)
            exact_end = exact_start + len(text)
            word_count = len(text.split())
            blocks.append((exact_start, exact_end, text, word_count))

    if not blocks:
        return []

    ticker = filing_record["ticker"]
    company_name = filing_record["company_name"]
    cik = filing_record["cik"]
    fy = filing_record["target_fiscal_year"]
    form = filing_record["form"]
    accession = filing_record["accession_number"]
    report_date = filing_record["report_date"]
    source_url = filing_record["source_url"]
    clean_accn = accession.replace("-", "")

    def get_section_info(start_pos: int, end_pos: int) -> tuple[str, str]:
        mid = (start_pos + end_pos) // 2
        for sec in sections:
            if sec["start"] <= mid < sec["end"]:
                return sec["name"], sec["confidence"]
        return "unknown", "unknown"

    chunks: list[dict[str, Any]] = []
    i = 0
    seq = 1

    while i < len(blocks):
        cur_words = 0
        start_idx = i
        end_idx = i
        current_sec_name, _ = get_section_info(blocks[start_idx][0], blocks[start_idx][1])

        while end_idx < len(blocks):
            b_start, b_end, _, b_words = blocks[end_idx]
            block_sec, _ = get_section_info(b_start, b_end)

            if end_idx > start_idx and block_sec != current_sec_name and block_sec != "unknown":
                break

            if cur_words > 0 and cur_words + b_words > (target_words + 100):
                break

            cur_words += b_words
            end_idx += 1

            if cur_words >= target_words:
                break

        c_start = blocks[start_idx][0]
        c_end = blocks[end_idx - 1][1]
        c_text = canonical_text[c_start:c_end]
        sec_name, sec_conf = get_section_info(c_start, c_end)

        chunk_id = f"{ticker}_{fy}_10K_{clean_accn}_{seq:04d}"
        seq += 1

        chunk_record = {
            "chunk_id": chunk_id,
            "ticker": ticker,
            "company_name": company_name,
            "cik": cik,
            "fiscal_year": fy,
            "form": form,
            "accession": accession,
            "report_date": report_date,
            "section": sec_name,
            "section_confidence": sec_conf,
            "text": c_text,
            "char_start": c_start,
            "char_end": c_end,
            "source_url": source_url,
            "file_sha256": file_sha256,
        }
        chunks.append(chunk_record)

        if end_idx >= len(blocks):
            break

        next_i = end_idx
        overlap_accum = 0
        for j in range(end_idx - 1, start_idx, -1):
            overlap_accum += blocks[j][3]
            if overlap_accum >= overlap_words:
                next_i = j
                break

        if next_i <= start_idx:
            next_i = start_idx + 1

        i = next_i

    for ch in chunks:
        assert canonical_text[ch["char_start"]:ch["char_end"]] == ch["text"], "Character offset mismatch!"

    return chunks


def build_filing_corpus(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    all_filings: bool = False,
    manifest: list[dict[str, Any]] | None = None,
    manifest_path: Path | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    chunks_path: Path | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Build the source-aware filing text corpus and write chunks.jsonl atomically."""
    if manifest is not None:
        manifest_records = manifest
    elif manifest_path is not None:
        try:
            manifest_records = load_filing_manifest(manifest_path)
        except ValueError:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_records = json.load(f)
    else:
        manifest_records = load_filing_manifest(DEFAULT_MANIFEST_PATH)

    if not all_filings and (not tickers or not years):
        raise ValueError("Must specify either --all or both tickers and years.")

    if all_filings:
        target_records = manifest_records
    else:
        tickers_clean = [t.strip().upper() for t in tickers]
        years_list = list(years)
        target_records = [
            find_filing(t, y, manifest=manifest_records)
            for t in tickers_clean
            for y in years_list
        ]

    raw_filings_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_FILINGS_DIR
    processed_text_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_TEXT_DIR
    out_chunks_path = Path(chunks_path) if chunks_path is not None else DEFAULT_CHUNKS_PATH

    raw_filings_dir.mkdir(parents=True, exist_ok=True)
    processed_text_dir.mkdir(parents=True, exist_ok=True)
    out_chunks_path.parent.mkdir(parents=True, exist_ok=True)

    sec_session = session if session is not None else make_sec_session()

    successes: list[str] = []
    failures: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []

    for rec in target_records:
        ticker = rec["ticker"]
        fy = rec["target_fiscal_year"]
        accn = rec["accession_number"]
        ident = f"{ticker} FY{fy} ({accn})"
        try:
            html_file, file_hash = download_filing(rec, session=sec_session, cache_dir=raw_filings_dir)
            html_bytes = html_file.read_bytes()
            canonical_text = extract_visible_text(html_bytes)

            txt_file = processed_text_dir / f"{accn}.txt"
            txt_file.write_text(canonical_text, encoding="utf-8")

            sections = extract_sections(canonical_text)
            chunks = make_chunks(canonical_text, rec, file_hash, sections=sections)
            all_chunks.extend(chunks)
            successes.append(ident)
        except Exception as exc:
            failures.append({"identifier": ident, "error": str(exc)})

    # Atomically write chunks.jsonl
    tmp_chunks = out_chunks_path.parent / f"chunks.jsonl.tmp_{uuid.uuid4().hex}"
    try:
        with open(tmp_chunks, "w", encoding="utf-8") as f:
            for ch in all_chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        tmp_chunks.replace(out_chunks_path)
    except Exception as exc:
        if tmp_chunks.exists():
            tmp_chunks.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to write chunks to {out_chunks_path}: {exc}") from exc

    return {
        "filings_requested": len(target_records),
        "filings_succeeded": len(successes),
        "filings_failed": len(failures),
        "successes": successes,
        "failures": failures,
        "total_chunks": len(all_chunks),
        "chunks_path": str(out_chunks_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SEC filing text extraction and source-aware evidence chunks.")
    parser.add_argument("--tickers", nargs="+", help="Company tickers to process (e.g. MSFT JPM WMT XOM).")
    parser.add_argument("--years", nargs="+", type=int, help="Target fiscal years to process (e.g. 2024).")
    parser.add_argument("--all", dest="all_filings", action="store_true", help="Process all filings in manifest.")
    args = parser.parse_args(argv)
    if not args.all_filings and (not args.tickers or not args.years):
        parser.error("You must specify --tickers and --years, or use --all.")
    return args


def main(argv: list[str] | None = None) -> None:
    """Entry point for SEC filing text extraction CLI."""
    args = parse_args(argv)
    summary = build_filing_corpus(
        tickers=args.tickers,
        years=args.years,
        all_filings=args.all_filings,
    )
    print(f"Filings requested: {summary['filings_requested']}")
    print(f"Filings succeeded: {summary['filings_succeeded']}")
    print(f"Filings failed:    {summary['filings_failed']}")
    print(f"Total chunks:      {summary['total_chunks']}")
    print(f"Chunks saved to:   {summary['chunks_path']}")
    if summary["failures"]:
        print("Failures encountered:")
        for fail in summary["failures"]:
            print(f"  - {fail['identifier']}: {fail['error']}")


if __name__ == "__main__":
    main()
