"""SEC Form 10-K document processing and evidence chunking for Fyntrace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import warnings

import bs4
from bs4 import XMLParsedAsHTMLWarning

from fyntrace import config
from fyntrace import sec_data

# Suppress XMLParsedAsHTMLWarning when BeautifulSoup parses XHTML 10-K filings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Default data directories
DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "manifest.json"

# Form 10-K standard Item ordering and canonical naming
ITEM_ORDER: dict[str, int] = {
    k: idx
    for idx, k in enumerate([
        "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
        "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
    ])
}

ITEM_CANONICAL: dict[str, str] = {
    "1": "Item 1 - Business",
    "1A": "Item 1A - Risk Factors",
    "1B": "Item 1B - Unresolved Staff Comments",
    "1C": "Item 1C - Cybersecurity",
    "2": "Item 2 - Properties",
    "3": "Item 3 - Legal Proceedings",
    "4": "Item 4 - Mine Safety Disclosures",
    "5": "Item 5 - Market for Registrant's Common Equity",
    "6": "Item 6 - Selected Financial Data",
    "7": "Item 7 - Management's Discussion and Analysis",
    "7A": "Item 7A - Market Risk Disclosures",
    "8": "Item 8 - Financial Statements",
    "9": "Item 9 - Changes in and Disagreements With Accountants",
    "9A": "Item 9A - Controls and Procedures",
    "9B": "Item 9B - Other Information",
    "9C": "Item 9C - Disclosure Regarding Foreign Jurisdictions",
    "10": "Item 10 - Directors, Executive Officers and Corporate Governance",
    "11": "Item 11 - Executive Compensation",
    "12": "Item 12 - Security Ownership of Certain Beneficial Owners",
    "13": "Item 13 - Certain Relationships and Related Transactions",
    "14": "Item 14 - Principal Accountant Fees and Services",
    "15": "Item 15 - Exhibits and Financial Statement Schedules",
    "16": "Item 16 - Form 10-K Summary",
}

ITEM_KEYWORDS: dict[str, list[str]] = {
    "1": ["BUSINESS"],
    "1A": ["RISK", "FACTORS"],
    "1B": ["UNRESOLVED", "STAFF"],
    "1C": ["CYBERSECURITY"],
    "2": ["PROPERTIES"],
    "3": ["LEGAL", "PROCEEDINGS"],
    "4": ["MINE", "SAFETY"],
    "5": ["MARKET", "COMMON EQUITY", "EQUITY"],
    "6": ["SELECTED", "RESERVED"],
    "7": ["MANAGEMENT", "DISCUSSION", "ANALYSIS"],
    "7A": ["QUANTITATIVE", "QUALITATIVE", "MARKET RISK"],
    "8": ["FINANCIAL STATEMENTS", "SUPPLEMENTARY DATA"],
    "9": ["CHANGES IN AND DISAGREEMENTS", "ACCOUNTANTS"],
    "9A": ["CONTROLS", "PROCEDURES"],
    "9B": ["OTHER INFORMATION"],
    "9C": ["FOREIGN JURISDICTIONS"],
    "10": ["DIRECTORS", "EXECUTIVE OFFICERS", "GOVERNANCE"],
    "11": ["EXECUTIVE COMPENSATION"],
    "12": ["SECURITY OWNERSHIP", "BENEFICIAL OWNERS"],
    "13": ["CERTAIN RELATIONSHIPS", "RELATED TRANSACTIONS"],
    "14": ["PRINCIPAL ACCOUNTANT", "ACCOUNTING FEES"],
    "15": ["EXHIBITS", "SCHEDULES"],
    "16": ["SUMMARY"],
}

# HTML block tags that imply line breaks
BLOCK_TAGS: set[str] = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "ul", "ol", "section", "article", "header", "footer", "tr", "blockquote",
}


def load_validated_filings(
    manifest_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Read and validate the existing filing manifest and local source files.

    Ensures all 15 panel companies and supported fiscal years are valid,
    unaltered, and securely contained within the raw filings directory.
    """
    manifest_file = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Filing manifest not found at '{manifest_file}'.")

    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse manifest JSON at '{manifest_file}': {exc}") from exc

    if not isinstance(manifest_data, dict):
        raise ValueError("Invalid manifest structure: root is not a dictionary.")

    # Build CIK lookup from company identities
    cik_by_ticker: dict[str, str] = {}
    for identity in manifest_data.get("company_identities", []):
        if isinstance(identity, dict) and "ticker" in identity and "cik" in identity:
            cik_by_ticker[str(identity["ticker"]).upper()] = str(identity["cik"]).zfill(10)

    filings_raw = manifest_data.get("filings")
    if not isinstance(filings_raw, list):
        raise ValueError("Invalid manifest structure: 'filings' is not a list.")

    filings_dir = (Path(raw_dir) / "filings") if raw_dir else (DEFAULT_RAW_DIR / "filings")
    resolved_filings_dir = filings_dir.resolve()

    validated: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    for item in filings_raw:
        if not isinstance(item, dict):
            continue

        ticker = str(item.get("ticker", "")).strip().upper()
        if not config.is_supported_ticker(ticker):
            continue

        try:
            year = int(item.get("fiscal_year", 0))
        except (ValueError, TypeError):
            continue
        if not config.is_supported_year(year):
            continue

        status = str(item.get("status", "")).lower()
        if status != "downloaded":
            raise ValueError(f"Filing {ticker} FY{year} status is '{status}', expected 'downloaded'.")

        form = str(item.get("form", "")).upper()
        if form != "10-K":
            raise ValueError(f"Filing {ticker} FY{year} form is '{form}', expected '10-K'.")

        accession = str(item.get("accession", "")).strip()
        if not re.match(r"^\d{10}-\d{2}-\d{6}$", accession):
            raise ValueError(f"Filing {ticker} FY{year} has invalid accession format '{accession}'.")

        # CIK resolution
        cik = cik_by_ticker.get(ticker)
        if not cik:
            cik_cand = item.get("cik")
            if cik_cand:
                cik = str(cik_cand).zfill(10)
        if not cik or not re.match(r"^\d{10}$", cik):
            raise ValueError(f"Filing {ticker} FY{year} missing valid 10-digit CIK.")

        # Source URL check
        source_url = str(item.get("source_url", "")).strip()
        try:
            sec_data.validate_sec_url(source_url)
        except Exception as exc:
            raise ValueError(f"Filing {ticker} FY{year} source URL '{source_url}' is not a valid SEC URL: {exc}") from exc

        # Path validation and directory containment
        local_path_str = str(item.get("local_path", ""))
        local_path = Path(local_path_str)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local filing file not found for {ticker} FY{year}: '{local_path}'.")

        resolved_local_path = local_path.resolve()
        try:
            resolved_local_path.relative_to(resolved_filings_dir)
        except ValueError as exc:
            raise ValueError(
                f"Security violation: filing file '{local_path}' is outside filings dir '{filings_dir}'."
            ) from exc

        # SHA-256 verification
        expected_sha = str(item.get("file_sha256", "")).strip().lower()
        if not expected_sha or len(expected_sha) != 64:
            raise ValueError(f"Filing {ticker} FY{year} missing valid SHA-256 in manifest.")

        actual_bytes = local_path.read_bytes()
        actual_sha = hashlib.sha256(actual_bytes).hexdigest().lower()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Filing {ticker} FY{year} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}."
            )

        # Avoid duplicates
        key = (ticker, year)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        validated.append({
            "ticker": ticker,
            "cik": cik,
            "fiscal_year": year,
            "form": form,
            "accession": accession,
            "local_path": str(local_path.as_posix()),
            "source_url": source_url,
            "file_sha256": actual_sha,
            "filing_date": item.get("filing_date"),
            "period_end_date": item.get("period_end_date"),
            "primary_document": item.get("primary_document"),
        })

    # Order by configured company order, then ascending fiscal year
    ticker_order = {t: idx for idx, t in enumerate(config.ALL_TICKERS)}
    validated.sort(key=lambda f: (ticker_order.get(f["ticker"], 999), f["fiscal_year"]))

    return validated


def is_heading_layout_table(table: bs4.element.Tag) -> bool:
    """Check if an HTML table is purely a 1-2 row layout container for an Item heading."""
    direct_rows = [r for r in table.find_all("tr") if r.find_parent("table") == table]
    if len(direct_rows) > 3:
        return False
    text = table.get_text(" ", strip=True)
    item_match = re.search(r"(?i)\bITEM\s*(?:1A|1B|1C|7A|9A|9B|9C|[1-9]|1[0-6])\b", text)
    if not item_match:
        return False
    # If there are numbers other than the item number itself, it is likely a data table
    text_no_item = re.sub(r"(?i)\bITEM\s*(?:1A|1B|1C|7A|9A|9B|9C|[1-9]|1[0-6])\b", "", text)
    if re.search(r"\d", text_no_item):
        return False
    return True


def extract_table_text(table: bs4.element.Tag) -> str:
    """Convert an HTML financial table into readable rows.

    Maintains row and column matrix alignment handling colspan, rowspan,
    and prevents nested table row duplication.
    """
    direct_rows = [r for r in table.find_all("tr") if r.find_parent("table") == table]
    if not direct_rows:
        return ""

    grid: list[list[str | None]] = []
    for r_idx, row in enumerate(direct_rows):
        while len(grid) <= r_idx:
            grid.append([])

        col_idx = 0
        direct_cells = [c for c in row.find_all(["th", "td"]) if c.find_parent("table") == table]
        for cell in direct_cells:
            while col_idx < len(grid[r_idx]) and grid[r_idx][col_idx] is not None:
                col_idx += 1

            try:
                colspan = max(1, min(50, int(cell.get("colspan", 1))))
            except (ValueError, TypeError):
                colspan = 1

            try:
                rowspan = max(1, min(100, int(cell.get("rowspan", 1))))
            except (ValueError, TypeError):
                rowspan = 1

            # Handle nested tables inside cell without duplicating
            nested_tables = [t for t in cell.find_all("table") if t.find_parent("table") == table]
            for nested in nested_tables:
                nested_text = extract_table_text(nested)
                nested.replace_with(f" {nested_text} ")

            cell_text = " ".join(cell.get_text(" ", strip=True).split())

            for dr in range(rowspan):
                target_r = r_idx + dr
                while len(grid) <= target_r:
                    grid.append([])
                for dc in range(colspan):
                    target_c = col_idx + dc
                    while len(grid[target_r]) <= target_c:
                        grid[target_r].append(None)
                    grid[target_r][target_c] = cell_text if (dr == 0 and dc == 0) else ""

            col_idx += colspan

    lines: list[str] = []
    caption_tag = table.find("caption")
    if caption_tag and caption_tag.find_parent("table") == table:
        cap = " ".join(caption_tag.get_text(" ", strip=True).split())
        if cap:
            lines.append(f"Table: {cap}")

    for row in grid:
        cleaned_cells = [c if c is not None else "" for c in row]
        if any(c.strip() for c in cleaned_cells):
            while cleaned_cells and not cleaned_cells[-1].strip():
                cleaned_cells.pop()
            lines.append(" | ".join(cleaned_cells))

    return "\n".join(lines)


def extract_visible_text(html_bytes: bytes) -> str:
    """Convert SEC filing HTML bytes into ordered, readable plain text.

    Removes scripts, styles, and hidden XBRL metadata while preserving visible
    financial text, numbers, inline XBRL tags, and formatted tables.
    """
    if not html_bytes or not html_bytes.strip():
        raise ValueError("Cannot extract text from empty HTML bytes.")

    soup = bs4.BeautifulSoup(html_bytes, "lxml")

    # Remove non-content technical tags
    for tag in list(soup.find_all(["script", "style", "noscript", "ix:header", "ix:hidden"])):
        tag.decompose()

    # Remove hidden elements (CSS display:none or visibility:hidden)
    for tag in list(soup.find_all(style=True)):
        if isinstance(tag, bs4.element.Tag) and tag.attrs and "style" in tag.attrs:
            s_lower = str(tag["style"]).lower().replace(" ", "")
            if "display:none" in s_lower or "visibility:hidden" in s_lower:
                tag.decompose()

    # Process tables in document order
    top_tables = [t for t in soup.find_all("table") if t.find_parent("table") is None]
    for table in top_tables:
        if is_heading_layout_table(table):
            cells = [" ".join(c.get_text().split()) for c in table.find_all(["th", "td"]) if c.get_text().strip()]
            table.replace_with(bs4.NavigableString("\n\n" + " ".join(cells) + "\n\n"))
        else:
            tbl_text = extract_table_text(table)
            table.replace_with(bs4.NavigableString(f"\n\n[TABLE START]\n{tbl_text}\n[TABLE END]\n\n"))

    # Block tags create line breaks; <br> becomes newline
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n\n")

    raw_text = soup.get_text()

    # Clean whitespace while preserving line structure and financial values
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in raw_text.split("\n")]
    clean_text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    if not clean_text or not clean_text.strip():
        raise ValueError("Extracted text is empty.")

    return clean_text


def extract_sections(text: str) -> list[dict[str, Any]]:
    """Detect Form 10-K report sections and assign confidence levels.

    Distinguishes actual body sections from Table of Contents listings
    and passing cross-references.
    """
    # Locate all table ranges: true document section headings never appear inside tables
    table_ranges: list[tuple[int, int]] = []
    for tm in re.finditer(r"\[TABLE START\][\s\S]*?\[TABLE END\]", text):
        table_ranges.append((tm.start(), tm.end()))

    pattern = re.compile(
        r"(?i)(?:^|\n)\s*(?:\[TABLE START\]\s*)?(ITEM\s+(1A|1B|1C|7A|9A|9B|9C|[1-9]|1[0-6]))\b[\.\:\—\–\-\s]+(.*?)(?=\n|$)",
        re.MULTILINE,
    )

    candidates: list[dict[str, Any]] = []
    for m in pattern.finditer(text):
        item_id = m.group(2).upper()
        full_line = m.group(0).strip()
        title_part = m.group(3).strip()
        pos = m.start(1)
        end_pos = m.end()

        # Reject any match that appears inside a table (such as TOC tables or data tables)
        if any(ts <= pos < te for ts, te in table_ranges):
            continue

        lower = full_line.lower()
        # Filter passing cross-references
        if any(x in lower for x in [
            "of this form 10-k", "of this report", "on page", "on pages",
            "refer to", "see item", "“", "”", '"',
        ]):
            continue

        # Filter TOC entries with trailing page numbers
        if re.search(r"\b\d{1,3}\s*$", full_line) and 0 < len(title_part) < 60:
            continue

        order = ITEM_ORDER.get(item_id, 999)
        keywords = ITEM_KEYWORDS.get(item_id, [])
        has_kw = any(kw.lower() in full_line.lower() for kw in keywords)

        candidates.append({
            "item_id": item_id,
            "pos": pos,
            "end_pos": end_pos,
            "line": full_line[:80],
            "order": order,
            "has_kw": has_kw,
            "section": ITEM_CANONICAL.get(item_id, f"Item {item_id}"),
        })

    # Filter TOC candidates: where text between candidate and next candidate is trivial (< 10 words)
    non_toc: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        is_toc = False
        if i + 1 < len(candidates):
            next_c = candidates[i + 1]
            between_text = text[c["end_pos"]:next_c["pos"]].strip()
            if len(between_text) < 120 and len(between_text.split()) < 10:
                is_toc = True
        if not is_toc:
            non_toc.append(c)

    # Monotonic selection for body sections
    selected: list[dict[str, Any]] = []
    curr_order = -1
    curr_pos = -1
    for c in non_toc:
        if c["order"] > curr_order and c["pos"] > curr_pos:
            if c["has_kw"]:
                selected.append(c)
                curr_order = c["order"]
                curr_pos = c["pos"]

    # Fallback if no sections identified
    if not selected:
        return [{
            "section": "unknown",
            "item_id": "unknown",
            "char_start": 0,
            "char_end": len(text),
            "confidence": "unknown",
        }]

    sections: list[dict[str, Any]] = []

    # Front matter before the first detected section
    if selected[0]["pos"] > 0:
        sections.append({
            "section": "Cover and Front Matter",
            "item_id": "front",
            "char_start": 0,
            "char_end": selected[0]["pos"],
            "confidence": "unknown",
        })

    for i, s in enumerate(selected):
        start = s["pos"]
        end = selected[i + 1]["pos"] if i + 1 < len(selected) else len(text)
        length = end - start
        conf = "high" if (s["has_kw"] and length > 200) else ("low" if length > 80 else "unknown")
        sections.append({
            "section": s["section"],
            "item_id": s["item_id"],
            "char_start": start,
            "char_end": end,
            "confidence": conf,
        })

    return sections


def _get_section_for_offset(
    offset: int,
    sections: list[dict[str, Any]],
) -> tuple[str, str]:
    """Find the section name and confidence for a given character offset."""
    for s in sections:
        if s["char_start"] <= offset < s["char_end"]:
            return s["section"], s["confidence"]
    if sections:
        return sections[-1]["section"], sections[-1]["confidence"]
    return "unknown", "unknown"


def split_long_paragraph(
    canonical_text: str,
    p_start: int,
    p_end: int,
    max_words: int = 500,
    target_words: int = 400,
    overlap_words: int = 50,
) -> list[tuple[int, int]]:
    """Split an oversized paragraph into spans of at most max_words with overlap."""
    p_text = canonical_text[p_start:p_end]
    words = list(re.finditer(r"\S+", p_text))
    if len(words) <= max_words:
        return [(p_start, p_end)]

    slices: list[tuple[int, int]] = []
    w_idx = 0
    while w_idx < len(words):
        w_end = min(w_idx + target_words, len(words))
        if w_end < len(words):
            sentence_break = None
            search_start = min(w_idx + 350, len(words) - 1)
            search_end = min(w_idx + max_words, len(words))
            for i in range(search_start, search_end):
                w_str = words[i].group(0)
                if w_str.endswith((".", "!", "?")) and not re.search(
                    r"\b(?:[A-Z]\.|Inc\.|Corp\.|Ltd\.|Co\.|U\.S\.|e\.g\.|i\.e\.|vs\.|No\.)",
                    w_str,
                ):
                    sentence_break = i + 1
                    break
            if sentence_break is not None:
                w_end = sentence_break
            else:
                w_end = min(w_idx + max_words, len(words))

        c_start = p_start + words[w_idx].start()
        c_end = p_start + words[w_end - 1].end()
        slices.append((c_start, c_end))

        if w_end >= len(words):
            break

        next_w_idx = max(w_idx + 1, w_end - overlap_words)
        w_idx = next_w_idx

    return slices


def make_chunks(
    filing: dict[str, Any],
    canonical_text: str,
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create source-aware narrative and table chunks from canonical text.

    Ensures canonical_text[char_start:char_end] == chunk['text'] exactly,
    and guarantees chunks do not mix two different confirmed report sections.
    """
    chunks: list[dict[str, Any]] = []
    clean_accn = filing["accession"].replace("-", "")
    clean_form = filing["form"].replace("-", "")
    chunk_seq = 1

    # Locate table spans in canonical text
    table_pattern = re.compile(r"\[TABLE START\]\n(.*?)\n\[TABLE END\]", re.DOTALL)
    table_spans = [(m.start(), m.end(), m.group(1)) for m in table_pattern.finditer(canonical_text)]

    target_words = 400
    max_words = 500
    overlap_words = 50

    # Process section by section so chunks never cross confirmed section boundaries
    for sec in sections:
        sec_start = sec["char_start"]
        sec_end = sec["char_end"]
        sec_name = sec["section"]
        sec_conf = sec["confidence"]

        # Tables within this section
        sec_tables = [(ts, te, content) for ts, te, content in table_spans if ts >= sec_start and te <= sec_end]

        # Sub-segments within section
        curr_idx = sec_start
        sub_segments: list[tuple[int, int, str, str]] = []
        for ts, te, content in sec_tables:
            if ts > curr_idx:
                sub_segments.append((curr_idx, ts, "text", ""))
            sub_segments.append((ts, te, "table", content))
            curr_idx = te
        if curr_idx < sec_end:
            sub_segments.append((curr_idx, sec_end, "text", ""))

        for seg_start, seg_end, seg_type, tbl_content in sub_segments:
            raw_seg_text = canonical_text[seg_start:seg_end]
            if not raw_seg_text.strip():
                continue

            # Strip leading/trailing blank whitespace while maintaining exact offsets
            l_strip = len(raw_seg_text) - len(raw_seg_text.lstrip())
            r_strip = len(raw_seg_text) - len(raw_seg_text.rstrip())
            clean_start = seg_start + l_strip
            clean_end = seg_end - r_strip
            if clean_end <= clean_start:
                continue

            seg_text = canonical_text[clean_start:clean_end]

            if seg_type == "table":
                # Extract optional table title
                table_title = None
                for line in tbl_content.split("\n"):
                    sline = line.strip()
                    if sline.startswith("Table:"):
                        table_title = sline[6:].strip()
                        break
                    elif sline and not table_title:
                        table_title = sline[:80]

                tbl_words = len(seg_text.split())
                if tbl_words <= 600:
                    chunk_id = f"{filing['ticker']}_{filing['fiscal_year']}_{clean_form}_{clean_accn}_{chunk_seq:05d}"
                    chunk_seq += 1
                    chunks.append({
                        "chunk_id": chunk_id,
                        "ticker": filing["ticker"],
                        "cik": filing["cik"],
                        "fiscal_year": filing["fiscal_year"],
                        "form": filing["form"],
                        "accession": filing["accession"],
                        "section": sec_name,
                        "section_confidence": sec_conf,
                        "chunk_type": "table",
                        "text": seg_text,
                        "char_start": clean_start,
                        "char_end": clean_end,
                        "source_url": filing["source_url"],
                        "source_file_sha256": filing["file_sha256"],
                        "canonical_text_path": filing.get("canonical_text_path"),
                        "filing_date": filing.get("filing_date"),
                        "period_end_date": filing.get("period_end_date"),
                        "table_title": table_title,
                    })
                else:
                    # Split large tables safely at line/row boundaries
                    lines = seg_text.split("\n")
                    line_offsets: list[int] = []
                    cur = clean_start
                    for line in lines:
                        line_offsets.append(cur)
                        cur += len(line) + 1  # +1 for newline

                    sub_start_idx = 0
                    while sub_start_idx < len(lines):
                        sub_words = 0
                        sub_end_idx = sub_start_idx
                        while sub_end_idx < len(lines):
                            line_w = len(lines[sub_end_idx].split())
                            if sub_words + line_w > max_words and sub_end_idx > sub_start_idx:
                                break
                            sub_words += line_w
                            sub_end_idx += 1

                        c_start = line_offsets[sub_start_idx]
                        if sub_end_idx < len(lines):
                            c_end = line_offsets[sub_end_idx] - 1
                        else:
                            c_end = clean_end

                        c_text = canonical_text[c_start:c_end]
                        chunk_id = f"{filing['ticker']}_{filing['fiscal_year']}_{clean_form}_{clean_accn}_{chunk_seq:05d}"
                        chunk_seq += 1
                        chunks.append({
                            "chunk_id": chunk_id,
                            "ticker": filing["ticker"],
                            "cik": filing["cik"],
                            "fiscal_year": filing["fiscal_year"],
                            "form": filing["form"],
                            "accession": filing["accession"],
                            "section": sec_name,
                            "section_confidence": sec_conf,
                            "chunk_type": "table",
                            "text": c_text,
                            "char_start": c_start,
                            "char_end": c_end,
                            "source_url": filing["source_url"],
                            "source_file_sha256": filing["file_sha256"],
                            "canonical_text_path": filing.get("canonical_text_path"),
                            "filing_date": filing.get("filing_date"),
                            "period_end_date": filing.get("period_end_date"),
                            "table_title": table_title,
                        })
                        sub_start_idx = sub_end_idx

            else:
                # Narrative text chunking within section
                paragraphs = [p for p in re.split(r"(\n\n+)", seg_text) if p]
                para_spans: list[tuple[int, int]] = []
                cur = clean_start
                for p in paragraphs:
                    p_len = len(p)
                    if not re.match(r"^\n\n+$", p):
                        para_spans.append((cur, cur + p_len))
                    cur += p_len

                if not para_spans:
                    continue

                p_idx = 0
                while p_idx < len(para_spans):
                    p_start, p_end = para_spans[p_idx]
                    p_words = len(canonical_text[p_start:p_end].split())

                    # Split oversized paragraph (> max_words) directly into chunks
                    if p_words > max_words:
                        long_slices = split_long_paragraph(
                            canonical_text,
                            p_start,
                            p_end,
                            max_words=max_words,
                            target_words=target_words,
                            overlap_words=overlap_words,
                        )
                        for c_start, c_end in long_slices:
                            c_text = canonical_text[c_start:c_end]
                            chunk_id = f"{filing['ticker']}_{filing['fiscal_year']}_{clean_form}_{clean_accn}_{chunk_seq:05d}"
                            chunk_seq += 1
                            chunks.append({
                                "chunk_id": chunk_id,
                                "ticker": filing["ticker"],
                                "cik": filing["cik"],
                                "fiscal_year": filing["fiscal_year"],
                                "form": filing["form"],
                                "accession": filing["accession"],
                                "section": sec_name,
                                "section_confidence": sec_conf,
                                "chunk_type": "text",
                                "text": c_text,
                                "char_start": c_start,
                                "char_end": c_end,
                                "source_url": filing["source_url"],
                                "source_file_sha256": filing["file_sha256"],
                                "canonical_text_path": filing.get("canonical_text_path"),
                                "filing_date": filing.get("filing_date"),
                                "period_end_date": filing.get("period_end_date"),
                                "table_title": None,
                            })
                        p_idx += 1
                        continue

                    # Accumulate normal paragraphs up to max_words
                    c_start = p_start
                    cur_end_p = p_idx
                    cur_words = 0

                    while cur_end_p < len(para_spans):
                        nxt_start, nxt_end = para_spans[cur_end_p]
                        pw = len(canonical_text[nxt_start:nxt_end].split())
                        if cur_words + pw > max_words and cur_end_p > p_idx:
                            break
                        if pw > max_words and cur_end_p > p_idx:
                            break
                        cur_words += pw
                        cur_end_p += 1

                    c_end = para_spans[cur_end_p - 1][1]
                    c_text = canonical_text[c_start:c_end]

                    chunk_id = f"{filing['ticker']}_{filing['fiscal_year']}_{clean_form}_{clean_accn}_{chunk_seq:05d}"
                    chunk_seq += 1
                    chunks.append({
                        "chunk_id": chunk_id,
                        "ticker": filing["ticker"],
                        "cik": filing["cik"],
                        "fiscal_year": filing["fiscal_year"],
                        "form": filing["form"],
                        "accession": filing["accession"],
                        "section": sec_name,
                        "section_confidence": sec_conf,
                        "chunk_type": "text",
                        "text": c_text,
                        "char_start": c_start,
                        "char_end": c_end,
                        "source_url": filing["source_url"],
                        "source_file_sha256": filing["file_sha256"],
                        "canonical_text_path": filing.get("canonical_text_path"),
                        "filing_date": filing.get("filing_date"),
                        "period_end_date": filing.get("period_end_date"),
                        "table_title": None,
                    })

                    if cur_end_p >= len(para_spans):
                        break

                    # Advance with ~50 words overlap among normal paragraphs
                    overlap_accum = 0
                    step_back = 0
                    for back_idx in range(cur_end_p - 1, p_idx, -1):
                        pw = len(canonical_text[para_spans[back_idx][0]:para_spans[back_idx][1]].split())
                        overlap_accum += pw
                        step_back += 1
                        if overlap_accum >= overlap_words:
                            break

                    p_idx = max(p_idx + 1, cur_end_p - step_back)

    # Sanity check: exact offset verification for all chunks
    for ch in chunks:
        actual_slice = canonical_text[ch["char_start"]:ch["char_end"]]
        if actual_slice != ch["text"]:
            raise ValueError(
                f"Offset invariant failed for chunk {ch['chunk_id']}: slice length {len(actual_slice)} != text length {len(ch['text'])}."
            )

    return chunks


def build_filing_corpus(
    processed_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Process all validated annual filings and create data/processed/chunks.jsonl.

    Stages newly generated canonical text in a temporary directory and uses
    atomic replacement directly for chunks.jsonl only after all filings succeed.
    """
    proc_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    proc_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = proc_dir / "manifest.json"
    validated_filings = load_validated_filings(manifest_path=manifest_file, raw_dir=raw_dir)

    canonical_dir = proc_dir / "text"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # Temporary staging paths
    tmp_canonical_dir = proc_dir / "text_tmp"
    tmp_canonical_dir.mkdir(parents=True, exist_ok=True)
    tmp_chunks_path = proc_dir / "chunks.jsonl.tmp"
    final_chunks_path = proc_dir / "chunks.jsonl"

    text_chunks_count = 0
    table_chunks_count = 0
    seen_chunk_ids: set[str] = set()

    conf_counts = {"high": 0, "low": 0, "unknown": 0}

    try:
        with tmp_chunks_path.open("w", encoding="utf-8") as out_f:
            for filing in validated_filings:
                local_file = Path(filing["local_path"])
                html_bytes = local_file.read_bytes()

                canonical_text = extract_visible_text(html_bytes)

                clean_accn = filing["accession"].replace("-", "")
                clean_form = filing["form"].replace("-", "")
                txt_filename = f"{filing['ticker']}_{filing['fiscal_year']}_{clean_form}_{clean_accn}.txt"

                # Stage text file in temporary directory
                staged_txt_path = tmp_canonical_dir / txt_filename
                staged_txt_path.write_text(canonical_text, encoding="utf-8")

                # Store the final canonical text path in chunk metadata
                final_txt_path = canonical_dir / txt_filename
                filing["canonical_text_path"] = str(final_txt_path.as_posix())

                sections = extract_sections(canonical_text)
                chunks = make_chunks(filing, canonical_text, sections)

                for chunk in chunks:
                    cid = chunk["chunk_id"]
                    if cid in seen_chunk_ids:
                        raise ValueError(f"Duplicate chunk ID generated: '{cid}'.")
                    seen_chunk_ids.add(cid)

                    if chunk["chunk_type"] == "table":
                        table_chunks_count += 1
                    else:
                        text_chunks_count += 1

                    conf = chunk.get("section_confidence", "unknown")
                    conf_counts[conf] = conf_counts.get(conf, 0) + 1

                    out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # Move staged canonical text files into the final directory after all filings succeed
        for staged_file in tmp_canonical_dir.iterdir():
            if staged_file.is_file():
                dest_file = canonical_dir / staged_file.name
                staged_file.replace(dest_file)

        # Atomic replacement directly from the completed temporary file (no prior deletion)
        tmp_chunks_path.replace(final_chunks_path)

    finally:
        # Clean up temporary files after success or failure
        if tmp_chunks_path.is_file():
            tmp_chunks_path.unlink(missing_ok=True)
        if tmp_canonical_dir.is_dir():
            for leftover in tmp_canonical_dir.iterdir():
                if leftover.is_file():
                    leftover.unlink(missing_ok=True)
            try:
                tmp_canonical_dir.rmdir()
            except OSError:
                pass

    summary = {
        "filings_expected": len(validated_filings),
        "filings_processed": len(validated_filings),
        "failed_filings": 0,
        "canonical_text_files": len(validated_filings),
        "text_chunks": text_chunks_count,
        "table_chunks": table_chunks_count,
        "total_chunks": text_chunks_count + table_chunks_count,
        "high_confidence_sections": conf_counts.get("high", 0),
        "low_confidence_sections": conf_counts.get("low", 0),
        "unknown_confidence_sections": conf_counts.get("unknown", 0),
        "chunks_file": str(final_chunks_path.as_posix()),
    }
    return summary


def inspect_filing_corpus(
    ticker: str,
    year: int,
    processed_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect processed filing chunks and section metadata for a company and year."""
    proc_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    chunks_path = proc_dir / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Corpus chunks file not found at '{chunks_path}'. Run build first.")

    target_ticker = ticker.strip().upper()
    matching_chunks: list[dict[str, Any]] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("ticker") == target_ticker and chunk.get("fiscal_year") == year:
                matching_chunks.append(chunk)

    if not matching_chunks:
        raise ValueError(f"No processed chunks found for ticker '{target_ticker}' and fiscal year {year}.")

    sections_found = sorted(list({c.get("section") for c in matching_chunks if c.get("section")}))
    text_count = sum(1 for c in matching_chunks if c.get("chunk_type") == "text")
    table_count = sum(1 for c in matching_chunks if c.get("chunk_type") == "table")

    first_chunk = matching_chunks[0]
    sample_text = first_chunk.get("text", "")[:200]

    return {
        "ticker": target_ticker,
        "fiscal_year": year,
        "accession": first_chunk.get("accession"),
        "source_url": first_chunk.get("source_url"),
        "source_file_sha256": first_chunk.get("source_file_sha256"),
        "total_chunks": len(matching_chunks),
        "text_chunks": text_count,
        "table_chunks": table_count,
        "sections": sections_found,
        "sample_chunk_id": first_chunk.get("chunk_id"),
        "sample_excerpt": sample_text,
    }


def main() -> None:
    """CLI entrypoint for building and inspecting the Form 10-K document corpus."""
    parser = argparse.ArgumentParser(description="Fyntrace SEC Form 10-K Document Processing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_parser = subparsers.add_parser("build", help="Process filings and build chunks.jsonl")
    build_parser.add_argument("--processed-dir", default=None, help="Directory for processed data")
    build_parser.add_argument("--raw-dir", default=None, help="Directory for raw data")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect processed content for a filing")
    inspect_parser.add_argument("--ticker", required=True, help="Company ticker symbol (e.g. MSFT)")
    inspect_parser.add_argument("--year", type=int, required=True, help="Fiscal year (e.g. 2024)")
    inspect_parser.add_argument("--processed-dir", default=None, help="Directory for processed data")

    args = parser.parse_args()

    if args.command == "build":
        summary = build_filing_corpus(processed_dir=args.processed_dir, raw_dir=args.raw_dir)
        print("Filings processed:", summary["filings_processed"])
        print("Text chunks:", summary["text_chunks"])
        print("Table chunks:", summary["table_chunks"])
        print("Total chunks:", summary["total_chunks"])
        print("Processing failures:", summary["failed_filings"])

    elif args.command == "inspect":
        info = inspect_filing_corpus(ticker=args.ticker, year=args.year, processed_dir=args.processed_dir)
        print(f"Company: {info['ticker']}")
        print(f"Fiscal year: {info['fiscal_year']}")
        print(f"Accession: {info['accession']}")
        print(f"Total chunks: {info['total_chunks']} (text: {info['text_chunks']}, table: {info['table_chunks']})")
        print(f"Sections detected: {', '.join(info['sections'])}")
        print(f"Sample chunk ID: {info['sample_chunk_id']}")
        print(f"Sample excerpt: {repr(info['sample_excerpt'])}")


if __name__ == "__main__":
    main()
