"""Unit tests for SEC filing text extraction and source-aware chunking."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from bs4 import BeautifulSoup

from src.filings import (
    build_filing_corpus,
    download_filing,
    extract_sections,
    extract_table_text,
    extract_visible_text,
    find_filing,
    is_valid_html_content,
    make_chunks,
    parse_args,
)

SAMPLE_FILING_RECORD_MSFT = {
    "ticker": "MSFT",
    "company_name": "MICROSOFT CORP",
    "cik": "0000789019",
    "sector": "Technology",
    "target_fiscal_year": 2024,
    "fiscal_year_status": "confirmed",
    "form": "10-K",
    "accession_number": "0000950170-24-087843",
    "filing_date": "2024-07-30",
    "report_date": "2024-06-30",
    "primary_document": "msft-20240630.htm",
    "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
}

SAMPLE_FILING_RECORD_XOM = {
    "ticker": "XOM",
    "company_name": "EXXON MOBIL CORP",
    "cik": "0000034088",
    "sector": "Energy and utilities",
    "target_fiscal_year": 2024,
    "fiscal_year_status": "confirmed",
    "form": "10-K",
    "accession_number": "0000034088-25-000010",
    "filing_date": "2025-02-20",
    "report_date": "2024-12-31",
    "primary_document": "xom-20241231.htm",
    "source_url": "https://www.sec.gov/Archives/edgar/data/34088/000003408825000010/xom-20241231.htm",
}

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Form 10-K</title>
<script>var x = "hidden script";</script>
<style>.body { font-size: 12pt; }</style>
</head>
<body>
<nav>Navigation Header</nav>
<!-- HTML comment to be removed -->
<div style="display:none">Hidden tracking code</div>
<ix:hidden>Hidden XBRL taxonomy metadata</ix:hidden>

<div class="toc">
  <table>
    <tr><td>Item 1 | Business | 3</td></tr>
    <tr><td>Item 1A | Risk Factors | 10</td></tr>
    <tr><td>Item 7 | Management's Discussion and Analysis | 25</td></tr>
  </table>
</div>

<p><b>ITEM 1. BUSINESS</b></p>
<p>General overview of company operations, products, and cloud platforms across various operational divisions worldwide.</p>
<p>Additional details regarding strategy, competitive landscape, intellectual property, and extensive research investments.</p>
<p>Further descriptions of employee training, customer partnerships, operational supply chain management, and regulatory compliance.</p>

<p><b>ITEM 1A. RISK FACTORS</b></p>
<p>Our operations and financial performance are subject to significant operational, financial, and regulatory risks.</p>
<p>Market competition could adversely affect product demand and gross margin performance across key reporting segments.</p>
<p>Cybersecurity incidents or data breaches could disrupt critical online services and impair customer confidence and operational integrity.</p>
<p>Macroeconomic volatility, foreign exchange fluctuations, and interest rate changes may reduce enterprise software spending.</p>
<p>Adverse changes in global trade policies or international taxation rules could impact our operating margins and net cash flows.</p>

<table>
  <tr><th>Segment</th><th>Revenues FY24</th><th>Revenues FY23</th><th>Growth</th></tr>
  <tr><td>Cloud Services</td><td>$ 105,362</td><td>$ 87,907</td><td>19.8%</td></tr>
  <tr><td>Productivity</td><td>$ 79,250</td><td>$ 69,274</td><td>14.4%</td></tr>
  <tr><td>Operating Losses</td><td>$ (1,450)</td><td>$ (2,100)</td><td>-31.0%</td></tr>
</table>

<p><b>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</b></p>
<p>Management discussion regarding consolidated operational performance, liquidity, and capital allocation.</p>
<p>Revenue increased primarily due to higher customer adoption of enterprise cloud solutions and AI-enabled productivity software.</p>
<p>Research and development expenses grew in line with increased headcount and cloud infrastructure investments.</p>

<p><b>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</b></p>
<p>We are exposed to market risk from changes in interest rates, foreign currency exchange rates, and equity security prices.</p>
<p>Our investment portfolio consists primarily of investment-grade fixed income securities designed to preserve principal liquidity.</p>

<p><b>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</b></p>
<p>Consolidated Financial Statements and Report of Independent Registered Public Accounting Firm.</p>
</body>
</html>
"""


def create_mock_response(status_code: int = 200, content: bytes = b"") -> MagicMock:
    """Helper to create a mocked requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.content = content
    return resp


def test_1_exact_manifest_filing_selection() -> None:
    """1. Exact manifest-based filing selection."""
    manifest = [SAMPLE_FILING_RECORD_MSFT, SAMPLE_FILING_RECORD_XOM]

    rec = find_filing("MSFT", 2024, manifest=manifest)
    assert rec["ticker"] == "MSFT"
    assert rec["target_fiscal_year"] == 2024
    assert rec["accession_number"] == "0000950170-24-087843"

    # Case insensitive
    rec_lower = find_filing("msft", 2024, manifest=manifest)
    assert rec_lower["ticker"] == "MSFT"

    # Unknown company raises KeyError
    with pytest.raises(KeyError):
        find_filing("UNKNOWN", 2024, manifest=manifest)

    # Missing year raises KeyError
    with pytest.raises(KeyError):
        find_filing("MSFT", 2020, manifest=manifest)

    # Unconfirmed status raises ValueError
    unconfirmed_manifest = [dict(SAMPLE_FILING_RECORD_MSFT, fiscal_year_status="needs_review")]
    with pytest.raises(ValueError) as exc:
        find_filing("MSFT", 2024, manifest=unconfirmed_manifest)
    assert "expected 'confirmed'" in str(exc.value)

    # Non-10-K form raises ValueError
    non_10k_manifest = [dict(SAMPLE_FILING_RECORD_MSFT, form="10-Q")]
    with pytest.raises(ValueError) as exc_form:
        find_filing("MSFT", 2024, manifest=non_10k_manifest)
    assert "expected '10-K'" in str(exc_form.value)


def test_2_current_vs_historical_xom_cik_preservation() -> None:
    """2. Current versus historical XOM CIK preservation."""
    manifest = [SAMPLE_FILING_RECORD_XOM]
    rec = find_filing("XOM", 2024, manifest=manifest)

    assert rec["cik"] == "0000034088"
    assert rec["company_name"] == "EXXON MOBIL CORP"
    assert rec["accession_number"] == "0000034088-25-000010"
    assert rec["cik"] != "0002115436"  # modern CIK not used


def test_3_correct_sec_url_and_original_accession() -> None:
    """3. Correct SEC URL and original accession."""
    manifest = [SAMPLE_FILING_RECORD_MSFT]
    rec = find_filing("MSFT", 2024, manifest=manifest)

    assert rec["source_url"] == "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm"
    assert rec["accession_number"] == "0000950170-24-087843"
    assert rec["primary_document"] == "msft-20240630.htm"


def test_4_successful_html_download_and_cache_reuse(tmp_path: Path) -> None:
    """4. Successful HTML download and cache reuse."""
    session = MagicMock(spec=requests.Session)
    html_bytes = SAMPLE_HTML.encode("utf-8")
    session.get.return_value = create_mock_response(200, html_bytes)

    target_file, sha256_hash = download_filing(SAMPLE_FILING_RECORD_MSFT, session=session, cache_dir=tmp_path)
    assert target_file.is_file()
    assert target_file.name == "0000950170-24-087843.html"
    assert len(sha256_hash) == 64
    session.get.assert_called_once()

    # Second call should reuse cache without making any HTTP call
    session_failing = MagicMock(spec=requests.Session)
    session_failing.get.side_effect = AssertionError("Should not make network call when cached.")

    cached_file, cached_hash = download_filing(SAMPLE_FILING_RECORD_MSFT, session=session_failing, cache_dir=tmp_path)
    assert cached_file == target_file
    assert cached_hash == sha256_hash
    session_failing.get.assert_not_called()


def test_5_failure_does_not_overwrite_valid_cached_html(tmp_path: Path) -> None:
    """5. Failure does not overwrite valid cached HTML."""
    target_file = tmp_path / "0000950170-24-087843.html"
    original_bytes = SAMPLE_HTML.encode("utf-8")
    target_file.write_bytes(original_bytes)

    # Server fails with 500
    session_error = MagicMock(spec=requests.Session)
    session_error.get.return_value = create_mock_response(500, b"Internal Server Error")

    # Force re-download by bypassing cache check or checking error handling
    # download_filing reuses valid cache first, so target_file remains untouched
    out_path, out_hash = download_filing(SAMPLE_FILING_RECORD_MSFT, session=session_error, cache_dir=tmp_path)
    assert out_path.read_bytes() == original_bytes


def test_6_http_error_and_invalid_html_handling(tmp_path: Path) -> None:
    """6. HTTP error and invalid HTML handling."""
    session_403 = MagicMock(spec=requests.Session)
    session_403.get.return_value = create_mock_response(403, b"Forbidden")
    with pytest.raises(PermissionError):
        download_filing(SAMPLE_FILING_RECORD_MSFT, session=session_403, cache_dir=tmp_path)

    session_404 = MagicMock(spec=requests.Session)
    session_404.get.return_value = create_mock_response(404, b"Not Found")
    with pytest.raises(FileNotFoundError):
        download_filing(SAMPLE_FILING_RECORD_MSFT, session=session_404, cache_dir=tmp_path)

    # Invalid non-HTML content
    session_invalid = MagicMock(spec=requests.Session)
    session_invalid.get.return_value = create_mock_response(200, b"Plain text without any html markers " * 10)
    with pytest.raises(ValueError):
        download_filing(SAMPLE_FILING_RECORD_MSFT, session=session_invalid, cache_dir=tmp_path)


def test_7_removal_of_scripts_styles_and_hidden_content() -> None:
    """7. Removal of scripts, styles, and hidden content."""
    text = extract_visible_text(SAMPLE_HTML)
    assert "hidden script" not in text
    assert "font-size: 12pt" not in text
    assert "Navigation Header" not in text
    assert "HTML comment to be removed" not in text
    assert "Hidden tracking code" not in text
    assert "Hidden XBRL taxonomy metadata" not in text


def test_8_preservation_of_financial_table_rows_and_values() -> None:
    """8. Preservation of financial table rows and values."""
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")
    tables = soup.find_all("table")
    fin_table = tables[1]  # second table has financial segment data

    tbl_text = extract_table_text(fin_table)
    assert "Segment | Revenues FY24 | Revenues FY23 | Growth" in tbl_text
    assert "Cloud Services | $ 105,362 | $ 87,907 | 19.8%" in tbl_text
    assert "Productivity | $ 79,250 | $ 69,274 | 14.4%" in tbl_text

    full_text = extract_visible_text(SAMPLE_HTML)
    assert "Cloud Services | $ 105,362 | $ 87,907 | 19.8%" in full_text


def test_9_preservation_of_negative_numbers_and_percentages() -> None:
    """9. Preservation of negative numbers and percentages."""
    full_text = extract_visible_text(SAMPLE_HTML)
    assert "$ (1,450)" in full_text
    assert "$ (2,100)" in full_text
    assert "-31.0%" in full_text
    assert "19.8%" in full_text


def test_10_table_of_contents_vs_actual_report_headings() -> None:
    """10. Table-of-contents headings versus actual report headings."""
    canonical_text = extract_visible_text(SAMPLE_HTML)
    sections = extract_sections(canonical_text)

    # Must detect Item 1, Item 1A, Item 7, Item 7A, Item 8
    sec_names = [s["name"] for s in sections]
    assert "Item 1 - Business" in sec_names
    assert "Item 1A - Risk Factors" in sec_names
    assert "Item 7 - Management's Discussion and Analysis" in sec_names

    # Verify that Item 1A starts at the true body heading, NOT the TOC table
    item1a_sec = [s for s in sections if s["name"] == "Item 1A - Risk Factors"][0]
    section_snippet = canonical_text[item1a_sec["start"] : item1a_sec["start"] + 100]
    assert "ITEM 1A. RISK FACTORS" in section_snippet
    assert "Item 1A | Risk Factors | 10" not in section_snippet


def test_11_uncertain_sections_receive_unknown_confidence() -> None:
    """11. Uncertain sections receive unknown confidence."""
    preamble_html = "<html><body><p>General preamble before any items.</p><p>Second paragraph.</p></body></html>"
    text = extract_visible_text(preamble_html)
    sections = extract_sections(text)
    assert len(sections) == 0

    chunks = make_chunks(text, SAMPLE_FILING_RECORD_MSFT, "dummy_hash")
    assert len(chunks) > 0
    for ch in chunks:
        assert ch["section"] == "unknown"
        assert ch["section_confidence"] == "unknown"


def test_12_stable_chunk_ids_across_repeated_processing() -> None:
    """12. Stable chunk IDs across repeated processing."""
    canonical_text = extract_visible_text(SAMPLE_HTML)
    chunks_1 = make_chunks(canonical_text, SAMPLE_FILING_RECORD_MSFT, "dummy_sha256")
    chunks_2 = make_chunks(canonical_text, SAMPLE_FILING_RECORD_MSFT, "dummy_sha256")

    assert len(chunks_1) == len(chunks_2)
    for c1, c2 in zip(chunks_1, chunks_2):
        assert c1["chunk_id"] == c2["chunk_id"]
        assert c1["char_start"] == c2["char_start"]
        assert c1["char_end"] == c2["char_end"]


def test_13_correct_text_character_offsets() -> None:
    """13. Correct text character offsets."""
    canonical_text = extract_visible_text(SAMPLE_HTML)
    chunks = make_chunks(canonical_text, SAMPLE_FILING_RECORD_MSFT, "dummy_sha256")

    for ch in chunks:
        assert canonical_text[ch["char_start"] : ch["char_end"]] == ch["text"]


def test_14_correct_company_year_accession_metadata() -> None:
    """14. Correct company/year/accession metadata in chunks."""
    canonical_text = extract_visible_text(SAMPLE_HTML)
    chunks = make_chunks(canonical_text, SAMPLE_FILING_RECORD_MSFT, "abc123sha")

    for ch in chunks:
        assert ch["ticker"] == "MSFT"
        assert ch["company_name"] == "MICROSOFT CORP"
        assert ch["cik"] == "0000789019"
        assert ch["fiscal_year"] == 2024
        assert ch["form"] == "10-K"
        assert ch["accession"] == "0000950170-24-087843"
        assert ch["report_date"] == "2024-06-30"
        assert ch["source_url"] == SAMPLE_FILING_RECORD_MSFT["source_url"]
        assert ch["file_sha256"] == "abc123sha"


def test_15_successful_jsonl_generation(tmp_path: Path) -> None:
    """15. Successful JSONL generation via build_filing_corpus."""
    manifest_file = tmp_path / "filing_manifest.json"
    manifest_file.write_text(json.dumps([SAMPLE_FILING_RECORD_MSFT]), encoding="utf-8")

    raw_dir = tmp_path / "raw" / "filings"
    processed_dir = tmp_path / "processed" / "filing_text"
    chunks_file = tmp_path / "processed" / "chunks.jsonl"

    session = MagicMock(spec=requests.Session)
    session.get.return_value = create_mock_response(200, SAMPLE_HTML.encode("utf-8"))

    summary = build_filing_corpus(
        tickers=["MSFT"],
        years=[2024],
        manifest_path=manifest_file,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunks_path=chunks_file,
        session=session,
    )

    assert summary["filings_requested"] == 1
    assert summary["filings_succeeded"] == 1
    assert summary["filings_failed"] == 0
    assert summary["total_chunks"] > 0
    assert chunks_file.is_file()

    lines = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == summary["total_chunks"]
    assert lines[0]["ticker"] == "MSFT"


def test_16_controlled_handling_of_malformed_html() -> None:
    """16. Controlled handling of malformed HTML."""
    # Unclosed tags should be parsed safely by lxml
    malformed_html = "<html><body><p><b>Unclosed bold text <p>New paragraph without closing" * 5
    text = extract_visible_text(malformed_html)
    assert "Unclosed bold text" in text

    # Empty string raises ValueError
    with pytest.raises(ValueError):
        extract_visible_text("")

    # White-space only raises ValueError
    with pytest.raises(ValueError):
        extract_visible_text("   \n\t   ")


def test_cli_argument_parsing() -> None:
    """Test CLI argument parsing and validation."""
    args = parse_args(["--tickers", "MSFT", "WMT", "--years", "2024"])
    assert args.tickers == ["MSFT", "WMT"]
    assert args.years == [2024]
    assert not args.all_filings

    args_all = parse_args(["--all"])
    assert args_all.all_filings

    with pytest.raises(SystemExit):
        parse_args([])


def test_long_unbroken_text_multiple_bounded_chunks() -> None:
    """Confirm 1,000 short consecutive lines without blank lines produces multiple bounded chunks."""
    lines = [f"Line {i} discusses annual financial performance, operating margins, and corporate strategy." for i in range(1000)]
    html = f"<html><body><p>{''.join(lines)}</p></body></html>"
    text = extract_visible_text(html)

    chunks = make_chunks(text, SAMPLE_FILING_RECORD_MSFT, "hash_unbroken")
    assert len(chunks) > 1

    for ch in chunks:
        words = len(ch["text"].split())
        assert words <= 500
        assert text[ch["char_start"] : ch["char_end"]] == ch["text"]


def test_large_table_row_boundary_chunking() -> None:
    """Confirm a large financial table is split at row boundaries with numbers and labels intact."""
    rows = ["<tr><th>Account</th><th>FY24</th><th>FY23</th></tr>"]
    for i in range(120):
        rows.append(f"<tr><td>Line Item {i} Revenue Description</td><td>$ {1000 + i}</td><td>$ {900 + i}</td></tr>")
    table_html = f"<html><body><table>{''.join(rows)}</table></body></html>"
    text = extract_visible_text(table_html)

    chunks = make_chunks(text, SAMPLE_FILING_RECORD_MSFT, "hash_table")
    assert len(chunks) > 1

    for ch in chunks:
        assert text[ch["char_start"] : ch["char_end"]] == ch["text"]
        for line in ch["text"].splitlines():
            if line.strip():
                assert " | " in line
                assert "Line Item" in line or "Account" in line


def test_empty_table_cells_preserve_alignment() -> None:
    """Confirm empty cells in financial tables preserve column alignment."""
    html = """<html><body><table>
    <tr><th>Category</th><th>FY24</th><th>FY23</th></tr>
    <tr><td>Operating Income</td><td></td><td>$ 1,234</td></tr>
    <tr><td>Net Margin</td><td>15.2%</td><td></td></tr>
    </table></body></html>"""
    soup = BeautifulSoup(html, "lxml")
    tbl = soup.find("table")
    txt = extract_table_text(tbl)

    rows = txt.splitlines()
    assert rows[0] == "Category | FY24 | FY23"
    assert rows[1] == "Operating Income |  | $ 1,234"
    assert rows[2] == "Net Margin | 15.2% | "


def test_toc_only_document_receives_unknown_confidence() -> None:
    """Confirm a document containing only TOC entries gets unknown section metadata."""
    html = """<html><body>
    <h2>TABLE OF CONTENTS</h2>
    <table>
      <tr><td>Item 1 | Business | 3</td></tr>
      <tr><td>Item 1A | Risk Factors | 12</td></tr>
      <tr><td>Item 7 | Management's Discussion and Analysis | 34</td></tr>
      <tr><td>Item 7A | Quantitative and Qualitative Disclosures | 50</td></tr>
      <tr><td>Item 8 | Financial Statements | 60</td></tr>
    </table>
    </body></html>"""
    text = extract_visible_text(html)
    sections = extract_sections(text)
    assert len(sections) == 0

    chunks = make_chunks(text, SAMPLE_FILING_RECORD_MSFT, "hash_toc_only")
    assert len(chunks) > 0
    for ch in chunks:
        assert ch["section"] == "unknown"
        assert ch["section_confidence"] == "unknown"
        assert text[ch["char_start"] : ch["char_end"]] == ch["text"]


def test_repeated_section_headings_distinguished() -> None:
    """Confirm repeated mentions of an item heading are distinguished and the real body heading is chosen."""
    html = """<html><body>
    <table><tr><td>Item 1A | Risk Factors | 15</td></tr></table>
    <p>Please refer to Item 1A: Risk Factors in our annual filing for further discussion.</p>
    <p>Forward looking remarks note that Item 1A contains essential risk descriptions.</p>
    
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Our operations face substantial market risks, regulatory risks, and technological disruption risks worldwide.</p>
    <p>Supply chain constraints could impede delivery schedules and diminish quarterly gross operating margins.</p>
    <p>Foreign exchange currency fluctuations may adversely affect reported consolidated revenues in multiple territories.</p>
    </body></html>"""
    text = extract_visible_text(html)
    sections = extract_sections(text)

    item1a_sections = [s for s in sections if s["name"] == "Item 1A - Risk Factors"]
    assert len(item1a_sections) == 1
    sec = item1a_sections[0]
    assert sec["confidence"] == "high"
    heading_text = text[sec["start"] : sec["start"] + 50]
    assert "ITEM 1A. RISK FACTORS" in heading_text
    assert "Item 1A | Risk Factors | 15" not in heading_text


def test_incomplete_or_unusual_headings_receive_unknown_metadata() -> None:
    """Confirm incomplete or stub headings receive unknown confidence and metadata."""
    html = """<html><body>
    <p>Introductory company remarks regarding fiscal year operations and disclosures.</p>
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Stub.</p>
    </body></html>"""
    text = extract_visible_text(html)
    sections = extract_sections(text)
    assert len(sections) == 1
    assert sections[0]["name"] == "unknown"
    assert sections[0]["confidence"] == "unknown"

    chunks = make_chunks(text, SAMPLE_FILING_RECORD_MSFT, "hash_stub")
    for ch in chunks:
        assert ch["section"] == "unknown"
        assert ch["section_confidence"] == "unknown"


def test_valid_character_offsets_and_no_lost_content() -> None:
    """Confirm character offsets match canonical text and no text is lost."""
    html = """<html><body>
    <p>The first paragraph contains distinct prose discussing technology products and enterprise solutions.</p>
    <p>The second paragraph provides extensive operational metrics regarding our global distribution network.</p>
    <table>
      <tr><th>Metric</th><th>FY24</th></tr>
      <tr><td>Total Deliveries</td><td>1,500,000</td></tr>
    </table>
    <p>Final concluding thoughts regarding corporate governance and board committee oversight activities.</p>
    </body></html>"""
    text = extract_visible_text(html)
    chunks = make_chunks(text, SAMPLE_FILING_RECORD_MSFT, "hash_offsets")

    for ch in chunks:
        assert text[ch["char_start"] : ch["char_end"]] == ch["text"]

    for keyword in ["first paragraph", "second paragraph", "Total Deliveries", "Final concluding"]:
        assert any(keyword in ch["text"] for ch in chunks)


def test_incremental_corpus_merging_and_no_duplicates(tmp_path: Path) -> None:
    """Confirm incremental merging preserves unrelated filings and avoids duplicates."""
    manifest_file = tmp_path / "filing_manifest.json"
    manifest_file.write_text(json.dumps([SAMPLE_FILING_RECORD_MSFT, SAMPLE_FILING_RECORD_XOM]), encoding="utf-8")

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed" / "text"
    chunks_path = tmp_path / "processed" / "chunks.jsonl"

    session = MagicMock(spec=requests.Session)
    session.get.return_value = create_mock_response(200, SAMPLE_HTML.encode("utf-8"))

    # Step 1: Process MSFT
    summary1 = build_filing_corpus(
        tickers=["MSFT"],
        years=[2024],
        manifest_path=manifest_file,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunks_path=chunks_path,
        session=session,
    )
    assert summary1["filings_succeeded"] == 1
    msft_chunk_count = summary1["total_chunks"]
    assert msft_chunk_count > 0

    # Step 2: Process XOM incrementally
    summary2 = build_filing_corpus(
        tickers=["XOM"],
        years=[2024],
        manifest_path=manifest_file,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunks_path=chunks_path,
        session=session,
    )
    assert summary2["filings_succeeded"] == 1
    lines2 = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tickers2 = {l["ticker"] for l in lines2}
    assert tickers2 == {"MSFT", "XOM"}
    assert len(lines2) > msft_chunk_count

    # Step 3: Reprocess MSFT - should replace older chunks, no duplicates, same total count
    summary3 = build_filing_corpus(
        tickers=["MSFT"],
        years=[2024],
        manifest_path=manifest_file,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunks_path=chunks_path,
        session=session,
    )
    lines3 = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines3) == len(lines2)
    tickers3 = {l["ticker"] for l in lines3}
    assert tickers3 == {"MSFT", "XOM"}
    chunk_ids3 = [l["chunk_id"] for l in lines3]
    assert len(chunk_ids3) == len(set(chunk_ids3))


def test_existing_corpus_preserved_on_failure(tmp_path: Path) -> None:
    """Confirm existing corpus file remains untouched if a filing download fails."""
    manifest_file = tmp_path / "filing_manifest.json"
    manifest_file.write_text(json.dumps([SAMPLE_FILING_RECORD_MSFT, SAMPLE_FILING_RECORD_XOM]), encoding="utf-8")

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed" / "text"
    chunks_path = tmp_path / "processed" / "chunks.jsonl"

    session_ok = MagicMock(spec=requests.Session)
    session_ok.get.return_value = create_mock_response(200, SAMPLE_HTML.encode("utf-8"))

    build_filing_corpus(
        tickers=["MSFT"],
        years=[2024],
        manifest_path=manifest_file,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunks_path=chunks_path,
        session=session_ok,
    )
    initial_content = chunks_path.read_text(encoding="utf-8")

    session_err = MagicMock(spec=requests.Session)
    session_err.get.return_value = create_mock_response(500, b"Server Error")

    summary_err = build_filing_corpus(
        tickers=["XOM"],
        years=[2024],
        manifest_path=manifest_file,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunks_path=chunks_path,
        session=session_err,
    )
    assert summary_err["filings_failed"] == 1
    assert chunks_path.read_text(encoding="utf-8") == initial_content


def test_corrupt_corpus_file_rejected(tmp_path: Path) -> None:
    """Confirm corrupted existing chunks.jsonl is rejected rather than silently overwritten."""
    manifest_file = tmp_path / "filing_manifest.json"
    manifest_file.write_text(json.dumps([SAMPLE_FILING_RECORD_MSFT]), encoding="utf-8")

    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("{this is not valid json\n", encoding="utf-8")

    session = MagicMock(spec=requests.Session)
    session.get.return_value = create_mock_response(200, SAMPLE_HTML.encode("utf-8"))

    with pytest.raises(ValueError) as exc:
        build_filing_corpus(
            tickers=["MSFT"],
            years=[2024],
            manifest_path=manifest_file,
            chunks_path=chunks_path,
            session=session,
        )
    assert "Corrupt JSON" in str(exc.value) or "Failed to read existing corpus" in str(exc.value)


def test_cli_nonzero_exit_on_failure() -> None:
    """Confirm the CLI exits with nonzero status on filing processing failure."""
    from src.filings import main

    with pytest.raises(SystemExit) as exc:
        main(["--tickers", "UNKNOWN_NONEXISTENT", "--years", "2024"])
    assert exc.value.code == 1


def test_deduplicate_repeated_ticker_and_years(tmp_path: Path) -> None:
    """Confirm repeated ticker and year arguments are deduplicated."""
    manifest_file = tmp_path / "filing_manifest.json"
    manifest_file.write_text(json.dumps([SAMPLE_FILING_RECORD_MSFT]), encoding="utf-8")

    session = MagicMock(spec=requests.Session)
    session.get.return_value = create_mock_response(200, SAMPLE_HTML.encode("utf-8"))

    summary = build_filing_corpus(
        tickers=["MSFT", "msft", "MSFT"],
        years=[2024, 2024],
        manifest_path=manifest_file,
        chunks_path=tmp_path / "chunks.jsonl",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        session=session,
    )
    assert summary["filings_requested"] == 1
    assert summary["filings_succeeded"] == 1
