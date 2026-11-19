"""Offline unit tests for SEC Form 10-K document processing and chunking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import requests

from fyntrace.filings import (
    build_filing_corpus,
    extract_sections,
    extract_table_text,
    extract_visible_text,
    inspect_filing_corpus,
    load_validated_filings,
    make_chunks,
)


@pytest.fixture(autouse=True)
def guard_no_outbound_network(monkeypatch):
    """Network guard to fail tests immediately if an unexpected real network request occurs."""
    def forbidden_send(self, *args, **kwargs):
        raise RuntimeError("Unexpected real network request attempted during offline tests.")

    monkeypatch.setattr(requests.Session, "send", forbidden_send)


def _make_sample_filing_data(tmp_path: Path, ticker="MSFT", year=2024, accession="0000950170-24-087843"):
    """Helper to create a temporary valid filing structure with manifest and raw HTML."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    filings_dir = raw_dir / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    clean_accn = accession.replace("-", "")
    html_file = filings_dir / f"{ticker}_{year}_{clean_accn}.html"

    # Multi-section synthetic 10-K HTML
    sample_html = f"""<!DOCTYPE html>
<html>
<head>
<title>Form 10-K</title>
<style>.hidden {{ display: none; }} body {{ font-family: serif; }}</style>
<script>var x = 123;</script>
</head>
<body>
<div class="hidden" style="display: none;">Invisible secret header</div>
<ix:header><ix:hidden>Technical taxonomy xbrl metadata</ix:hidden></ix:header>
<h1>Table of Contents</h1>
<p>Item 1. Business 5</p>
<p>Item 1A. Risk Factors 12</p>
<p>Item 7. Management Discussion 25</p>
<p>Item 8. Financial Statements 40</p>

<h2>ITEM 1. BUSINESS</h2>
<p>Microsoft Corporation develops software, services, devices, and solutions. Our mission is to empower every person and organization.</p>
<p>We operate globally across Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.</p>

<h2>ITEM 1A. RISK FACTORS</h2>
<p>Our operations face competitive, cybersecurity, and operational risks. Rapid changes in technology may adversely affect results.</p>
<p>We face risks related to strategic investments in AI infrastructure, supply chain dependencies, and regulatory compliance.</p>

<h2>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h2>
<p>Total revenue for fiscal year 2024 increased by 16% to <ix:nonFraction unitRef="USD" decimals="-6">$245,122</ix:nonFraction> million.</p>
<p>Operating income grew 24% to $109,433 million driven by cloud expansion and efficiency improvements.</p>

<table>
  <caption>Consolidated Income Statement</caption>
  <tr><th>Metric</th><th>2024</th><th>2023</th></tr>
  <tr><td>Total Revenue</td><td>$245,122</td><td>$211,915</td></tr>
  <tr><td>Operating Income</td><td>$109,433</td><td>$88,523</td></tr>
  <tr><td>Net Income</td><td>$88,136</td><td>$72,361</td></tr>
</table>

<h2>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h2>
<p>The consolidated financial statements of the Company and subsidiaries are presented below.</p>
<p>Cash and cash equivalents at period end were $75,500 million.</p>
</body>
</html>
""".encode("utf-8")

    html_file.write_bytes(sample_html)
    file_sha = hashlib.sha256(sample_html).hexdigest()

    manifest_data = {
        "company_identities": [
            {"ticker": ticker, "cik": "0000789019", "name": "Microsoft Corporation", "sector": "Technology"},
        ],
        "filings": [
            {
                "ticker": ticker,
                "fiscal_year": year,
                "status": "downloaded",
                "form": "10-K",
                "accession": accession,
                "local_path": str(html_file.as_posix()),
                "source_url": f"https://www.sec.gov/Archives/edgar/data/789019/{clean_accn}/msft-20240630.htm",
                "file_sha256": file_sha,
                "filing_date": "2024-07-30",
                "period_end_date": "2024-06-30",
                "primary_document": "msft-20240630.htm",
                "selected_metrics": ["revenue", "net_income"],
                "unavailable_metrics": {},
                "errors": [],
            }
        ],
    }
    manifest_path = processed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    return raw_dir, processed_dir, manifest_path, html_file, sample_html


# 1. Normal HTML paragraphs are extracted correctly.
def test_normal_html_paragraphs_extracted_correctly():
    html = b"<html><body><p>First paragraph about operations.</p><p>Second paragraph about revenue.</p></body></html>"
    text = extract_visible_text(html)
    assert "First paragraph about operations." in text
    assert "Second paragraph about revenue." in text
    assert "\n\n" in text


# 2. Scripts and CSS are removed.
def test_scripts_and_css_are_removed():
    html = b"""<html><head>
    <style>body { color: red; } .nav { margin: 0; }</style>
    <script>function doBad() { return false; }</script>
    </head>
    <body>
    <noscript><p>Please enable javascript</p></noscript>
    <p>Legitimate financial paragraph with more than one hundred characters to meet the minimum size requirement of the document extraction function.</p>
    </body></html>"""
    text = extract_visible_text(html)
    assert "body { color: red; }" not in text
    assert "doBad" not in text
    assert "enable javascript" not in text
    assert "Legitimate financial paragraph" in text


# 3. Visible Inline XBRL financial values are preserved.
def test_visible_inline_xbrl_financial_values_preserved():
    html = b"""<html><body>
    <p>Total product revenue reached <ix:nonFraction contextRef="c1" unitRef="USD" decimals="-6">$64,773</ix:nonFraction> million during the year, showing continued growth over previous periods.</p>
    </body></html>"""
    text = extract_visible_text(html)
    assert "$64,773" in text
    assert "Total product revenue reached $64,773 million" in text


# 4. Hidden technical Inline XBRL metadata is excluded.
def test_hidden_technical_inline_xbrl_metadata_excluded():
    html = b"""<html><body>
    <ix:header>
      <ix:hidden>
        <ix:nonNumeric contextRef="c1" name="us-gaap:TechnicalElement">HIDDEN_XBRL_TAG_VALUE</ix:nonNumeric>
      </ix:hidden>
    </ix:header>
    <div style="display:none">Hidden style text that should never appear in visible financial output</div>
    <p>Visible narrative describing corporate financial results for shareholders and analysts over the entire year.</p>
    </body></html>"""
    text = extract_visible_text(html)
    assert "HIDDEN_XBRL_TAG_VALUE" not in text
    assert "Hidden style text" not in text
    assert "Visible narrative describing corporate" in text


# 5. Financial numbers, signs, units, and percentages remain unchanged.
def test_financial_numbers_signs_units_percentages_unchanged():
    html = b"""<html><body>
    <p>Revenue increased by 15.6% to $245,122 million. Net losses were $(3,450) million, reflecting -4.2% change. Cash was $12,345.67 million.</p>
    </body></html>"""
    text = extract_visible_text(html)
    assert "15.6%" in text
    assert "$245,122" in text
    assert "$(3,450)" in text
    assert "-4.2%" in text
    assert "$12,345.67" in text


# 6. Nested HTML does not duplicate text.
def test_nested_html_does_not_duplicate_text():
    html = b"""<html><body>
    <div>
      <div>
        <p>This is a <b>bold <i>and italic</i></b> sentence with length that ensures it satisfies minimum threshold requirements.</p>
      </div>
    </div>
    </body></html>"""
    text = extract_visible_text(html)
    count = text.count("This is a bold and italic sentence")
    assert count == 1


# 7. A normal financial table preserves headings and values.
def test_normal_financial_table_preserves_headings_and_values():
    html = b"""<html><body>
    <p>Below is the summary table of financial operations for the reporting entity across two annual fiscal periods.</p>
    <table>
      <caption>Operating Results</caption>
      <tr><th>Metric</th><th>2024</th><th>2023</th></tr>
      <tr><td>Revenue</td><td>$100,000</td><td>$80,000</td></tr>
      <tr><td>Expenses</td><td>$60,000</td><td>$50,000</td></tr>
    </table>
    </body></html>"""
    text = extract_visible_text(html)
    assert "[TABLE START]" in text
    assert "[TABLE END]" in text
    assert "Table: Operating Results" in text
    assert "Metric | 2024 | 2023" in text
    assert "Revenue | $100,000 | $80,000" in text
    assert "Expenses | $60,000 | $50,000" in text


# 8. A table with merged cells does not invent financial values.
def test_table_with_merged_cells_does_not_invent_financial_values():
    html = b"""<html><body>
    <p>Detailed breakdown of consolidated product segment revenues showing multi-column headers and spanning cells.</p>
    <table>
      <tr><th colspan="2">North America</th><th>Europe</th></tr>
      <tr><td>Product A</td><td>$50,000</td><td>$30,000</td></tr>
    </table>
    </body></html>"""
    text = extract_visible_text(html)
    # The merged header gets the label, spanned cell gets placeholder empty string, no invented numbers
    assert "North America |  | Europe" in text or "North America | | Europe" in text
    assert "Product A | $50,000 | $30,000" in text


# 9. Nested tables do not duplicate rows.
def test_nested_tables_do_not_duplicate_rows():
    html = b"""<html><body>
    <p>Outer table contains an inner table inside one of its cells for visual presentation purposes only.</p>
    <table id="outer">
      <tr>
        <td>Outer Label</td>
        <td>
          <table id="inner">
            <tr><td>Inner Subtotal</td><td>$12,000</td></tr>
          </table>
        </td>
      </tr>
    </table>
    </body></html>"""
    text = extract_visible_text(html)
    # Inner subtotal should appear once, not duplicated as an outer row
    assert text.count("Inner Subtotal") == 1
    assert "$12,000" in text


# 10. Report headings are identified correctly.
def test_report_headings_identified_correctly():
    text = """Cover Page Info
ITEM 1. BUSINESS
Microsoft develops software and provides cloud services globally.
This section contains substantial text describing all lines of business across our three operating segments.
More information about customers, competitors, and human capital resources.

ITEM 1A. RISK FACTORS
Our business operations are subject to numerous risks and uncertainties.
Economic conditions, supply constraints, and cybersecurity threats may materially affect performance.
Additional paragraphs detailing competitive challenges and regulatory developments.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
Management presents this overview of results of operations and financial condition.
Revenue increased significantly due to cloud demand.
Capital expenditures were focused on datacenter infrastructure.

ITEM 8. FINANCIAL STATEMENTS
Consolidated financial statements and accompanying notes are presented herein.
Audit reports and detailed balance sheets follow.
"""
    sections = extract_sections(text)
    section_names = [s["section"] for s in sections]
    assert "Item 1 - Business" in section_names
    assert "Item 1A - Risk Factors" in section_names
    assert "Item 7 - Management's Discussion and Analysis" in section_names
    assert "Item 8 - Financial Statements" in section_names

    # Check confidences
    for s in sections:
        if s["section"] != "Cover and Front Matter":
            assert s["confidence"] in ("high", "low")


# 11. Table-of-contents headings are not mistaken for confirmed body sections.
def test_table_of_contents_not_mistaken_for_confirmed_body_sections():
    text = """Cover Page
Table of Contents
Item 1. Business 5
Item 1A. Risk Factors 15
Item 7. Management Discussion 35
Item 8. Financial Statements 50

Narrative introductory remarks before any section starts.

ITEM 1. BUSINESS
Here is the actual detailed business description covering operations, segments, and workforce.
With multiple substantial paragraphs expanding on corporate strategy and technology platforms.

ITEM 1A. RISK FACTORS
Actual risk disclosure section detailing operational, financial, and regulatory challenges.
Followed by extensive risk discussions spanning multiple pages.
"""
    sections = extract_sections(text)
    # The first real section should start at the actual body Item 1, not the TOC line
    sec_1 = next(s for s in sections if s["section"] == "Item 1 - Business")
    assert sec_1["char_start"] == text.find("ITEM 1. BUSINESS")


# 12. Unknown sections retain their text.
def test_unknown_sections_retain_text():
    text = """Unstructured filing without any standard Item headers.
This document contains general financial disclosures and commentary about annual performance.
Revenue was strong and operations expanded into new geographic regions without disruption.
"""
    sections = extract_sections(text)
    assert len(sections) == 1
    assert sections[0]["section"] == "unknown"
    assert sections[0]["confidence"] == "unknown"
    assert sections[0]["char_start"] == 0
    assert sections[0]["char_end"] == len(text)


# 13. Chunks remain within the requested filing.
def test_chunks_remain_within_requested_filing(tmp_path):
    _, _, manifest_path, _, sample_html = _make_sample_filing_data(tmp_path)
    filings = load_validated_filings(manifest_path, raw_dir=tmp_path / "raw")
    filing = filings[0]
    canonical_text = extract_visible_text(sample_html)
    sections = extract_sections(canonical_text)
    chunks = make_chunks(filing, canonical_text, sections)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["ticker"] == "MSFT"
        assert chunk["fiscal_year"] == 2024
        assert chunk["accession"] == "0000950170-24-087843"
        assert chunk["form"] == "10-K"


# 14. Chunks respect confirmed section boundaries.
def test_chunks_respect_confirmed_section_boundaries(tmp_path):
    _, _, manifest_path, _, sample_html = _make_sample_filing_data(tmp_path)
    filings = load_validated_filings(manifest_path, raw_dir=tmp_path / "raw")
    canonical_text = extract_visible_text(sample_html)
    sections = extract_sections(canonical_text)
    chunks = make_chunks(filings[0], canonical_text, sections)

    # Risk factors chunk should not contain Item 1 Business text
    risk_chunks = [c for c in chunks if c["section"] == "Item 1A - Risk Factors"]
    assert len(risk_chunks) > 0
    for rc in risk_chunks:
        assert "ITEM 1. BUSINESS" not in rc["text"]


# 15. Large narrative passages split without dropping content.
def test_large_narrative_passages_split_without_dropping_content(tmp_path):
    _, _, manifest_path, _, _ = _make_sample_filing_data(tmp_path)
    filings = load_validated_filings(manifest_path, raw_dir=tmp_path / "raw")

    # Generate a long narrative text (>1200 words)
    long_paras = []
    for i in range(15):
        long_paras.append(f"Paragraph {i}: " + "Financial analysis and discussion of corporate activities. " * 15)
    canonical_text = "ITEM 1. BUSINESS\n\n" + "\n\n".join(long_paras)

    sections = extract_sections(canonical_text)
    chunks = make_chunks(filings[0], canonical_text, sections)

    assert len(chunks) > 1
    # Verify that all paragraphs from original text are present across chunks
    for i in range(15):
        assert any(f"Paragraph {i}:" in c["text"] for c in chunks)


# 16. Tables are kept intact or split safely at row boundaries.
def test_tables_kept_intact_or_split_safely_at_row_boundaries(tmp_path):
    _, _, manifest_path, _, sample_html = _make_sample_filing_data(tmp_path)
    filings = load_validated_filings(manifest_path, raw_dir=tmp_path / "raw")
    canonical_text = extract_visible_text(sample_html)
    sections = extract_sections(canonical_text)
    chunks = make_chunks(filings[0], canonical_text, sections)

    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) >= 1
    for tc in table_chunks:
        # Every line in the table chunk should be a complete row
        lines = [line.strip() for line in tc["text"].split("\n") if line.strip()]
        for line in lines:
            if line not in ("[TABLE START]", "[TABLE END]") and not line.startswith("Table:"):
                assert "|" in line


# 17. Chunk IDs are stable across repeated processing.
def test_chunk_ids_are_stable_across_repeated_processing(tmp_path):
    _, _, manifest_path, _, sample_html = _make_sample_filing_data(tmp_path)
    filings = load_validated_filings(manifest_path, raw_dir=tmp_path / "raw")
    canonical_text = extract_visible_text(sample_html)
    sections = extract_sections(canonical_text)

    run1_chunks = make_chunks(filings[0], canonical_text, sections)
    run2_chunks = make_chunks(filings[0], canonical_text, sections)

    assert [c["chunk_id"] for c in run1_chunks] == [c["chunk_id"] for c in run2_chunks]
    assert [c["text"] for c in run1_chunks] == [c["text"] for c in run2_chunks]


# 18. Text offsets match the exact saved canonical text.
def test_text_offsets_match_exact_saved_canonical_text(tmp_path):
    _, _, manifest_path, _, sample_html = _make_sample_filing_data(tmp_path)
    filings = load_validated_filings(manifest_path, raw_dir=tmp_path / "raw")
    canonical_text = extract_visible_text(sample_html)
    sections = extract_sections(canonical_text)
    chunks = make_chunks(filings[0], canonical_text, sections)

    for chunk in chunks:
        start = chunk["char_start"]
        end = chunk["char_end"]
        assert canonical_text[start:end] == chunk["text"]


# 19. Source accession, fiscal year, and SHA-256 are preserved.
def test_source_accession_fiscal_year_and_sha256_preserved(tmp_path):
    _, _, manifest_path, _, sample_html = _make_sample_filing_data(tmp_path)
    filings = load_validated_filings(manifest_path, raw_dir=tmp_path / "raw")
    canonical_text = extract_visible_text(sample_html)
    sections = extract_sections(canonical_text)
    chunks = make_chunks(filings[0], canonical_text, sections)

    expected_sha = filings[0]["file_sha256"]
    for chunk in chunks:
        assert chunk["accession"] == "0000950170-24-087843"
        assert chunk["fiscal_year"] == 2024
        assert chunk["source_file_sha256"] == expected_sha


# 20. A corrupted or missing filing does not overwrite valid processed output.
def test_corrupted_or_missing_filing_does_not_overwrite_valid_output(tmp_path):
    raw_dir, proc_dir, manifest_path, html_file, _ = _make_sample_filing_data(tmp_path)

    # Initial successful build
    summary = build_filing_corpus(processed_dir=proc_dir, raw_dir=raw_dir)
    assert summary["filings_processed"] == 1
    chunks_file = proc_dir / "chunks.jsonl"
    original_chunks_bytes = chunks_file.read_bytes()

    # Corrupt the HTML file
    html_file.write_bytes(b"corrupted contents")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_filing_corpus(processed_dir=proc_dir, raw_dir=raw_dir)

    # Ensure valid original chunks file was preserved
    assert chunks_file.read_bytes() == original_chunks_bytes


# 21. A second run produces identical corpus content for unchanged inputs.
def test_second_run_produces_identical_corpus_content_for_unchanged_inputs(tmp_path):
    raw_dir, proc_dir, manifest_path, _, _ = _make_sample_filing_data(tmp_path)

    build_filing_corpus(processed_dir=proc_dir, raw_dir=raw_dir)
    chunks_path = proc_dir / "chunks.jsonl"
    sha1 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()

    build_filing_corpus(processed_dir=proc_dir, raw_dir=raw_dir)
    sha2 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()

    assert sha1 == sha2


# 22. No unexpected network request occurs.
def test_no_unexpected_network_request_occurs(tmp_path):
    raw_dir, proc_dir, _, _, _ = _make_sample_filing_data(tmp_path)
    # The build_filing_corpus and inspect_filing_corpus run under guard_no_outbound_network fixture
    build_filing_corpus(processed_dir=proc_dir, raw_dir=raw_dir)
    info = inspect_filing_corpus(ticker="MSFT", year=2024, processed_dir=proc_dir)
    assert info["ticker"] == "MSFT"
    assert info["fiscal_year"] == 2024
