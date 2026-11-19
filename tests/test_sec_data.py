"""Unit tests for SEC data collection, validation, and fact normalization."""

import json
from pathlib import Path
import pytest

from fyntrace.sec_data import (
    METRIC_CONFIG,
    REVENUE_CANDIDATE_TAGS,
    SUPPORTED_METRICS,
    build_archive_url,
    collect_data,
    download_filing,
    get_fact,
    get_filing,
    infer_fiscal_year_from_filing,
    normalize_annual_fact,
    normalize_cik,
    resolve_ticker,
    select_annual_filing,
    validate_sec_url,
    validate_user_agent,
)


# --- 1 & 2: Ticker & CIK Resolution ---

def test_valid_ticker_resolves_to_zero_padded_cik():
    mapping = {
        "0": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
        "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }
    resolved = resolve_ticker("MSFT", mapping)
    assert resolved["ticker"] == "MSFT"
    assert resolved["cik"] == "0000789019"
    assert resolved["sector"] == "Technology"
    assert resolved["name"] == "Microsoft Corporation"


def test_unsupported_ticker_rejected():
    mapping = {
        "0": {"cik_str": 123456, "ticker": "DIS", "title": "Walt Disney Co"},
    }
    with pytest.raises(ValueError, match="not supported in the configured company panel"):
        resolve_ticker("DIS", mapping)


def test_historical_registrant_resolution():
    # XOM maps to historical CIK 0000034088
    mapping = {
        "0": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"},
    }
    resolved = resolve_ticker("XOM", mapping)
    assert resolved["ticker"] == "XOM"
    assert resolved["cik"] == "0000034088"
    assert resolved["name"] == "EXXON MOBIL CORP"


# --- 3, 4, 5: Annual 10-K Selection & Fiscal Year Logic ---

def test_10k_selected_instead_of_10q_or_amendments():
    submissions = {
        "cik": "0000789019",
        "fiscalYearEnd": "0630",
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K/A", "10-K", "8-K"],
                "reportDate": ["2024-03-31", "2024-06-30", "2024-06-30", "2024-07-15"],
                "filingDate": ["2024-04-25", "2024-08-15", "2024-07-30", "2024-07-15"],
                "accessionNumber": [
                    "0000950170-24-000001",
                    "0000950170-24-000002",
                    "0000950170-24-087843",
                    "0000950170-24-000003",
                ],
                "primaryDocument": ["q.htm", "ka.htm", "msft-20240630.htm", "8k.htm"],
            }
        },
    }
    selected = select_annual_filing(submissions, fiscal_year=2024)
    assert selected is not None
    assert selected["form"] == "10-K"
    assert selected["accession"] == "0000950170-24-087843"
    assert selected["primary_document"] == "msft-20240630.htm"


def test_filing_date_not_confused_with_report_date():
    # Filing date in 2025, but reportDate in 2024 -> must match FY2024
    submissions = {
        "cik": "0000019617",
        "fiscalYearEnd": "1231",
        "filings": {
            "recent": {
                "form": ["10-K"],
                "reportDate": ["2024-12-31"],
                "filingDate": ["2025-02-14"],
                "accessionNumber": ["0001628280-25-000001"],
                "primaryDocument": ["jpm-20241231.htm"],
            }
        },
    }
    # Query for 2024
    selected_2024 = select_annual_filing(submissions, fiscal_year=2024)
    assert selected_2024 is not None
    assert selected_2024["accession"] == "0001628280-25-000001"

    # Query for 2025 should be None
    selected_2025 = select_annual_filing(submissions, fiscal_year=2025)
    assert selected_2025 is None


def test_non_december_and_52_week_fiscal_years():
    # January year-end (e.g. NVDA FY2024 ending 2024-01-28)
    assert infer_fiscal_year_from_filing("2024-01-28", "2024-02-21", "0131") == 2024

    # June year-end (e.g. MSFT FY2024 ending 2024-06-30)
    assert infer_fiscal_year_from_filing("2024-06-30", "2024-07-30", "0630") == 2024

    # September year-end (e.g. AAPL FY2024 ending 2024-09-28)
    assert infer_fiscal_year_from_filing("2024-09-28", "2024-11-01", "0925") == 2024

    # 52-week year ending in first week of January for December FYE (e.g. JNJ 2023-01-01 is FY2022)
    assert infer_fiscal_year_from_filing("2023-01-01", "2023-02-16", "0103") == 2022
    assert infer_fiscal_year_from_filing("2023-12-31", "2024-02-16", "0103") == 2023


# --- 6, 7, 8, 9, 10, 11, 12: Fact Normalization ---

@pytest.fixture
def sample_filing():
    return {
        "ticker": "MSFT",
        "cik": "0000789019",
        "company_name": "Microsoft Corporation",
        "sector": "Technology",
        "fiscal_year": 2024,
        "accession": "0000950170-24-087843",
        "filing_date": "2024-07-30",
        "report_date": "2024-06-30",
        "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
    }


def test_assets_selects_year_end_instant_value(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "entityName": "Microsoft Corporation",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            # Quarterly value
                            {"end": "2024-03-31", "val": 484275000000, "fy": 2024, "form": "10-Q", "accn": "0000950170-24-000010"},
                            # Year-end value in 10-K
                            {"end": "2024-06-30", "val": 512163000000, "fy": 2024, "form": "10-K", "accn": "0000950170-24-087843"},
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "assets")
    assert fact is not None
    assert fact["metric"] == "assets"
    assert fact["tag"] == "Assets"
    assert fact["value"] == 512163000000
    assert fact["period_end"] == "2024-06-30"
    assert fact["period_start"] is None  # Instant value
    assert fact["accession"] == "0000950170-24-087843"


def test_revenue_selects_annual_duration_not_quarterly(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "entityName": "Microsoft Corporation",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            # Q4 quarterly fact (~91 days)
                            {
                                "start": "2024-04-01",
                                "end": "2024-06-30",
                                "val": 64727000000,
                                "fy": 2024,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            },
                            # Full annual fact (366 days)
                            {
                                "start": "2023-07-01",
                                "end": "2024-06-30",
                                "val": 245122000000,
                                "fy": 2024,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            },
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "revenue")
    assert fact is not None
    assert fact["value"] == 245122000000
    assert fact["period_start"] == "2023-07-01"
    assert fact["period_end"] == "2024-06-30"


def test_prior_year_comparative_fact_not_selected(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "entityName": "Microsoft Corporation",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            # FY2023 comparison row reported in FY2024 10-K
                            {
                                "start": "2022-07-01",
                                "end": "2023-06-30",
                                "val": 72361000000,
                                "fy": 2024,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            },
                            # FY2024 current row
                            {
                                "start": "2023-07-01",
                                "end": "2024-06-30",
                                "val": 88136000000,
                                "fy": 2024,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            },
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "net_income")
    assert fact is not None
    # Must select the fact ending on 2024-06-30, not the prior year ending 2023-06-30
    assert fact["period_end"] == "2024-06-30"
    assert fact["value"] == 88136000000


def test_conflicting_matching_facts_produce_ambiguous_result(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "entityName": "Microsoft Corporation",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-07-01",
                                "end": "2024-06-30",
                                "val": 245000000000,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-07-01",
                                "end": "2024-06-30",
                                "val": 250000000000,  # Conflicting value
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            }
                        ]
                    }
                },
            }
        },
    }
    # Multiple candidate tags with conflicting values must return None (ambiguous)
    fact = normalize_annual_fact(companyfacts, sample_filing, "revenue")
    assert fact is None


def test_missing_concept_returns_none_not_zero(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "facts": {"us-gaap": {}},
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "operating_income")
    assert fact is None


def test_incorrect_financial_units_rejected(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "EUR": [
                            {"end": "2024-06-30", "val": 400000000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "assets")
    assert fact is None


def test_original_filing_accession_preserved(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "facts": {
            "us-gaap": {
                "Liabilities": {
                    "units": {
                        "USD": [
                            {"end": "2024-06-30", "val": 243686000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "liabilities")
    assert fact is not None
    assert fact["accession"] == "0000950170-24-087843"
    assert fact["selection_policy"] == "original_filing"


# --- 13, 14, 15: Caching, Atomic writes & Integrity ---

def test_valid_cached_filing_reused(tmp_path, sample_filing):
    raw_dir = tmp_path / "raw"
    filings_dir = raw_dir / "filings"
    filings_dir.mkdir(parents=True)

    acc_clean = sample_filing["accession"].replace("-", "")
    html_file = filings_dir / f"MSFT_2024_{acc_clean}.html"
    valid_content = b"<!DOCTYPE html><html><body>Valid 10-K filing content with more than 1000 bytes. " + b"X" * 1500 + b"</body></html>"
    html_file.write_bytes(valid_content)

    # Call download_filing without session (must not hit network)
    local_path, sha = download_filing(sample_filing, session=None, raw_dir=raw_dir)
    assert local_path == html_file
    import hashlib
    assert sha == hashlib.sha256(valid_content).hexdigest()


def test_failed_download_does_not_overwrite_valid_cache(tmp_path, sample_filing):
    raw_dir = tmp_path / "raw"
    filings_dir = raw_dir / "filings"
    filings_dir.mkdir(parents=True)

    acc_clean = sample_filing["accession"].replace("-", "")
    html_file = filings_dir / f"MSFT_2024_{acc_clean}.html"
    valid_content = b"<!DOCTYPE html><html><body>Preserved existing valid content. " + b"Z" * 1200 + b"</body></html>"
    html_file.write_bytes(valid_content)

    # If force_refresh is True but network fails or URL is invalid, file must remain intact
    invalid_filing = dict(sample_filing)
    invalid_filing["source_url"] = "https://invalid-host.example.com/test.htm"

    with pytest.raises(ValueError):
        download_filing(invalid_filing, session=None, raw_dir=raw_dir, force_refresh=True)

    # Original file is unchanged
    assert html_file.read_bytes() == valid_content


# --- 16: User-Agent Validation ---

def test_missing_or_placeholder_user_agent_rejected():
    with pytest.raises(ValueError, match="missing or empty"):
        validate_user_agent(None)

    with pytest.raises(ValueError, match="missing or empty"):
        validate_user_agent("")

    with pytest.raises(ValueError, match="placeholder text"):
        validate_user_agent("FyntraceResearch YourRealName your-real-email@example.com")

    with pytest.raises(ValueError, match="contact email address"):
        validate_user_agent("FyntraceResearch NoEmailHere")

    with pytest.raises(ValueError, match="contact email domain"):
        validate_user_agent("FyntraceResearch test@nodomain")

    # Valid User-Agent passes
    valid = validate_user_agent("FyntraceResearch JohnDoe jdoe@researchinstitute.org")
    assert valid == "FyntraceResearch JohnDoe jdoe@researchinstitute.org"


# --- 17: URL Safety ---

def test_unsafe_urls_rejected():
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        validate_sec_url("http://www.sec.gov/files/company_tickers.json")

    with pytest.raises(ValueError, match="Only approved SEC domains"):
        validate_sec_url("https://malicious-mirror.com/files/company_tickers.json")

    with pytest.raises(ValueError, match="path traversal detected"):
        validate_sec_url("https://www.sec.gov/Archives/edgar/data/../../../etc/passwd")


def test_build_archive_url_validates_path_traversal():
    with pytest.raises(ValueError, match="Unsafe primary document"):
        build_archive_url("0000789019", "0000950170-24-087843", "../evil.htm")

    with pytest.raises(ValueError, match="Invalid accession"):
        build_archive_url("0000789019", "invalid-accession", "doc.htm")


# --- 18: Unsupported Years & Companies Rejected ---

def test_unsupported_years_and_companies_rejected():
    # Unsupported year
    with pytest.raises(ValueError, match="Fiscal year 2020 is not supported"):
        select_annual_filing({"cik": "0000789019", "filings": {"recent": {}}}, fiscal_year=2020)

    # Unsupported ticker in collect_data
    with pytest.raises(ValueError, match="Unsupported ticker 'DIS'"):
        collect_data(tickers=["DIS"])

    with pytest.raises(ValueError, match="Unsupported year 2022"):
        collect_data(tickers=["MSFT"], years=[2022])

def test_get_fact_and_get_filing_lookups(tmp_path):
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir(parents=True)
    facts_file = processed_dir / 'facts.jsonl'
    manifest_file = processed_dir / 'manifest.json'

    fact_data = {
        'ticker': 'MSFT',
        'metric': 'assets',
        'fiscal_year': 2024,
        'value': 512163000000,
        'unit': 'USD',
        'tag': 'Assets',
        'form': '10-K',
        'accession': '0000950170-24-087843'
    }
    facts_file.write_text(json.dumps(fact_data) + '\n', encoding='utf-8')

    manifest_data = {
        'filings': [
            {
                'ticker': 'MSFT',
                'fiscal_year': 2024,
                'status': 'downloaded',
                'accession': '0000950170-24-087843',
                'form': '10-K'
            }
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding='utf-8')

    # Successful lookups
    fact = get_fact('MSFT', 'assets', 2024, processed_dir=processed_dir)
    assert fact['value'] == 512163000000

    filing = get_filing('MSFT', 2024, processed_dir=processed_dir)
    assert filing['status'] == 'downloaded'

    # Missing lookups raise KeyError
    with pytest.raises(KeyError):
        get_fact('AAPL', 'assets', 2024, processed_dir=processed_dir)

    with pytest.raises(KeyError):
        get_filing('AAPL', 2024, processed_dir=processed_dir)
