"""Unit tests for SEC data collection, validation, and fact normalization."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from fyntrace.sec_data import (
    AmbiguousFilingError,
    METRIC_CONFIG,
    REVENUE_CANDIDATE_TAGS,
    SUPPORTED_METRICS,
    SecAccessError,
    build_archive_url,
    collect_data,
    download_filing,
    get_companyfacts,
    get_fact,
    get_filing,
    infer_fiscal_year_from_filing,
    normalize_annual_fact,
    normalize_cik,
    resolve_ticker,
    select_annual_filing,
    validate_filing_html,
    validate_sec_url,
    validate_user_agent,
)


@pytest.fixture(autouse=True)
def guard_no_outbound_network(monkeypatch):
    """Network guard to fail tests immediately if an unexpected real network request occurs."""
    def forbidden_send(self, *args, **kwargs):
        raise RuntimeError("Unexpected real network request attempted during offline tests.")

    monkeypatch.setattr(requests.Session, "send", forbidden_send)


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
    mapping = {
        "0": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"},
    }
    resolved = resolve_ticker("XOM", mapping)
    assert resolved["ticker"] == "XOM"
    assert resolved["cik"] == "0000034088"
    assert resolved["name"] == "EXXON MOBIL CORP"


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
    selected_2024 = select_annual_filing(submissions, fiscal_year=2024)
    assert selected_2024 is not None
    assert selected_2024["accession"] == "0001628280-25-000001"

    selected_2025 = select_annual_filing(submissions, fiscal_year=2025)
    assert selected_2025 is None


def test_non_december_and_52_week_fiscal_years():
    assert infer_fiscal_year_from_filing("2024-01-28", "2024-02-21", "0131") == 2024
    assert infer_fiscal_year_from_filing("2024-06-30", "2024-07-30", "0630") == 2024
    assert infer_fiscal_year_from_filing("2024-09-28", "2024-11-01", "0925") == 2024
    assert infer_fiscal_year_from_filing("2023-01-01", "2023-02-16", "0103") == 2022
    assert infer_fiscal_year_from_filing("2023-12-31", "2024-02-16", "0103") == 2023


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
                            {"end": "2024-03-31", "val": 484275000000, "fy": 2024, "form": "10-Q", "accn": "0000950170-24-000010"},
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
    assert fact["period_start"] is None
    assert fact["accession"] == "0000950170-24-087843"


def test_instant_facts_with_duration_rejected(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "entityName": "Microsoft Corporation",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            # Has start date -> not a pure instant balance sheet fact
                            {
                                "start": "2023-07-01",
                                "end": "2024-06-30",
                                "val": 512163000000,
                                "fy": 2024,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            }
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "assets")
    assert fact is None


def test_revenue_selects_annual_duration_not_quarterly(sample_filing):
    companyfacts = {
        "cik": "0000789019",
        "entityName": "Microsoft Corporation",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-04-01",
                                "end": "2024-06-30",
                                "val": 64727000000,
                                "fy": 2024,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            },
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
                            {
                                "start": "2022-07-01",
                                "end": "2023-06-30",
                                "val": 72361000000,
                                "fy": 2024,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            },
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
                                "val": 250000000000,
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            }
                        ]
                    }
                },
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "revenue")
    assert fact is None


def test_duplicate_identical_records_do_not_produce_ambiguity(sample_filing):
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
                                "val": 245122000000,
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
                                "val": 245122000000,  # Same value as contract revenue
                                "form": "10-K",
                                "accn": "0000950170-24-087843",
                            }
                        ]
                    }
                },
            }
        },
    }
    fact = normalize_annual_fact(companyfacts, sample_filing, "revenue")
    assert fact is not None
    assert fact["value"] == 245122000000


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


def test_valid_cached_filing_reused(tmp_path, sample_filing):
    raw_dir = tmp_path / "raw"
    filings_dir = raw_dir / "filings"
    filings_dir.mkdir(parents=True)

    acc_clean = sample_filing["accession"].replace("-", "")
    html_file = filings_dir / f"MSFT_2024_{acc_clean}.html"
    valid_content = b"<!DOCTYPE html><html><body>Valid 10-K filing content. " + b"X" * 1500 + b"</body></html>"
    html_file.write_bytes(valid_content)

    local_path, sha = download_filing(sample_filing, session=None, raw_dir=raw_dir)
    assert local_path == html_file
    assert sha == hashlib.sha256(valid_content).hexdigest()


def test_http_200_containing_access_denied_page_rejected():
    block_page = (
        b"<!DOCTYPE html><html><head><title>Access Denied</title></head>"
        b"<body>Your request has been blocked by SEC.gov automated tool detection. "
        + b"A" * 1200
        + b"</body></html>"
    )
    with pytest.raises(SecAccessError, match="SEC block or error page detected"):
        validate_filing_html(block_page)


def test_fresh_non_html_content_rejected():
    plain_text = b"This is plain text with no markup " * 50
    with pytest.raises(ValueError, match="does not appear to be valid HTML"):
        validate_filing_html(plain_text)


def test_cached_filing_with_mismatched_stored_hash_rejected(tmp_path, sample_filing):
    raw_dir = tmp_path / "raw"
    filings_dir = raw_dir / "filings"
    filings_dir.mkdir(parents=True)

    acc_clean = sample_filing["accession"].replace("-", "")
    html_file = filings_dir / f"MSFT_2024_{acc_clean}.html"
    valid_content = b"<!DOCTYPE html><html><body>Content A " + b"A" * 1200 + b"</body></html>"
    html_file.write_bytes(valid_content)

    # When expected_sha256 does not match, download_filing must reject it
    with pytest.raises(ValueError, match="Cached filing SHA mismatch"):
        download_filing(
            sample_filing,
            session=None,
            raw_dir=raw_dir,
            expected_sha256="wrong_expected_sha256",
        )


def test_companyfacts_hash_equals_real_saved_file_hash(tmp_path, sample_filing):
    raw_dir = tmp_path / "raw"
    cf_dir = raw_dir / "companyfacts"
    cf_dir.mkdir(parents=True)

    cf_data = {
        "cik": 789019,
        "entityName": "Microsoft Corporation",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2024-06-30", "val": 512163000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                }
            }
        },
    }
    cf_bytes = json.dumps(cf_data, indent=2).encode("utf-8")
    cf_file = cf_dir / "CIK0000789019.json"
    cf_file.write_bytes(cf_bytes)
    expected_sha = hashlib.sha256(cf_bytes).hexdigest()

    loaded_cf = get_companyfacts("0000789019", session=None, raw_dir=raw_dir)
    assert loaded_cf["_file_sha256"] == expected_sha

    fact = normalize_annual_fact(loaded_cf, sample_filing, "assets")
    assert fact is not None
    assert fact["companyfacts_sha256"] == expected_sha


def test_sec_403_stops_subsequent_collection_requests(tmp_path):
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_session.get.return_value = mock_resp

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    res = collect_data(
        tickers=["MSFT", "AAPL"],
        years=[2024],
        cache_dir=raw_dir,
        session=mock_session,
    )
    curr = res["summary"]["current_run"]
    assert curr["failed_filings"] == 2
    assert curr["successful_filings"] == 0
    # Confirm mock_session was called minimally (stopped further request attempts after 403)
    assert mock_session.get.call_count == 1


def test_network_failures_not_labeled_as_ambiguous(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    mapping = {"0": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"}}
    (raw_dir / "company_tickers.json").write_text(json.dumps(mapping), encoding="utf-8")

    mock_session = MagicMock()
    mock_session.get.side_effect = ConnectionError("Temporary DNS resolution failure")

    res = collect_data(
        tickers=["MSFT"],
        years=[2024],
        cache_dir=raw_dir,
        session=mock_session,
    )
    filing_res = res["filings"][0]
    # Must be 'failed', not 'ambiguous'
    assert filing_res["status"] == "failed"
    assert "ambiguous" not in filing_res["status"]


def test_invalid_reporting_period_date_rejected():
    # MSFT fiscal year ends 0630. A reportDate of 2024-03-31 must be rejected.
    with pytest.raises(ValueError, match="does not align with fiscal year end"):
        infer_fiscal_year_from_filing("2024-03-31", "2024-04-25", "0630")


def test_mismatched_submissions_arrays_rejected():
    mismatched = {
        "cik": "0000789019",
        "fiscalYearEnd": "0630",
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q"],
                "reportDate": ["2024-06-30"],  # Length 1 vs form length 2
                "filingDate": ["2024-07-30", "2024-10-25"],
                "accessionNumber": ["0000950170-24-087843", "0000950170-24-087844"],
                "primaryDocument": ["msft.htm", "q.htm"],
            }
        },
    }
    with pytest.raises(ValueError, match="Mismatched aligned submission arrays"):
        select_annual_filing(mismatched, fiscal_year=2024)


def test_smaller_collection_preserves_unrelated_validated_records(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    # 1. Pre-populate WMT local file and manifest
    (raw_dir / "filings").mkdir(parents=True)
    wmt_html = b"<!DOCTYPE html><html><body>WMT 10K content " + b"W" * 1500 + b"</body></html>"
    wmt_file = raw_dir / "filings" / "WMT_2024_000010416924000056.html"
    wmt_file.write_bytes(wmt_html)
    wmt_sha = hashlib.sha256(wmt_html).hexdigest()

    manifest_data = {
        "company_identities": [
            {"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corp", "sector": "Technology"},
            {"ticker": "WMT", "cik": "0000104169", "name": "Walmart Inc", "sector": "Consumer"},
        ],
        "filings": [
            {
                "ticker": "WMT",
                "fiscal_year": 2024,
                "status": "downloaded",
                "accession": "0000104169-24-000056",
                "form": "10-K",
                "local_path": str(wmt_file.as_posix()),
                "file_sha256": wmt_sha,
                "selected_metrics": ["assets"],
                "unavailable_metrics": {},
                "errors": [],
            },
        ],
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    wmt_fact = {
        "ticker": "WMT",
        "cik": "0000104169",
        "company_name": "Walmart Inc",
        "sector": "Consumer",
        "fiscal_year": 2024,
        "metric": "assets",
        "taxonomy": "us-gaap",
        "tag": "Assets",
        "value": 252399000000,
        "unit": "USD",
        "form": "10-K",
        "accession": "0000104169-24-000056",
    }
    (processed_dir / "facts.jsonl").write_text(json.dumps(wmt_fact) + "\n", encoding="utf-8")

    # 2. Provide complete small mock SEC data for MSFT so no live network requests occur
    mapping = {
        "0": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        "1": {"cik_str": 104169, "ticker": "WMT", "title": "Walmart Inc"},
    }
    (raw_dir / "company_tickers.json").write_text(json.dumps(mapping), encoding="utf-8")

    sub_data = {
        "cik": "0000789019",
        "fiscalYearEnd": "0630",
        "filings": {
            "recent": {
                "form": ["10-K"],
                "reportDate": ["2024-06-30"],
                "filingDate": ["2024-07-30"],
                "accessionNumber": ["0000950170-24-087843"],
                "primaryDocument": ["msft-20240630.htm"],
            }
        },
    }
    (raw_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (raw_dir / "submissions" / "CIK0000789019.json").write_text(json.dumps(sub_data), encoding="utf-8")

    cf_data = {
        "cik": 789019,
        "entityName": "Microsoft Corp",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2024-06-30", "val": 512163000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                }
            }
        },
    }
    (raw_dir / "companyfacts").mkdir(parents=True, exist_ok=True)
    (raw_dir / "companyfacts" / "CIK0000789019.json").write_text(json.dumps(cf_data), encoding="utf-8")

    msft_html = b"<!DOCTYPE html><html><body>MSFT 10K content " + b"M" * 1500 + b"</body></html>"
    msft_file = raw_dir / "filings" / "MSFT_2024_000095017024087843.html"
    msft_file.write_bytes(msft_html)

    # 3. Run smaller collection ONLY for MSFT 2024
    res = collect_data(tickers=["MSFT"], years=[2024], cache_dir=raw_dir, user_agent="TestApp/1.0 test@sec.gov")
    curr = res["summary"]["current_run"]

    # Assertions required by Section 1:
    # - The requested MSFT collection succeeds.
    assert curr["successful_filings"] == 1
    # - The current run reports zero failed filings.
    assert curr["failed_filings"] == 0
    # - The existing WMT filing remains unchanged.
    wmt_filing = next(f for f in res["filings"] if f["ticker"] == "WMT" and f["fiscal_year"] == 2024)
    assert wmt_filing["status"] == "downloaded"
    assert wmt_filing["accession"] == "0000104169-24-000056"
    assert wmt_filing["file_sha256"] == wmt_sha
    # - The existing WMT financial fact remains unchanged.
    with (processed_dir / "facts.jsonl").open("r", encoding="utf-8") as f:
        stored_facts = [json.loads(line) for line in f if line.strip()]
    wmt_facts = [f for f in stored_facts if f["ticker"] == "WMT"]
    assert len(wmt_facts) == 1
    assert wmt_facts[0]["value"] == 252399000000
    # - The newly collected MSFT fact is valid.
    msft_facts = [f for f in stored_facts if f["ticker"] == "MSFT"]
    assert len(msft_facts) == 1
    assert msft_facts[0]["value"] == 512163000000
    assert msft_facts[0]["accession"] == "0000950170-24-087843"


def test_repeated_collection_does_not_duplicate_facts(tmp_path, sample_filing):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    mapping = {"0": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"}}
    (raw_dir / "company_tickers.json").write_text(json.dumps(mapping), encoding="utf-8")

    sub_data = {
        "cik": "0000789019",
        "fiscalYearEnd": "0630",
        "filings": {
            "recent": {
                "form": ["10-K"],
                "reportDate": ["2024-06-30"],
                "filingDate": ["2024-07-30"],
                "accessionNumber": ["0000950170-24-087843"],
                "primaryDocument": ["msft-20240630.htm"],
            }
        },
    }
    (raw_dir / "submissions").mkdir(parents=True)
    (raw_dir / "submissions" / "CIK0000789019.json").write_text(json.dumps(sub_data), encoding="utf-8")

    cf_data = {
        "cik": 789019,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2024-06-30", "val": 512163000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                }
            }
        },
    }
    (raw_dir / "companyfacts").mkdir(parents=True)
    (raw_dir / "companyfacts" / "CIK0000789019.json").write_text(json.dumps(cf_data), encoding="utf-8")

    (raw_dir / "filings").mkdir(parents=True)
    html_content = b"<!DOCTYPE html><html><body>MSFT 10K content " + b"A" * 1500 + b"</body></html>"
    (raw_dir / "filings" / "MSFT_2024_000095017024087843.html").write_bytes(html_content)

    # First collection
    collect_data(tickers=["MSFT"], years=[2024], cache_dir=raw_dir, user_agent="TestApp/1.0 test@sec.gov")
    with (processed_dir / "facts.jsonl").open("r", encoding="utf-8") as f:
        count_first = len(f.readlines())

    # Second collection
    collect_data(tickers=["MSFT"], years=[2024], cache_dir=raw_dir, user_agent="TestApp/1.0 test@sec.gov")
    with (processed_dir / "facts.jsonl").open("r", encoding="utf-8") as f:
        count_second = len(f.readlines())

    assert count_first == count_second == 1


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

    valid = validate_user_agent("FyntraceResearch JohnDoe jdoe@researchinstitute.org")
    assert valid == "FyntraceResearch JohnDoe jdoe@researchinstitute.org"


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


def test_unsupported_years_and_companies_rejected():
    with pytest.raises(ValueError, match="Fiscal year 2020 is not supported"):
        select_annual_filing({"cik": "0000789019", "filings": {"recent": {}}}, fiscal_year=2020)

    with pytest.raises(ValueError, match="Unsupported ticker 'DIS'"):
        collect_data(tickers=["DIS"])

    with pytest.raises(ValueError, match="Unsupported year 2022"):
        collect_data(tickers=["MSFT"], years=[2022])


def test_get_fact_and_get_filing_lookups(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    facts_file = processed_dir / "facts.jsonl"
    manifest_file = processed_dir / "manifest.json"

    fact_data = {
        "ticker": "MSFT",
        "metric": "assets",
        "fiscal_year": 2024,
        "value": 512163000000,
        "unit": "USD",
        "tag": "Assets",
        "form": "10-K",
        "accession": "0000950170-24-087843",
    }
    facts_file.write_text(json.dumps(fact_data) + "\n", encoding="utf-8")

    manifest_data = {
        "filings": [
            {
                "ticker": "MSFT",
                "fiscal_year": 2024,
                "status": "downloaded",
                "accession": "0000950170-24-087843",
                "form": "10-K",
            }
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    fact = get_fact("MSFT", "assets", 2024, processed_dir=processed_dir)
    assert fact["value"] == 512163000000

    filing = get_filing("MSFT", 2024, processed_dir=processed_dir)
    assert filing["status"] == "downloaded"

    with pytest.raises(KeyError):
        get_fact("AAPL", "assets", 2024, processed_dir=processed_dir)

    with pytest.raises(KeyError):
        get_filing("AAPL", 2024, processed_dir=processed_dir)


def test_revenue_two_different_values_same_tag_rejected(sample_filing):
    cf = {
        "cik": 789019,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"start": "2023-07-01", "end": "2024-06-30", "val": 245122000000, "form": "10-K", "accn": "0000950170-24-087843"},
                            {"start": "2023-07-01", "end": "2024-06-30", "val": 250000000000, "form": "10-K", "accn": "0000950170-24-087843"},
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(cf, sample_filing, "revenue")
    assert fact is None


def test_revenue_two_identical_records_same_tag_accepted(sample_filing):
    cf = {
        "cik": 789019,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"start": "2023-07-01", "end": "2024-06-30", "val": 245122000000, "form": "10-K", "accn": "0000950170-24-087843"},
                            {"start": "2023-07-01", "end": "2024-06-30", "val": 245122000000, "form": "10-K", "accn": "0000950170-24-087843"},
                        ]
                    }
                }
            }
        },
    }
    fact = normalize_annual_fact(cf, sample_filing, "revenue")
    assert fact is not None
    assert fact["value"] == 245122000000


def test_revenue_conflicting_values_across_different_tags_rejected(sample_filing):
    cf = {
        "cik": 789019,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"start": "2023-07-01", "end": "2024-06-30", "val": 245122000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {"start": "2023-07-01", "end": "2024-06-30", "val": 250000000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                },
            }
        },
    }
    fact = normalize_annual_fact(cf, sample_filing, "revenue")
    assert fact is None


def test_source_hash_preserves_real_cached_bytes_and_none_for_mock_data(tmp_path, sample_filing):
    in_memory_cf = {
        "cik": 789019,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2024-06-30", "val": 512163000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                }
            }
        },
    }
    # In-memory mock data without _file_sha256 returns None
    fact_mock = normalize_annual_fact(in_memory_cf, sample_filing, "assets")
    assert fact_mock is not None
    assert fact_mock["companyfacts_sha256"] is None

    # Real cached file on disk has verified SHA-256
    cf_dir = tmp_path / "raw" / "companyfacts"
    cf_dir.mkdir(parents=True)
    raw_file = cf_dir / "CIK0000789019.json"
    content_bytes = json.dumps(in_memory_cf).encode("utf-8")
    raw_file.write_bytes(content_bytes)
    expected_sha = hashlib.sha256(content_bytes).hexdigest()

    loaded_cf = get_companyfacts("0000789019", raw_dir=tmp_path / "raw")
    fact_real = normalize_annual_fact(loaded_cf, sample_filing, "assets")
    assert fact_real is not None
    assert fact_real["companyfacts_sha256"] == expected_sha


def test_company_identity_mismatch_rejected(sample_filing):
    mismatched_cf = {
        "cik": "0000320193",  # AAPL CIK vs MSFT sample_filing
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2024-06-30", "val": 512163000000, "form": "10-K", "accn": "0000950170-24-087843"}
                        ]
                    }
                }
            }
        },
    }
    with pytest.raises(ValueError, match="Company identity mismatch"):
        normalize_annual_fact(mismatched_cf, sample_filing, "assets")


def test_corrupted_manifest_file_rejected_without_overwriting(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    manifest_file = processed_dir / "manifest.json"
    manifest_file.write_text("{ corrupt json ...", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted or invalid manifest"):
        collect_data(tickers=["MSFT"], years=[2024], cache_dir=raw_dir)

    assert manifest_file.read_text(encoding="utf-8") == "{ corrupt json ..."


def test_corrupted_facts_file_rejected_without_overwriting(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    manifest_file = processed_dir / "manifest.json"
    manifest_file.write_text(json.dumps({"filings": [], "company_identities": []}), encoding="utf-8")
    facts_file = processed_dir / "facts.jsonl"
    facts_file.write_text('{"ticker": "MSFT", "fiscal_year": 2024, "metric": "assets"}\n{ invalid json line\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted or invalid facts file"):
        collect_data(tickers=["MSFT"], years=[2024], cache_dir=raw_dir)


def test_failed_refresh_preserves_valid_filing_and_facts(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    filings_dir = raw_dir / "filings"
    filings_dir.mkdir(parents=True)
    html_content = b"<!DOCTYPE html><html><body>MSFT 10K content " + b"M" * 1500 + b"</body></html>"
    msft_file = filings_dir / "MSFT_2024_000095017024087843.html"
    msft_file.write_bytes(html_content)
    file_sha = hashlib.sha256(html_content).hexdigest()

    manifest_data = {
        "company_identities": [
            {"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corp", "sector": "Technology"},
        ],
        "filings": [
            {
                "ticker": "MSFT",
                "fiscal_year": 2024,
                "status": "downloaded",
                "accession": "0000950170-24-087843",
                "form": "10-K",
                "local_path": str(msft_file.as_posix()),
                "file_sha256": file_sha,
                "selected_metrics": ["assets"],
                "unavailable_metrics": {},
                "errors": [],
            }
        ],
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    fact = {
        "ticker": "MSFT",
        "cik": "0000789019",
        "company_name": "Microsoft Corp",
        "sector": "Technology",
        "fiscal_year": 2024,
        "metric": "assets",
        "taxonomy": "us-gaap",
        "tag": "Assets",
        "value": 512163000000,
        "unit": "USD",
        "form": "10-K",
        "accession": "0000950170-24-087843",
    }
    (processed_dir / "facts.jsonl").write_text(json.dumps(fact) + "\n", encoding="utf-8")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_session.get.return_value = mock_resp

    res = collect_data(tickers=["MSFT"], years=[2024], cache_dir=raw_dir, session=mock_session)
    curr = res["summary"]["current_run"]

    assert curr["failed_filings"] == 1
    assert curr["successful_filings"] == 0

    preserved_filing = next(f for f in res["filings"] if f["ticker"] == "MSFT")
    assert preserved_filing["status"] == "downloaded"
    assert preserved_filing["accession"] == "0000950170-24-087843"
    assert any("Refresh failed" in err for err in preserved_filing.get("errors", []))

    with (processed_dir / "facts.jsonl").open("r", encoding="utf-8") as f:
        stored = [json.loads(line) for line in f if line.strip()]
    assert len(stored) == 1
    assert stored[0]["value"] == 512163000000


def test_failed_refresh_clears_stale_facts_for_invalid_filing(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    manifest_data = {
        "company_identities": [
            {"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corp", "sector": "Technology"},
        ],
        "filings": [
            {
                "ticker": "MSFT",
                "fiscal_year": 2024,
                "status": "failed",
                "accession": None,
                "form": "10-K",
                "local_path": None,
                "file_sha256": None,
            }
        ],
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    stale_fact = {
        "ticker": "MSFT",
        "fiscal_year": 2024,
        "metric": "assets",
        "value": 999999999,
        "unit": "USD",
    }
    (processed_dir / "facts.jsonl").write_text(json.dumps(stale_fact) + "\n", encoding="utf-8")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_session.get.return_value = mock_resp

    res = collect_data(tickers=["MSFT"], years=[2024], cache_dir=raw_dir, session=mock_session)
    assert res["summary"]["current_run"]["failed_filings"] == 1

    with (processed_dir / "facts.jsonl").open("r", encoding="utf-8") as f:
        stored = [json.loads(line) for line in f if line.strip()]
    assert len(stored) == 0


def test_network_guard_blocks_real_http_calls():
    s = requests.Session()
    with pytest.raises(RuntimeError, match="Unexpected real network request attempted during offline tests"):
        s.get("https://www.sec.gov")
