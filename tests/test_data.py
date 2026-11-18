"""Unit tests for the SEC data client and filing discovery (offline, mocked)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.sec_data import (
    COLLECTION_CUTOFF_DATE,
    COMPANY_PANEL,
    COMPANY_SPECIFIC_REVENUE_TAGS,
    FINANCIAL_METRICS,
    HISTORICAL_PANEL_REGISTRANTS,
    PANEL_TICKERS,
    TARGET_FISCAL_YEARS,
    VERIFIED_COMPREHENSIVE_REVENUE_TAGS,
    collect_panel_data,
    extract_xbrl_10k_fiscal_years,
    filing_url,
    get_companyfacts,
    get_submissions,
    list_filings,
    load_company_tickers,
    load_filing_manifest,
    make_sec_session,
    normalize_cik,
    normalize_panel_facts,
    resolve_ticker,
    select_annual_filings_for_company,
    select_financial_fact,
    validate_companyfacts_response,
    validate_submissions_response,
    validate_ticker_payload,
    validate_user_agent,
)

MOCK_TICKERS_PAYLOAD = {
    "0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "2": {"cik_str": 1318605, "ticker": "TSLA", "title": "TESLA, INC."},
}

MOCK_FACTS_PAYLOAD = {
    "cik": 789019,
    "entityName": "MICROSOFT CORP",
    "facts": {
        "dei": {},
        "us-gaap": {
            "Assets": {
                "label": "Assets",
                "units": {
                    "USD": [
                        {
                            "val": 100,
                            "form": "10-K",
                            "fp": "FY",
                            "fy": 2024,
                            "accn": "0000789019-24-000050",
                            "end": "2024-06-30",
                        },
                        {
                            "val": 90,
                            "form": "10-K",
                            "fp": "FY",
                            "fy": 2023,
                            "accn": "0000789019-23-000040",
                            "end": "2023-06-30",
                        },
                    ]
                },
            },
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        {
                            "val": 200,
                            "form": "10-K",
                            "fp": "FY",
                            "fy": 2024,
                            "accn": "0000789019-24-000050",
                            "end": "2024-06-30",
                        }
                    ]
                },
            },
        },
    },
}

MOCK_SUBMISSIONS_PAYLOAD = {
    "cik": "0000789019",
    "entityType": "operating",
    "name": "MICROSOFT CORP",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000789019-24-000050",
                "0000789019-24-000099",
                "0000789019-23-000040",
                "0000789019-26-000999",
            ],
            "form": ["10-K", "10-K/A", "10-K", "10-K"],
            "filingDate": ["2024-07-30", "2024-08-15", "2023-07-27", "2026-12-01"],
            "reportDate": ["2024-06-30", "2024-06-30", "2023-06-30", "2026-06-30"],
            "primaryDocument": [
                "msft-20240630.htm",
                "msft-20240630a.htm",
                "msft-20230630.htm",
                "msft-20260630.htm",
            ],
        }
    },
}


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure all tests run isolated from any real local .env file or environment variable."""
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    isolated_env = tmp_path / ".env.isolated"
    monkeypatch.setattr("src.sec_data.PROJECT_ENV_PATH", isolated_env)


def create_mock_session(
    status_code: int = 200,
    json_data: dict | list | None = None,
    raise_http_error: bool = False,
) -> MagicMock:
    """Helper to build a mock requests.Session."""
    session = MagicMock(spec=requests.Session)
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code

    if raise_http_error:
        http_error = requests.HTTPError(f"HTTP Error {status_code}", response=response)
        response.raise_for_status.side_effect = http_error
    else:
        response.raise_for_status.return_value = None

    if json_data is not None:
        response.json.return_value = json_data

    session.get.return_value = response
    return session


def test_company_panel_constant() -> None:
    """Verify company panel contains 30 distinct companies across 5 sectors."""
    assert len(COMPANY_PANEL) == 5
    expected_sectors = {
        "Technology",
        "Financial services",
        "Healthcare",
        "Energy and utilities",
        "Consumer",
    }
    assert set(COMPANY_PANEL.keys()) == expected_sectors

    all_tickers = [ticker for tickers in COMPANY_PANEL.values() for ticker in tickers]
    assert len(all_tickers) == 30
    assert len(set(all_tickers)) == 30
    assert len(PANEL_TICKERS) == 30

    for expected in ("AAPL", "MSFT", "NVDA", "JPM", "JNJ", "XOM", "WMT"):
        assert expected in PANEL_TICKERS


def test_filing_url_construction() -> None:
    """Verify construction of reliable official SEC archive URL."""
    url = filing_url("0000789019", "0000789019-24-000050", "msft-20240630.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/789019/000078901924000050/msft-20240630.htm"


def test_1_ticker_to_cik_mapping(tmp_path: Path) -> None:
    """Test 1: Correct ticker-to-CIK mapping."""
    mock_session = create_mock_session(json_data=MOCK_TICKERS_PAYLOAD)
    result = resolve_ticker("MSFT", session=mock_session, cache_dir=tmp_path)

    assert result["cik"] == "0000789019"
    assert result["company_name"] == "MICROSOFT CORP"
    assert result["ticker"] == "MSFT"


def test_2_case_insensitive_ticker_lookup(tmp_path: Path) -> None:
    """Test 2: Case-insensitive ticker lookup."""
    mock_session = create_mock_session(json_data=MOCK_TICKERS_PAYLOAD)

    res_lower = resolve_ticker("msft", session=mock_session, cache_dir=tmp_path)
    res_mixed = resolve_ticker("MsFt", session=mock_session, cache_dir=tmp_path)

    assert res_lower["cik"] == "0000789019"
    assert res_mixed["cik"] == "0000789019"
    assert res_lower["ticker"] == "MSFT"
    assert res_mixed["ticker"] == "MSFT"


def test_3_unknown_ticker_handling(tmp_path: Path) -> None:
    """Test 3: Unknown ticker handling raises a clear error."""
    mock_session = create_mock_session(json_data=MOCK_TICKERS_PAYLOAD)

    with pytest.raises(KeyError) as exc_info:
        resolve_ticker("UNKNOWN_TICKER", session=mock_session, cache_dir=tmp_path)

    assert "UNKNOWN_TICKER" in str(exc_info.value)


def test_4_correct_10_digit_cik_formatting() -> None:
    """Test 4: Correct 10-digit zero-padded CIK formatting and validation."""
    assert normalize_cik(789019) == "0000789019"
    assert normalize_cik("789019") == "0000789019"
    assert normalize_cik("0000789019") == "0000789019"
    assert normalize_cik(1) == "0000000001"
    assert normalize_cik(9999999999) == "9999999999"

    with pytest.raises(ValueError):
        normalize_cik("invalid")
    with pytest.raises(ValueError):
        normalize_cik("-123")
    with pytest.raises(ValueError):
        normalize_cik(0)
    with pytest.raises(ValueError):
        normalize_cik(None)  # type: ignore[arg-type]


def test_rejection_of_cik_exceeding_ten_digits() -> None:
    """Test rejection of numeric identifiers exceeding 10 digits."""
    with pytest.raises(ValueError) as exc1:
        normalize_cik(12345678901)
    assert "10 digits" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        normalize_cik("12345678901")
    assert "10 digits" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        normalize_cik("00000000001")
    assert "10 digits" in str(exc3.value)


def test_5_company_facts_retrieval(tmp_path: Path) -> None:
    """Test 5: Successful company-facts retrieval using a mocked response."""
    mock_session = create_mock_session(json_data=MOCK_FACTS_PAYLOAD)
    facts = get_companyfacts("0000789019", session=mock_session, cache_dir=tmp_path)

    assert facts["cik"] == 789019
    assert facts["entityName"] == "MICROSOFT CORP"
    assert "Assets" in facts["facts"]["us-gaap"]

    cached_file = tmp_path / "companyfacts" / "CIK0000789019.json"
    assert cached_file.is_file()
    with open(cached_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["entityName"] == "MICROSOFT CORP"


def test_6_reusing_cached_responses_without_http_request(tmp_path: Path) -> None:
    """Test 6: Reusing cached responses without another HTTP request."""
    companyfacts_dir = tmp_path / "companyfacts"
    companyfacts_dir.mkdir(parents=True)
    cache_file = companyfacts_dir / "CIK0000789019.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(MOCK_FACTS_PAYLOAD, f)

    failing_session = MagicMock(spec=requests.Session)
    failing_session.get.side_effect = AssertionError("HTTP request should not be made when cache exists.")

    facts = get_companyfacts("0000789019", session=failing_session, cache_dir=tmp_path)
    assert facts["entityName"] == "MICROSOFT CORP"
    failing_session.get.assert_not_called()


def test_7_invalid_json_and_http_error_handling(tmp_path: Path) -> None:
    """Test 7: Invalid JSON and HTTP error handling."""
    error_session = create_mock_session(status_code=404, raise_http_error=True)
    with pytest.raises(requests.HTTPError):
        get_companyfacts("0000789019", session=error_session, cache_dir=tmp_path)

    rate_limit_session = create_mock_session(status_code=429)
    with pytest.raises(requests.HTTPError) as exc_info:
        get_companyfacts("0000789019", session=rate_limit_session, cache_dir=tmp_path)
    assert "429" in str(exc_info.value)

    bad_json_session = MagicMock(spec=requests.Session)
    bad_response = MagicMock(spec=requests.Response)
    bad_response.status_code = 200
    bad_response.raise_for_status.return_value = None
    bad_response.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    bad_json_session.get.return_value = bad_response

    with pytest.raises(ValueError) as exc_json:
        get_companyfacts("0000789019", session=bad_json_session, cache_dir=tmp_path)
    assert "Malformed JSON" in str(exc_json.value)


def test_8_missing_sec_user_agent_handling() -> None:
    """Test 8: Missing SEC User-Agent handling and rejection of placeholders."""
    with pytest.raises(ValueError):
        validate_user_agent(None)
    with pytest.raises(ValueError):
        validate_user_agent("")
    with pytest.raises(ValueError):
        validate_user_agent("   ")

    with pytest.raises(ValueError) as exc1:
        validate_user_agent("Fyntrace/0.1 YourRealName your-real-email@example.com")
    assert "placeholder" in str(exc1.value).lower()

    with pytest.raises(ValueError):
        validate_user_agent("Sample Company Name AdminContact@<sample company domain>.com")

    with pytest.raises(ValueError) as exc2:
        validate_user_agent("MyCompanyResearchTeam")
    assert "contact email" in str(exc2.value).lower()

    valid = validate_user_agent("Fyntrace/0.1 ResearchTeam research@valid-domain.org")
    assert valid == "Fyntrace/0.1 ResearchTeam research@valid-domain.org"

    with pytest.raises(ValueError):
        make_sec_session(user_agent="Fyntrace/0.1 YourRealName your-real-email@example.com")

    with pytest.raises(ValueError) as exc_missing:
        make_sec_session()
    assert "Missing SEC User-Agent" in str(exc_missing.value)


def test_9_failed_downloads_do_not_overwrite_cached_files(tmp_path: Path) -> None:
    """Test 9: Failed downloads must not overwrite valid cached files."""
    companyfacts_dir = tmp_path / "companyfacts"
    companyfacts_dir.mkdir(parents=True)
    cache_file = companyfacts_dir / "CIK0000789019.json"

    original_content = {
        "cik": 789019,
        "entityName": "ORIGINAL_UNTOUCHED_CONTENT",
        "facts": {"us-gaap": {}},
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(original_content, f)

    failing_session = create_mock_session(status_code=500, raise_http_error=True)

    with pytest.raises(requests.HTTPError):
        get_companyfacts("0000789019", session=failing_session, cache_dir=tmp_path, force_refresh=True)

    with open(cache_file, "r", encoding="utf-8") as f:
        current_content = json.load(f)

    assert current_content["entityName"] == "ORIGINAL_UNTOUCHED_CONTENT"


def test_submissions_retrieval_and_caching(tmp_path: Path) -> None:
    """Test submissions retrieval and caching."""
    mock_session = create_mock_session(json_data=MOCK_SUBMISSIONS_PAYLOAD)
    submissions = get_submissions("0000789019", session=mock_session, cache_dir=tmp_path)

    assert submissions["cik"] == "0000789019"
    assert submissions["name"] == "MICROSOFT CORP"

    cached_file = tmp_path / "submissions" / "CIK0000789019.json"
    assert cached_file.is_file()


def test_cache_separation_between_submissions_and_companyfacts(tmp_path: Path) -> None:
    """Submissions and company facts must only read their respective cache directories."""
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir(parents=True)
    sub_file = submissions_dir / "CIK0000789019.json"
    sub_file.write_text(json.dumps(MOCK_SUBMISSIONS_PAYLOAD), encoding="utf-8")

    mock_session = create_mock_session(json_data=MOCK_FACTS_PAYLOAD)
    facts = get_companyfacts("0000789019", session=mock_session, cache_dir=tmp_path)
    assert facts["cik"] == 789019
    assert (tmp_path / "companyfacts" / "CIK0000789019.json").is_file()
    mock_session.get.assert_called_once()


def test_rejection_of_mismatched_cik_responses(tmp_path: Path) -> None:
    """Reject responses where returned CIK does not match requested CIK."""
    mismatched_facts = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {"us-gaap": {}},
    }
    mock_session = create_mock_session(json_data=mismatched_facts)
    with pytest.raises(ValueError) as exc_facts:
        get_companyfacts("0000789019", session=mock_session, cache_dir=tmp_path)
    assert "mismatch" in str(exc_facts.value).lower()
    assert not (tmp_path / "companyfacts" / "CIK0000789019.json").exists()

    mismatched_sub = {
        "cik": "0000320193",
        "entityType": "operating",
        "name": "Apple Inc.",
        "filings": {"recent": {}},
    }
    mock_session_sub = create_mock_session(json_data=mismatched_sub)
    with pytest.raises(ValueError) as exc_sub:
        get_submissions("0000789019", session=mock_session_sub, cache_dir=tmp_path)
    assert "mismatch" in str(exc_sub.value).lower()
    assert not (tmp_path / "submissions" / "CIK0000789019.json").exists()


def test_rejection_of_invalid_sec_response_structures(tmp_path: Path) -> None:
    """Reject malformed response structures before saving to cache."""
    session_not_dict = create_mock_session(json_data=["not", "a", "dict"])
    with pytest.raises(ValueError) as exc1:
        get_companyfacts("0000789019", session=session_not_dict, cache_dir=tmp_path)
    assert "json object" in str(exc1.value).lower()

    bad_facts = {"cik": 789019, "entityName": "MICROSOFT CORP"}
    session_bad_facts = create_mock_session(json_data=bad_facts)
    with pytest.raises(ValueError) as exc2:
        get_companyfacts("0000789019", session=session_bad_facts, cache_dir=tmp_path)
    assert "facts" in str(exc2.value).lower()

    bad_sub = {"cik": "0000789019", "name": "MICROSOFT CORP"}
    session_bad_sub = create_mock_session(json_data=bad_sub)
    with pytest.raises(ValueError) as exc3:
        get_submissions("0000789019", session=session_bad_sub, cache_dir=tmp_path)
    assert "filings" in str(exc3.value).lower()

    session_bad_tickers = create_mock_session(json_data={})
    with pytest.raises(ValueError) as exc4:
        load_company_tickers(session=session_bad_tickers, cache_dir=tmp_path)
    assert "no valid ticker entries" in str(exc4.value).lower()


def test_missing_user_agent_when_env_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing-User-Agent behavior when .env exists without SEC_USER_AGENT."""
    dummy_env = tmp_path / ".env"
    dummy_env.write_text("OTHER_CONFIG=123\n", encoding="utf-8")
    monkeypatch.setattr("src.sec_data.PROJECT_ENV_PATH", dummy_env)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(ValueError) as exc:
        make_sec_session()
    assert "Missing SEC User-Agent" in str(exc.value)


def test_placeholder_user_agent_when_env_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify placeholder rejection when .env exists with placeholder SEC_USER_AGENT."""
    dummy_env = tmp_path / ".env"
    dummy_env.write_text("SEC_USER_AGENT=Fyntrace/0.1 YourRealName your-real-email@example.com\n", encoding="utf-8")
    monkeypatch.setattr("src.sec_data.PROJECT_ENV_PATH", dummy_env)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(ValueError) as exc:
        make_sec_session()
    assert "placeholder" in str(exc.value).lower()


def test_list_filings_metadata_and_parallel_arrays(tmp_path: Path) -> None:
    """Test 1 & 2: Correct metadata extraction and matching across parallel SEC arrays."""
    mock_session = create_mock_session(json_data=MOCK_SUBMISSIONS_PAYLOAD)
    filings = list_filings("0000789019", session=mock_session, cache_dir=tmp_path)

    assert len(filings) == 4
    # Check first record correctly paired
    first = filings[0]
    assert first["accession_number"] == "0000789019-24-000050"
    assert first["form"] == "10-K"
    assert first["filing_date"] == "2024-07-30"
    assert first["report_date"] == "2024-06-30"
    assert first["primary_document"] == "msft-20240630.htm"
    assert "000078901924000050" in first["source_url"]


def test_10k_versus_10ka_distinction(tmp_path: Path) -> None:
    """Test 3: Distinguish 10-K from 10-K/A."""
    mock_session = create_mock_session(json_data=MOCK_SUBMISSIONS_PAYLOAD)
    only_10k = list_filings("0000789019", forms=["10-K"], session=mock_session, cache_dir=tmp_path)

    assert len(only_10k) == 3
    assert all(f["form"] == "10-K" for f in only_10k)
    assert not any(f["form"] == "10-K/A" for f in only_10k)

    only_10ka = list_filings("0000789019", forms=["10-K/A"], session=mock_session, cache_dir=tmp_path)
    assert len(only_10ka) == 1
    assert only_10ka[0]["form"] == "10-K/A"
    assert only_10ka[0]["accession_number"] == "0000789019-24-000099"


def test_filing_date_cutoff(tmp_path: Path) -> None:
    """Test 4: Exclude filings after cutoff date."""
    mock_session = create_mock_session(json_data=MOCK_SUBMISSIONS_PAYLOAD)
    cutoff = "2024-08-01"
    filings = list_filings("0000789019", as_of=cutoff, session=mock_session, cache_dir=tmp_path)

    # 2024-07-30 and 2023-07-27 are <= cutoff, while 2024-08-15 and 2026-12-01 are > cutoff
    assert len(filings) == 2
    filing_dates = [f["filing_date"] for f in filings]
    assert "2024-07-30" in filing_dates
    assert "2023-07-27" in filing_dates
    assert "2024-08-15" not in filing_dates
    assert "2026-12-01" not in filing_dates


def test_missing_and_malformed_submissions_metadata(tmp_path: Path) -> None:
    """Test 5: Gracefully handle missing or malformed filings arrays."""
    malformed_sub = {
        "cik": "0000789019",
        "entityType": "operating",
        "name": "MICROSOFT CORP",
        "filings": {
            "recent": {
                "accessionNumber": ["0000789019-24-000050"],
                # Mismatched lengths
                "form": [],
            }
        },
    }
    mock_session = create_mock_session(json_data=malformed_sub)
    filings = list_filings("0000789019", session=mock_session, cache_dir=tmp_path)
    assert filings == []


def test_fiscal_year_selection_and_uncertainty(tmp_path: Path) -> None:
    """Test 6 & 7: Test fiscal year confirmation, needs_review, missing, and accession preservation."""
    # Pre-populate submissions and facts in tmp_path cache
    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir(parents=True)
    (sub_dir / "CIK0000789019.json").write_text(json.dumps(MOCK_SUBMISSIONS_PAYLOAD), encoding="utf-8")

    facts_dir = tmp_path / "companyfacts"
    facts_dir.mkdir(parents=True)
    (facts_dir / "CIK0000789019.json").write_text(json.dumps(MOCK_FACTS_PAYLOAD), encoding="utf-8")

    records = select_annual_filings_for_company(
        cik="0000789019",
        ticker="MSFT",
        company_name="MICROSOFT CORP",
        sector="Technology",
        target_years=[2023, 2024, 2025],
        as_of=COLLECTION_CUTOFF_DATE,
        cache_dir=tmp_path,
    )

    assert len(records) == 3
    rec_by_year = {r["target_fiscal_year"]: r for r in records}

    # 2024: Confirmed original 10-K (not replaced by 10-K/A)
    rec_2024 = rec_by_year[2024]
    assert rec_2024["fiscal_year_status"] == "confirmed"
    assert rec_2024["form"] == "10-K"
    assert rec_2024["accession_number"] == "0000789019-24-000050"

    # 2023: Confirmed original 10-K
    rec_2023 = rec_by_year[2023]
    assert rec_2023["fiscal_year_status"] == "confirmed"
    assert rec_2023["accession_number"] == "0000789019-23-000040"

    # 2025: Missing (no 2025 10-K in mock payload)
    rec_2025 = rec_by_year[2025]
    assert rec_2025["fiscal_year_status"] == "missing"
    assert rec_2025["form"] is None
    assert rec_2025["accession_number"] is None


def test_collect_panel_data_mocked(tmp_path: Path) -> None:
    """Test panel collection orchestration with mocked responses."""
    # Pre-populate company_tickers.json
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    tickers_dict = {
        str(i): {"cik_str": 789019, "ticker": ticker, "title": f"{ticker} Inc"}
        for i, ticker in enumerate(PANEL_TICKERS.keys())
    }
    (raw_dir / "company_tickers.json").write_text(json.dumps(tickers_dict), encoding="utf-8")

    # Mock submissions and facts for standard CIK 0000789019
    sub_dir = raw_dir / "submissions"
    sub_dir.mkdir(parents=True)
    (sub_dir / "CIK0000789019.json").write_text(json.dumps(MOCK_SUBMISSIONS_PAYLOAD), encoding="utf-8")

    facts_dir = raw_dir / "companyfacts"
    facts_dir.mkdir(parents=True)
    (facts_dir / "CIK0000789019.json").write_text(json.dumps(MOCK_FACTS_PAYLOAD), encoding="utf-8")

    # Mock submissions and facts for XOM historical CIK 0000034088
    xom_sub = {
        "cik": "0000034088",
        "entityType": "operating",
        "name": "EXXON MOBIL CORP",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000034088-26-000045",
                    "0000034088-25-000010",
                    "0000034088-24-000018",
                ],
                "form": ["10-K", "10-K", "10-K"],
                "filingDate": ["2026-02-25", "2025-02-26", "2024-02-28"],
                "reportDate": ["2025-12-31", "2024-12-31", "2023-12-31"],
                "primaryDocument": ["xom-20251231.htm", "xom-20241231.htm", "xom-20231231.htm"],
            }
        },
    }
    (sub_dir / "CIK0000034088.json").write_text(json.dumps(xom_sub), encoding="utf-8")

    xom_facts = {
        "cik": 34088,
        "entityName": "EXXON MOBIL CORP",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "fy": 2025, "accn": "0000034088-26-000045", "val": 300},
                            {"form": "10-K", "fp": "FY", "fy": 2024, "accn": "0000034088-25-000010", "val": 200},
                            {"form": "10-K", "fp": "FY", "fy": 2023, "accn": "0000034088-24-000018", "val": 100},
                        ]
                    }
                }
            }
        },
    }
    (facts_dir / "CIK0000034088.json").write_text(json.dumps(xom_facts), encoding="utf-8")

    out_file = tmp_path / "processed" / "filing_manifest.json"
    mock_session = create_mock_session()
    summary = collect_panel_data(session=mock_session, cache_dir=raw_dir, output_path=out_file)

    assert summary["companies_processed"] == 30
    assert summary["companies_successful"] == 30
    assert summary["download_failures"] == 0
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 90

    # Verify XOM records in manifest use CIK 0000034088 and EXXON MOBIL CORP
    xom_records = [r for r in manifest if r["ticker"] == "XOM"]
    assert len(xom_records) == 3
    for xr in xom_records:
        assert xr["cik"] == "0000034088"
        assert xr["company_name"] == "EXXON MOBIL CORP"
        assert xr["fiscal_year_status"] == "confirmed"


def test_xom_current_ticker_resolution_unchanged(tmp_path: Path) -> None:
    """Verify resolve_ticker('XOM') returns current SEC mapping and is not overridden."""
    tickers_payload = {
        "0": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"},
    }
    mock_session = create_mock_session(json_data=tickers_payload)
    resolved = resolve_ticker("XOM", session=mock_session, cache_dir=tmp_path)

    assert resolved["ticker"] == "XOM"
    assert resolved["cik"] == "0002115436"
    assert resolved["company_name"] == "ExxonMobil Holdings Corp"


def test_historical_panel_registrants_isolation() -> None:
    """Verify only XOM has a historical registrant override for the 2023-2025 panel."""
    assert set(HISTORICAL_PANEL_REGISTRANTS.keys()) == {"XOM"}
    assert HISTORICAL_PANEL_REGISTRANTS["XOM"]["cik"] == "0000034088"
    assert HISTORICAL_PANEL_REGISTRANTS["XOM"]["company_name"] == "EXXON MOBIL CORP"

    for ticker in PANEL_TICKERS.keys():
        if ticker != "XOM":
            assert ticker not in HISTORICAL_PANEL_REGISTRANTS


def test_xom_historical_filings_selection(tmp_path: Path) -> None:
    """Verify XOM's three historical 10-K filings are assigned to the correct fiscal years."""
    raw_dir = tmp_path
    sub_dir = raw_dir / "submissions"
    sub_dir.mkdir(parents=True)
    facts_dir = raw_dir / "companyfacts"
    facts_dir.mkdir(parents=True)

    xom_sub = {
        "cik": "0000034088",
        "entityType": "operating",
        "name": "EXXON MOBIL CORP",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000034088-26-000045",
                    "0000034088-25-000010",
                    "0000034088-24-000018",
                ],
                "form": ["10-K", "10-K", "10-K"],
                "filingDate": ["2026-02-25", "2025-02-26", "2024-02-28"],
                "reportDate": ["2025-12-31", "2024-12-31", "2023-12-31"],
                "primaryDocument": ["xom-20251231.htm", "xom-20241231.htm", "xom-20231231.htm"],
            }
        },
    }
    (sub_dir / "CIK0000034088.json").write_text(json.dumps(xom_sub), encoding="utf-8")

    xom_facts = {
        "cik": 34088,
        "entityName": "EXXON MOBIL CORP",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "fy": 2025, "accn": "0000034088-26-000045", "val": 300},
                            {"form": "10-K", "fp": "FY", "fy": 2024, "accn": "0000034088-25-000010", "val": 200},
                            {"form": "10-K", "fp": "FY", "fy": 2023, "accn": "0000034088-24-000018", "val": 100},
                        ]
                    }
                }
            }
        },
    }
    (facts_dir / "CIK0000034088.json").write_text(json.dumps(xom_facts), encoding="utf-8")

    records = select_annual_filings_for_company(
        cik="0000034088",
        ticker="XOM",
        company_name="EXXON MOBIL CORP",
        sector="Energy and utilities",
        target_years=[2023, 2024, 2025],
        as_of=COLLECTION_CUTOFF_DATE,
        cache_dir=raw_dir,
    )

    rec_by_year = {r["target_fiscal_year"]: r for r in records}
    assert rec_by_year[2023]["accession_number"] == "0000034088-24-000018"
    assert rec_by_year[2023]["fiscal_year_status"] == "confirmed"
    assert rec_by_year[2023]["cik"] == "0000034088"
    assert rec_by_year[2023]["form"] == "10-K"

    assert rec_by_year[2024]["accession_number"] == "0000034088-25-000010"
    assert rec_by_year[2024]["fiscal_year_status"] == "confirmed"
    assert rec_by_year[2024]["cik"] == "0000034088"
    assert rec_by_year[2024]["form"] == "10-K"

    assert rec_by_year[2025]["accession_number"] == "0000034088-26-000045"
    assert rec_by_year[2025]["fiscal_year_status"] == "confirmed"
    assert rec_by_year[2025]["cik"] == "0000034088"
    assert rec_by_year[2025]["form"] == "10-K"


def test_cli_verification(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Test CLI execution prints expected summary fields."""
    from src import sec_data

    monkeypatch.setenv("SEC_USER_AGENT", "Fyntrace/0.1 Tester tester@example-corp.com")
    monkeypatch.setattr("sys.argv", ["sec_data", "--ticker", "MSFT"])

    with patch.object(sec_data, "resolve_ticker", return_value={"cik": "0000789019", "ticker": "MSFT", "company_name": "MICROSOFT CORP"}), \
         patch.object(sec_data, "get_companyfacts", return_value=MOCK_FACTS_PAYLOAD):
        sec_data.main()

    captured = capsys.readouterr()
    assert "Company: MICROSOFT CORP" in captured.out
    assert "Ticker: MSFT" in captured.out
    assert "CIK: 0000789019" in captured.out
    assert "Cache status:" in captured.out
    assert "US-GAAP concepts: 2" in captured.out
    assert "Source URL: https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json" in captured.out


def test_cli_missing_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI exits cleanly with error code when User-Agent is missing."""
    from src import sec_data

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr("sys.argv", ["sec_data", "--ticker", "MSFT"])

    with pytest.raises(SystemExit) as exc:
        sec_data.main()

    assert exc.value.code == 1


def test_cli_collect_panel(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Test CLI execution for --collect-panel flag."""
    from src import sec_data

    monkeypatch.setenv("SEC_USER_AGENT", "Fyntrace/0.1 Tester tester@example-corp.com")
    monkeypatch.setattr("sys.argv", ["sec_data", "--collect-panel"])

    dummy_summary = {
        "companies_processed": 30,
        "companies_successful": 30,
        "annual_filings_identified": 88,
        "confirmed_records": 88,
        "needs_review_records": 0,
        "missing_records": 2,
        "download_failures": 0,
        "manifest_path": "data/processed/filing_manifest.json",
    }

    with patch.object(sec_data, "collect_panel_data", return_value=dummy_summary):
        sec_data.main()

    captured = capsys.readouterr()
    assert "Companies processed: 30" in captured.out
    assert "Companies successful: 30" in captured.out
    assert "Annual filings identified: 88" in captured.out
    assert "Confirmed fiscal-year records: 88" in captured.out
    assert "Records needing review: 0" in captured.out
    assert "Missing records: 2" in captured.out
    assert "Download failures: 0" in captured.out


MOCK_AAPL_MANIFEST_RECORD = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "cik": "0000320193",
    "sector": "Technology",
    "target_fiscal_year": 2024,
    "fiscal_year_status": "confirmed",
    "form": "10-K",
    "accession_number": "0000320193-24-000100",
    "filing_date": "2024-11-01",
    "report_date": "2024-09-30",
    "primary_document": "aapl-20240930.htm",
    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000100/aapl-20240930.htm",
}


def test_fact_1_annual_revenue_extraction() -> None:
    """Test 1: Correct annual revenue extraction."""
    facts_data = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 391035000000,
                                "form": "10-K",
                                "fp": "FY",
                                "fy": 2024,
                                "accn": "0000320193-24-000100",
                                "start": "2023-10-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "revenue", facts_data)
    assert res["status"] == "found"
    assert res["value"] == 391035000000
    assert res["unit"] == "USD"
    assert res["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert res["period_start"] == "2023-10-01"
    assert res["period_end"] == "2024-09-30"
    assert res["accession"] == "0000320193-24-000100"
    assert res["reason"] is None


def test_fact_2_instant_asset_extraction() -> None:
    """Test 2: Correct instant asset extraction."""
    facts_data = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "val": 364980000000,
                                "form": "10-K",
                                "fp": "FY",
                                "fy": 2024,
                                "accn": "0000320193-24-000100",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "assets", facts_data)
    assert res["status"] == "found"
    assert res["value"] == 364980000000
    assert res["unit"] == "USD"
    assert res["tag"] == "Assets"
    assert res["period_start"] is None
    assert res["period_end"] == "2024-09-30"
    assert res["reason"] is None


def test_fact_3_june_fiscal_year_end_msft() -> None:
    """Test 3: Microsoft fiscal year ending in June."""
    msft_rec = {
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
    msft_facts = {
        "cik": 789019,
        "entityName": "MICROSOFT CORP",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 88136000000,
                                "form": "10-K",
                                "fp": "FY",
                                "fy": 2024,
                                "accn": "0000950170-24-087843",
                                "start": "2023-07-01",
                                "end": "2024-06-30",
                                "filed": "2024-07-30",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(msft_rec, "net_income", msft_facts)
    assert res["status"] == "found"
    assert res["value"] == 88136000000
    assert res["period_start"] == "2023-07-01"
    assert res["period_end"] == "2024-06-30"


def test_fact_4_january_fiscal_year_end_wmt() -> None:
    """Test 4: Walmart-style fiscal year ending in January."""
    wmt_rec = {
        "ticker": "WMT",
        "company_name": "Walmart Inc.",
        "cik": "0000104169",
        "sector": "Consumer",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000104169-24-000056",
        "filing_date": "2024-03-22",
        "report_date": "2024-01-31",
        "primary_document": "wmt-20240131.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/104169/000010416924000056/wmt-20240131.htm",
    }
    wmt_facts = {
        "cik": 104169,
        "entityName": "Walmart Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "val": 252399000000,
                                "form": "10-K",
                                "fp": "FY",
                                "fy": 2024,
                                "accn": "0000104169-24-000056",
                                "end": "2024-01-31",
                                "filed": "2024-03-22",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(wmt_rec, "assets", wmt_facts)
    assert res["status"] == "found"
    assert res["value"] == 252399000000
    assert res["period_end"] == "2024-01-31"


def test_fact_5_xom_historical_reporting_cik() -> None:
    """Test 5: ExxonMobil's historical reporting CIK."""
    xom_rec = {
        "ticker": "XOM",
        "company_name": "EXXON MOBIL CORP",
        "cik": "0000034088",
        "sector": "Energy and utilities",
        "target_fiscal_year": 2025,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000034088-26-000045",
        "filing_date": "2026-02-25",
        "report_date": "2025-12-31",
        "primary_document": "xom-20251231.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm",
    }
    xom_facts = {
        "cik": 34088,
        "entityName": "EXXON MOBIL CORP",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 28844000000,
                                "form": "10-K",
                                "fp": "FY",
                                "fy": 2025,
                                "accn": "0000034088-26-000045",
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "filed": "2026-02-25",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(xom_rec, "net_income", xom_facts)
    assert res["status"] == "found"
    assert res["cik"] == "0000034088"
    assert res["company_name"] == "EXXON MOBIL CORP"
    assert res["value"] == 28844000000
    assert res["accession"] == "0000034088-26-000045"


def test_fact_6_rejection_of_quarterly_revenue() -> None:
    """Test 6: Rejection of quarterly revenue (<330 days)."""
    facts_quarterly = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 94930000000,
                                "form": "10-K",
                                "fp": "Q4",
                                "fy": 2024,
                                "accn": "0000320193-24-000100",
                                "start": "2024-07-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "revenue", facts_quarterly)
    assert res["status"] == "missing"
    assert res["value"] is None
    assert "No matching 10-K facts" in res["reason"]


def test_fact_7_rejection_of_prior_year_comparison_figures() -> None:
    """Test 7: Rejection of prior-year comparison figures from the wrong period."""
    facts_comparative = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 383285000000,
                                "form": "10-K",
                                "fp": "FY",
                                "fy": 2023,
                                "accn": "0000320193-24-000100",
                                "start": "2022-10-01",
                                "end": "2023-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "revenue", facts_comparative)
    assert res["status"] == "missing"
    assert res["value"] is None


def test_fact_8_missing_accounting_tag() -> None:
    """Test 8: Missing accounting tag produces missing status and reason."""
    facts_no_opinc = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {"units": {"USD": []}}
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "operating_income", facts_no_opinc)
    assert res["status"] == "missing"
    assert res["value"] is None
    assert res["reason"] is not None


def test_fact_9_conflicting_financial_values() -> None:
    """Test 9: Conflicting financial values marked ambiguous."""
    facts_conflicting = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 300000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "start": "2023-10-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 350000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "start": "2023-10-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "revenue", facts_conflicting)
    assert res["status"] == "ambiguous"
    assert res["value"] is None
    assert "Conflicting candidate values" in res["reason"]


def test_fact_10_incorrect_currency_or_unit() -> None:
    """Test 10: Non-USD currency or unit rejection."""
    facts_eur = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "EUR": [
                            {
                                "val": 300000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "assets", facts_eur)
    assert res["status"] == "missing"
    assert res["value"] is None


def test_fact_11_original_filing_accession_selection() -> None:
    """Test 11: Original filing accession selection ignores amended accession."""
    facts_with_amendment = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "val": 364980000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            },
                            {
                                "val": 999999999999,
                                "form": "10-K",
                                "accn": "0000320193-25-000999",
                                "end": "2024-09-30",
                                "filed": "2025-02-01",
                            },
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(MOCK_AAPL_MANIFEST_RECORD, "assets", facts_with_amendment)
    assert res["status"] == "found"
    assert res["value"] == 364980000000
    assert res["accession"] == "0000320193-24-000100"


def test_fact_12_normalize_panel_facts_complete_450_records(tmp_path: Path) -> None:
    """Test 12: Correct creation of 450 output records from a complete mocked panel."""
    raw_dir = tmp_path / "raw"
    companyfacts_dir = raw_dir / "companyfacts"
    companyfacts_dir.mkdir(parents=True)

    manifest_records = []
    for sector, tickers in COMPANY_PANEL.items():
        for ticker in tickers:
            cik_int = hash(ticker) % 900000 + 100000
            cik_str = f"{cik_int:010d}"
            if ticker == "XOM":
                cik_str = "0000034088"

            for fy in [2023, 2024, 2025]:
                accn = f"{cik_str}-{str(fy)[2:]}-000001"
                rep_date = f"{fy}-12-31"
                manifest_records.append(
                    {
                        "ticker": ticker,
                        "company_name": f"{ticker} Inc",
                        "cik": cik_str,
                        "sector": sector,
                        "target_fiscal_year": fy,
                        "fiscal_year_status": "confirmed",
                        "form": "10-K",
                        "accession_number": accn,
                        "filing_date": f"{fy+1}-02-15",
                        "report_date": rep_date,
                        "primary_document": f"{ticker.lower()}-{fy}.htm",
                        "source_url": f"https://www.sec.gov/{accn}",
                    }
                )

            # Generate mocked company facts for this company
            c_facts = {
                "cik": int(cik_str),
                "entityName": f"{ticker} Inc",
                "facts": {
                    "us-gaap": {
                        "Assets": {
                            "units": {
                                "USD": [
                                    {
                                        "val": 1000000,
                                        "form": "10-K",
                                        "fp": "FY",
                                        "fy": fy,
                                        "accn": f"{cik_str}-{str(fy)[2:]}-000001",
                                        "end": f"{fy}-12-31",
                                        "filed": f"{fy+1}-02-15",
                                    }
                                    for fy in [2023, 2024, 2025]
                                ]
                            }
                        },
                        "NetIncomeLoss": {
                            "units": {
                                "USD": [
                                    {
                                        "val": 50000,
                                        "form": "10-K",
                                        "fp": "FY",
                                        "fy": fy,
                                        "accn": f"{cik_str}-{str(fy)[2:]}-000001",
                                        "start": f"{fy}-01-01",
                                        "end": f"{fy}-12-31",
                                        "filed": f"{fy+1}-02-15",
                                    }
                                    for fy in [2023, 2024, 2025]
                                ]
                            }
                        },
                    }
                },
            }
            facts_file = companyfacts_dir / f"CIK{cik_str}.json"
            facts_file.write_text(json.dumps(c_facts), encoding="utf-8")

    manifest_path = tmp_path / "filing_manifest.json"
    manifest_path.write_text(json.dumps(manifest_records), encoding="utf-8")

    out_file = tmp_path / "facts.jsonl"
    summary = normalize_panel_facts(
        manifest_path=manifest_path,
        cache_dir=raw_dir,
        output_path=out_file,
    )

    assert summary["companies_processed"] == 30
    assert summary["fiscal_year_records_processed"] == 90
    assert summary["total_metric_records"] == 450
    assert summary["found_records"] == 180  # assets (90) + net_income (90)
    assert summary["missing_records"] == 270  # liabilities (90) + operating_income (90) + revenue (90)
    assert summary["ambiguous_records"] == 0

    assert out_file.is_file()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 450

    seen_keys = set()
    for row in lines:
        key = (row["ticker"], row["fiscal_year"], row["metric"])
        assert key not in seen_keys
        seen_keys.add(key)
    assert len(seen_keys) == 450


def test_manifest_validation(tmp_path: Path) -> None:
    """Verify load_filing_manifest enforces all validation rules."""
    # Test valid manifest
    manifest_records = []
    for sector, tickers in COMPANY_PANEL.items():
        for ticker in tickers:
            cik_str = "0000034088" if ticker == "XOM" else f"{hash(ticker)%900000+100000:010d}"
            for fy in [2023, 2024, 2025]:
                manifest_records.append(
                    {
                        "ticker": ticker,
                        "company_name": f"{ticker} Inc",
                        "cik": cik_str,
                        "sector": sector,
                        "target_fiscal_year": fy,
                        "fiscal_year_status": "confirmed",
                        "form": "10-K",
                        "accession_number": f"{cik_str}-{str(fy)[2:]}-000001",
                        "filing_date": f"{fy+1}-02-15",
                        "report_date": f"{fy}-12-31",
                        "primary_document": f"{ticker.lower()}-{fy}.htm",
                        "source_url": f"https://www.sec.gov/{cik_str}",
                    }
                )

    valid_path = tmp_path / "valid_manifest.json"
    valid_path.write_text(json.dumps(manifest_records), encoding="utf-8")
    loaded = load_filing_manifest(valid_path)
    assert len(loaded) == 90

    # Test count mismatch (< 90)
    short_path = tmp_path / "short_manifest.json"
    short_path.write_text(json.dumps(manifest_records[:89]), encoding="utf-8")
    with pytest.raises(ValueError) as exc1:
        load_filing_manifest(short_path)
    assert "90 records" in str(exc1.value)

    # Test unconfirmed status
    unconfirmed = [dict(r) for r in manifest_records]
    unconfirmed[0]["fiscal_year_status"] = "needs_review"
    unconfirmed_path = tmp_path / "unconfirmed_manifest.json"
    unconfirmed_path.write_text(json.dumps(unconfirmed), encoding="utf-8")
    with pytest.raises(ValueError) as exc2:
        load_filing_manifest(unconfirmed_path)
    assert "expected 'confirmed'" in str(exc2.value)


def test_cli_normalize_facts(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Test CLI execution for --normalize-facts flag."""
    from src import sec_data

    monkeypatch.setattr("sys.argv", ["sec_data", "--normalize-facts"])

    dummy_summary = {
        "companies_processed": 30,
        "fiscal_year_records_processed": 90,
        "total_metric_records": 450,
        "found_records": 363,
        "missing_records": 77,
        "ambiguous_records": 10,
        "counts_by_metric": {
            "assets": {"found": 90, "missing": 0, "ambiguous": 0},
            "liabilities": {"found": 66, "missing": 24, "ambiguous": 0},
            "net_income": {"found": 88, "missing": 2, "ambiguous": 0},
            "operating_income": {"found": 54, "missing": 36, "ambiguous": 0},
            "revenue": {"found": 65, "missing": 15, "ambiguous": 10},
        },
        "output_path": "data/processed/facts.jsonl",
    }

    with patch.object(sec_data, "normalize_panel_facts", return_value=dummy_summary):
        sec_data.main()

    captured = capsys.readouterr()
    assert "Companies processed: 30" in captured.out
    assert "Fiscal-year records processed: 90" in captured.out
    assert "Total metric records: 450" in captured.out
    assert "Found records: 363" in captured.out
    assert "Missing records: 77" in captured.out
    assert "Ambiguous records: 10" in captured.out
    assert "Counts by metric:" in captured.out
    assert "assets: 90 found, 0 missing, 0 ambiguous" in captured.out
    assert "revenue: 65 found, 15 missing, 10 ambiguous" in captured.out


# ---------------------------------------------------------------------------
# Tests for verified revenue concepts and tags
# ---------------------------------------------------------------------------


def test_pfe_fy2023_prefers_verified_total_revenues() -> None:
    """1. PFE FY2023 prefers verified total Revenues over contract revenue."""
    manifest_pfe = {
        "ticker": "PFE",
        "company_name": "PFIZER INC",
        "cik": "0000078003",
        "sector": "Healthcare",
        "target_fiscal_year": 2023,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000078003-24-000039",
        "filing_date": "2024-02-22",
        "report_date": "2023-12-31",
        "primary_document": "pfe-20231231.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/78003/000007800324000039/pfe-20231231.htm",
    }
    facts = {
        "cik": 78003,
        "entityName": "PFIZER INC",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 50000000000,
                                "form": "10-K",
                                "accn": "0000078003-24-000039",
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-22",
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 58496000000,
                                "form": "10-K",
                                "accn": "0000078003-24-000039",
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-22",
                            }
                        ]
                    }
                },
            }
        },
    }
    res = select_financial_fact(manifest_pfe, "revenue", facts)
    assert res["status"] == "found"
    assert res["value"] == 58496000000
    assert res["tag"] == "Revenues"
    assert res["accession"] == "0000078003-24-000039"


def test_wmt_fy2024_chooses_comprehensive_revenue() -> None:
    """2. WMT FY2024 chooses comprehensive revenue instead of smaller customer-contract component."""
    manifest_wmt = {
        "ticker": "WMT",
        "company_name": "Walmart Inc.",
        "cik": "0000104169",
        "sector": "Consumer",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000104169-24-000056",
        "filing_date": "2024-03-22",
        "report_date": "2024-01-31",
        "primary_document": "wmt-20240131.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/104169/000010416924000056/wmt-20240131.htm",
    }
    facts = {
        "cik": 104169,
        "entityName": "Walmart Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 642637000000,
                                "form": "10-K",
                                "accn": "0000104169-24-000056",
                                "start": "2023-02-01",
                                "end": "2024-01-31",
                                "filed": "2024-03-22",
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 648125000000,
                                "form": "10-K",
                                "accn": "0000104169-24-000056",
                                "start": "2023-02-01",
                                "end": "2024-01-31",
                                "filed": "2024-03-22",
                            }
                        ]
                    }
                },
            }
        },
    }
    res = select_financial_fact(manifest_wmt, "revenue", facts)
    assert res["status"] == "found"
    assert res["value"] == 648125000000
    assert res["tag"] == "Revenues"


@pytest.mark.parametrize(
    "ticker,cik,accn,val",
    [
        ("CVX", "0000093410", "0000093410-25-000009", 202792000000),
        ("COP", "0001163165", "0001163165-25-000012", 54745000000),
    ],
)
def test_cvx_and_cop_preserve_correct_source_accession_and_periods(
    ticker: str, cik: str, accn: str, val: int
) -> None:
    """3. CVX and COP preserve correct source accession and reporting periods."""
    manifest_rec = {
        "ticker": ticker,
        "company_name": f"{ticker} Corp",
        "cik": cik,
        "sector": "Energy and utilities",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": accn,
        "filing_date": "2025-02-20",
        "report_date": "2024-12-31",
        "primary_document": f"{ticker.lower()}-20241231.htm",
        "source_url": f"https://www.sec.gov/{accn}",
    }
    facts = {
        "cik": int(cik),
        "entityName": f"{ticker} Corp",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Quarterly period (should be filtered out)
                            {
                                "val": 12345000000,
                                "form": "10-K",
                                "accn": accn,
                                "start": "2024-10-01",
                                "end": "2024-12-31",
                                "filed": "2025-02-20",
                            },
                            # Different accession (should be filtered out)
                            {
                                "val": 999999000000,
                                "form": "10-K",
                                "accn": f"{cik}-25-999999",
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-02-20",
                            },
                            # Correct full-year matching accession
                            {
                                "val": val,
                                "form": "10-K",
                                "accn": accn,
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-02-20",
                            },
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(manifest_rec, "revenue", facts)
    assert res["status"] == "found"
    assert res["value"] == val
    assert res["accession"] == accn
    assert res["tag"] == "Revenues"


def test_non_approved_company_with_conflicting_revenue_tags_remains_ambiguous() -> None:
    """4. A non-approved company with conflicting revenue tags remains ambiguous."""
    manifest_unapproved = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "sector": "Technology",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000320193-24-000100",
        "filing_date": "2024-11-01",
        "report_date": "2024-09-30",
        "primary_document": "aapl-20240930.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000100/aapl-20240930.htm",
    }
    facts = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 391035000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "start": "2023-10-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 395000000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "start": "2023-10-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
            }
        },
    }
    res = select_financial_fact(manifest_unapproved, "revenue", facts)
    assert res["status"] == "ambiguous"
    assert res["value"] is None
    assert "Conflicting candidate values" in res["reason"]


def test_conflicting_values_under_same_preferred_tag_remain_ambiguous() -> None:
    """5. Conflicting values under the same preferred tag remain ambiguous."""
    manifest_pfe = {
        "ticker": "PFE",
        "company_name": "PFIZER INC",
        "cik": "0000078003",
        "sector": "Healthcare",
        "target_fiscal_year": 2023,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000078003-24-000039",
        "filing_date": "2024-02-22",
        "report_date": "2023-12-31",
        "primary_document": "pfe-20231231.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/78003/000007800324000039/pfe-20231231.htm",
    }
    facts = {
        "cik": 78003,
        "entityName": "PFIZER INC",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 58496000000,
                                "form": "10-K",
                                "accn": "0000078003-24-000039",
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-22",
                            },
                            {
                                "val": 59000000000,
                                "form": "10-K",
                                "accn": "0000078003-24-000039",
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-22",
                            },
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(manifest_pfe, "revenue", facts)
    assert res["status"] == "ambiguous"
    assert res["value"] is None
    assert "Conflicting candidate values" in res["reason"]


def test_missing_preferred_comprehensive_revenue_does_not_silently_return_partial_revenue() -> None:
    """6. Missing preferred comprehensive revenue does not silently return partial revenue."""
    manifest_pfe = {
        "ticker": "PFE",
        "company_name": "PFIZER INC",
        "cik": "0000078003",
        "sector": "Healthcare",
        "target_fiscal_year": 2023,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000078003-24-000039",
        "filing_date": "2024-02-22",
        "report_date": "2023-12-31",
        "primary_document": "pfe-20231231.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/78003/000007800324000039/pfe-20231231.htm",
    }
    facts = {
        "cik": 78003,
        "entityName": "PFIZER INC",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 50000000000,
                                "form": "10-K",
                                "accn": "0000078003-24-000039",
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "filed": "2024-02-22",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(manifest_pfe, "revenue", facts)
    assert res["status"] == "missing"
    assert res["value"] is None
    assert "Verified comprehensive revenue tag" in res["reason"]


@pytest.mark.parametrize(
    "ticker,cik,accn,val",
    [
        ("GS", "0000886982", "0000886982-25-000005", 53512000000),
        ("MS", "0000895421", "0000895421-25-000304", 61761000000),
        ("WFC", "0000072971", "0000072971-25-000066", 82296000000),
    ],
)
def test_banks_can_use_revenues_net_of_interest_expense(
    ticker: str, cik: str, accn: str, val: int
) -> None:
    """7. GS, MS, and WFC can use RevenuesNetOfInterestExpense."""
    manifest_bank = {
        "ticker": ticker,
        "company_name": f"{ticker} Inc",
        "cik": cik,
        "sector": "Financial services",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": accn,
        "filing_date": "2025-02-25",
        "report_date": "2024-12-31",
        "primary_document": f"{ticker.lower()}-20241231.htm",
        "source_url": f"https://www.sec.gov/{accn}",
    }
    facts = {
        "cik": int(cik),
        "entityName": f"{ticker} Inc",
        "facts": {
            "us-gaap": {
                "RevenuesNetOfInterestExpense": {
                    "units": {
                        "USD": [
                            {
                                "val": val,
                                "form": "10-K",
                                "accn": accn,
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-02-25",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(manifest_bank, "revenue", facts)
    assert res["status"] == "found"
    assert res["value"] == val
    assert res["tag"] == "RevenuesNetOfInterestExpense"
    assert res["accession"] == accn


@pytest.mark.parametrize(
    "ticker,cik,accn,val",
    [
        ("DUK", "0001326160", "0001326160-25-000072", 30357000000),
        ("NEE", "0000753308", "0000753308-25-000011", 24753000000),
    ],
)
def test_utilities_can_use_regulated_and_unregulated_operating_revenue(
    ticker: str, cik: str, accn: str, val: int
) -> None:
    """8. DUK and NEE can use RegulatedAndUnregulatedOperatingRevenue."""
    manifest_util = {
        "ticker": ticker,
        "company_name": f"{ticker} Corp",
        "cik": cik,
        "sector": "Energy and utilities",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": accn,
        "filing_date": "2025-02-25",
        "report_date": "2024-12-31",
        "primary_document": f"{ticker.lower()}-20241231.htm",
        "source_url": f"https://www.sec.gov/{accn}",
    }
    facts = {
        "cik": int(cik),
        "entityName": f"{ticker} Corp",
        "facts": {
            "us-gaap": {
                "RegulatedAndUnregulatedOperatingRevenue": {
                    "units": {
                        "USD": [
                            {
                                "val": val,
                                "form": "10-K",
                                "accn": accn,
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-02-25",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(manifest_util, "revenue", facts)
    assert res["status"] == "found"
    assert res["value"] == val
    assert res["tag"] == "RegulatedAndUnregulatedOperatingRevenue"
    assert res["accession"] == accn


def test_new_concepts_not_automatically_used_for_unrelated_companies() -> None:
    """9. The new concepts are not automatically used for unrelated companies."""
    manifest_unrelated = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "sector": "Technology",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000320193-24-000100",
        "filing_date": "2024-11-01",
        "report_date": "2024-09-30",
        "primary_document": "aapl-20240930.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000100/aapl-20240930.htm",
    }
    facts = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenuesNetOfInterestExpense": {
                    "units": {
                        "USD": [
                            {
                                "val": 100000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "start": "2023-10-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
                "RegulatedAndUnregulatedOperatingRevenue": {
                    "units": {
                        "USD": [
                            {
                                "val": 200000000,
                                "form": "10-K",
                                "accn": "0000320193-24-000100",
                                "start": "2023-10-01",
                                "end": "2024-09-30",
                                "filed": "2024-11-01",
                            }
                        ]
                    }
                },
            }
        },
    }
    res = select_financial_fact(manifest_unrelated, "revenue", facts)
    assert res["status"] == "missing"
    assert res["value"] is None


def test_so_net_income_remains_missing_when_only_profit_loss_exists() -> None:
    """10. SO net income remains missing when only ProfitLoss exists."""
    manifest_so = {
        "ticker": "SO",
        "company_name": "SOUTHERN CO",
        "cik": "0000092122",
        "sector": "Energy and utilities",
        "target_fiscal_year": 2024,
        "fiscal_year_status": "confirmed",
        "form": "10-K",
        "accession_number": "0000092122-25-000015",
        "filing_date": "2025-02-19",
        "report_date": "2024-12-31",
        "primary_document": "so-20241231.htm",
        "source_url": "https://www.sec.gov/Archives/edgar/data/92122/000009212225000015/so-20241231.htm",
    }
    facts_profit_loss_only = {
        "cik": 92122,
        "entityName": "SOUTHERN CO",
        "facts": {
            "us-gaap": {
                "ProfitLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 4443000000,
                                "form": "10-K",
                                "accn": "0000092122-25-000015",
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-02-19",
                            }
                        ]
                    }
                }
            }
        },
    }
    res = select_financial_fact(manifest_so, "net_income", facts_profit_loss_only)
    assert res["status"] == "missing"
    assert res["value"] is None

