"""Unit tests for the SEC data client and filing discovery (offline, mocked)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.sec_data import (
    COLLECTION_CUTOFF_DATE,
    COMPANY_PANEL,
    PANEL_TICKERS,
    TARGET_FISCAL_YEARS,
    collect_panel_data,
    extract_xbrl_10k_fiscal_years,
    filing_url,
    get_companyfacts,
    get_submissions,
    list_filings,
    load_company_tickers,
    make_sec_session,
    normalize_cik,
    resolve_ticker,
    select_annual_filings_for_company,
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

    # Mock submissions and facts
    sub_dir = raw_dir / "submissions"
    sub_dir.mkdir(parents=True)
    (sub_dir / "CIK0000789019.json").write_text(json.dumps(MOCK_SUBMISSIONS_PAYLOAD), encoding="utf-8")

    facts_dir = raw_dir / "companyfacts"
    facts_dir.mkdir(parents=True)
    (facts_dir / "CIK0000789019.json").write_text(json.dumps(MOCK_FACTS_PAYLOAD), encoding="utf-8")

    out_file = tmp_path / "processed" / "filing_manifest.json"
    mock_session = create_mock_session()
    summary = collect_panel_data(session=mock_session, cache_dir=raw_dir, output_path=out_file)

    assert summary["companies_processed"] == 30
    assert summary["companies_successful"] == 30
    assert summary["download_failures"] == 0
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 90  # 30 companies * 3 years


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
