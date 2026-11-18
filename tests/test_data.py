"""Unit tests for the SEC data client (offline, mocked)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from src.sec_data import (
    get_companyfacts,
    get_submissions,
    load_company_tickers,
    make_sec_session,
    normalize_cik,
    resolve_ticker,
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
            "Assets": {"label": "Assets", "units": {"USD": [{"val": 100}]}},
            "Revenues": {"label": "Revenues", "units": {"USD": [{"val": 200}]}},
        },
    },
}

MOCK_SUBMISSIONS_PAYLOAD = {
    "cik": "0000789019",
    "entityType": "operating",
    "name": "MICROSOFT CORP",
    "filings": {
        "recent": {
            "accessionNumber": ["0000789019-24-000050"],
            "form": ["10-K"],
        }
    },
}


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure all tests run isolated from any real local .env file or environment variable."""
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    isolated_env = tmp_path / ".env.isolated"
    monkeypatch.setattr("src.sec_data.PROJECT_ENV_PATH", isolated_env)


def create_mock_session(status_code: int = 200, json_data: dict | list | None = None, raise_http_error: bool = False) -> MagicMock:
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
    # HTTP error (e.g. 404)
    error_session = create_mock_session(status_code=404, raise_http_error=True)
    with pytest.raises(requests.HTTPError):
        get_companyfacts("0000789019", session=error_session, cache_dir=tmp_path)

    # HTTP 429 Rate limit
    rate_limit_session = create_mock_session(status_code=429)
    with pytest.raises(requests.HTTPError) as exc_info:
        get_companyfacts("0000789019", session=rate_limit_session, cache_dir=tmp_path)
    assert "429" in str(exc_info.value)

    # Malformed JSON
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

    # Placeholder patterns
    with pytest.raises(ValueError) as exc1:
        validate_user_agent("Fyntrace/0.1 YourRealName your-real-email@example.com")
    assert "placeholder" in str(exc1.value).lower()

    with pytest.raises(ValueError):
        validate_user_agent("Sample Company Name AdminContact@<sample company domain>.com")

    # Missing email address
    with pytest.raises(ValueError) as exc2:
        validate_user_agent("MyCompanyResearchTeam")
    assert "contact email" in str(exc2.value).lower()

    # Valid User-Agent
    valid = validate_user_agent("Fyntrace/0.1 ResearchTeam research@valid-domain.org")
    assert valid == "Fyntrace/0.1 ResearchTeam research@valid-domain.org"

    # make_sec_session rejects placeholder
    with pytest.raises(ValueError):
        make_sec_session(user_agent="Fyntrace/0.1 YourRealName your-real-email@example.com")

    # make_sec_session rejects missing User-Agent
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

    # Force download with an error-producing session
    failing_session = create_mock_session(status_code=500, raise_http_error=True)

    with pytest.raises(requests.HTTPError):
        get_companyfacts("0000789019", session=failing_session, cache_dir=tmp_path, force_refresh=True)

    # Verify original file content is completely intact
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

    # Calling get_companyfacts must NOT read submissions cache
    mock_session = create_mock_session(json_data=MOCK_FACTS_PAYLOAD)
    facts = get_companyfacts("0000789019", session=mock_session, cache_dir=tmp_path)
    assert facts["cik"] == 789019
    assert (tmp_path / "companyfacts" / "CIK0000789019.json").is_file()
    mock_session.get.assert_called_once()

    # Ambiguous root file must not be read by either function
    tmp_path2 = tmp_path / "isolated_ambiguous"
    tmp_path2.mkdir()
    ambiguous_file = tmp_path2 / "CIK0000789019.json"
    ambiguous_file.write_text(json.dumps({"cik": 789019, "entityName": "AMBIGUOUS"}), encoding="utf-8")

    mock_session2 = create_mock_session(json_data=MOCK_FACTS_PAYLOAD)
    facts2 = get_companyfacts("0000789019", session=mock_session2, cache_dir=tmp_path2)
    assert facts2["entityName"] == "MICROSOFT CORP"
    mock_session2.get.assert_called_once()


def test_rejection_of_mismatched_cik_responses(tmp_path: Path) -> None:
    """Reject responses where returned CIK does not match requested CIK."""
    # Mismatched company facts
    mismatched_facts = {
        "cik": 320193,  # Apple CIK
        "entityName": "Apple Inc.",
        "facts": {"us-gaap": {}},
    }
    mock_session = create_mock_session(json_data=mismatched_facts)
    with pytest.raises(ValueError) as exc_facts:
        get_companyfacts("0000789019", session=mock_session, cache_dir=tmp_path)
    assert "mismatch" in str(exc_facts.value).lower()
    assert not (tmp_path / "companyfacts" / "CIK0000789019.json").exists()

    # Mismatched submissions
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
    # Facts payload not a dict
    session_not_dict = create_mock_session(json_data=["not", "a", "dict"])
    with pytest.raises(ValueError) as exc1:
        get_companyfacts("0000789019", session=session_not_dict, cache_dir=tmp_path)
    assert "json object" in str(exc1.value).lower()

    # Facts missing "facts" key
    bad_facts = {"cik": 789019, "entityName": "MICROSOFT CORP"}
    session_bad_facts = create_mock_session(json_data=bad_facts)
    with pytest.raises(ValueError) as exc2:
        get_companyfacts("0000789019", session=session_bad_facts, cache_dir=tmp_path)
    assert "facts" in str(exc2.value).lower()

    # Submissions missing "filings" key
    bad_sub = {"cik": "0000789019", "name": "MICROSOFT CORP"}
    session_bad_sub = create_mock_session(json_data=bad_sub)
    with pytest.raises(ValueError) as exc3:
        get_submissions("0000789019", session=session_bad_sub, cache_dir=tmp_path)
    assert "filings" in str(exc3.value).lower()

    # Malformed ticker payload
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


def test_cli_verification(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Test CLI execution prints expected summary fields."""
    from unittest.mock import patch
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
