"""SEC data client for retrieving company facts, submissions, and ticker mappings."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ENV_PATH = BASE_DIR / ".env"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Rate limit delay to stay comfortably below SEC's 10 requests/sec limit
SEC_REQUEST_INTERVAL = 0.2
DEFAULT_TIMEOUT = 15.0

_PLACEHOLDER_SUBSTRINGS = (
    "yourrealname",
    "your-real-email",
    "your_real_email",
    "example.com",
    "sample company",
    "user@example",
    "placeholder",
    "your-email",
    "your_email",
    "admincontact@<sample",
)


class SecSession(requests.Session):
    """Session subclass that enforces polite rate limiting and default timeouts."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, min_interval: float = SEC_REQUEST_INTERVAL):
        super().__init__()
        self.default_timeout = timeout
        self.min_interval = min_interval
        self._last_request_time = 0.0

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.default_timeout)
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()
        return super().request(method, url, **kwargs)


def load_project_env(env_path: Path | None = None) -> None:
    """Load environment variables from the project's .env file only."""
    target_path = env_path if env_path is not None else PROJECT_ENV_PATH
    if target_path.is_file():
        load_dotenv(dotenv_path=target_path)


def validate_user_agent(user_agent: str | None) -> str:
    """Validate that the User-Agent is present, genuine, and not a placeholder."""
    if not user_agent or not isinstance(user_agent, str) or not user_agent.strip():
        raise ValueError("Missing SEC User-Agent. Please configure SEC_USER_AGENT.")

    cleaned = user_agent.strip()
    cleaned_lower = cleaned.lower()

    if any(p in cleaned_lower for p in _PLACEHOLDER_SUBSTRINGS) or "<" in cleaned or ">" in cleaned:
        raise ValueError(
            "Invalid or placeholder SEC User-Agent. Genuine contact information is required by the SEC."
        )

    # Must contain a contact email address conforming to standard SEC requirements
    if not re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cleaned):
        raise ValueError(
            "Invalid SEC User-Agent: missing a valid contact email address. SEC requires genuine contact details."
        )

    return cleaned


def make_sec_session(user_agent: str | None = None, env_path: Path | None = None) -> requests.Session:
    """Create a configured requests.Session for SEC API requests."""
    load_project_env(env_path)
    ua = user_agent if user_agent is not None else os.environ.get("SEC_USER_AGENT")
    valid_ua = validate_user_agent(ua)

    session = SecSession(timeout=DEFAULT_TIMEOUT, min_interval=SEC_REQUEST_INTERVAL)
    session.headers.update(
        {
            "User-Agent": valid_ua,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
    )

    retry_strategy = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def sec_get_json(url: str, session: requests.Session | None = None) -> dict:
    """Fetch JSON from a trusted SEC URL, handling errors and rate limits."""
    if session is None:
        session = make_sec_session()

    try:
        response = session.get(url)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error requesting SEC URL {url}: {exc}") from exc

    if response.status_code == 429:
        raise requests.HTTPError(f"Rate limited by SEC (HTTP 429) for {url}", response=response)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"HTTP error {response.status_code} for {url}", response=response) from exc

    try:
        return response.json()
    except Exception as exc:
        raise ValueError(f"Malformed JSON response from SEC endpoint {url}: {exc}") from exc


def normalize_cik(cik: str | int) -> str:
    """Validate and zero-pad a CIK to 10 digits.

    Accepts positive numeric identifiers up to ten digits.
    Rejects negative, zero, nonnumeric, and longer-than-ten-digit identifiers.
    """
    if cik is None:
        raise ValueError("CIK cannot be None.")
    cik_str = str(cik).strip()
    if not cik_str.isdigit():
        raise ValueError(f"Invalid CIK: '{cik}'. CIK must be a numeric integer.")
    if len(cik_str) > 10:
        raise ValueError(f"Invalid CIK: '{cik}'. CIK identifier must not exceed 10 digits.")
    cik_int = int(cik_str)
    if cik_int <= 0:
        raise ValueError(f"Invalid CIK: '{cik}'. CIK must be greater than zero.")
    return f"{cik_int:010d}"


def _atomic_write_json(file_path: Path, data: Any) -> None:
    """Write data to a temporary file first, then atomically replace the target."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, file_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def validate_companyfacts_response(data: Any, expected_cik: str) -> None:
    """Validate that company-facts response is a dictionary with valid CIK and facts."""
    if not isinstance(data, dict):
        raise ValueError("Invalid company facts response: payload must be a JSON object.")
    if "facts" not in data or not isinstance(data["facts"], dict):
        raise ValueError("Invalid company facts response: missing or invalid 'facts' dictionary.")
    if "cik" not in data:
        raise ValueError("Invalid company facts response: missing 'cik' field.")
    try:
        response_cik = normalize_cik(data["cik"])
    except ValueError as exc:
        raise ValueError(f"Invalid company facts response: invalid 'cik' in payload ({exc}).") from exc

    expected_norm = normalize_cik(expected_cik)
    if response_cik != expected_norm:
        raise ValueError(
            f"CIK mismatch in company facts response: requested CIK {expected_norm}, got CIK {response_cik}."
        )


def validate_submissions_response(data: Any, expected_cik: str) -> None:
    """Validate that submissions response is a dictionary with valid CIK and filings."""
    if not isinstance(data, dict):
        raise ValueError("Invalid submissions response: payload must be a JSON object.")
    if "filings" not in data or not isinstance(data["filings"], dict):
        raise ValueError("Invalid submissions response: missing or invalid 'filings' dictionary.")
    if "cik" not in data:
        raise ValueError("Invalid submissions response: missing 'cik' field.")
    try:
        response_cik = normalize_cik(data["cik"])
    except ValueError as exc:
        raise ValueError(f"Invalid submissions response: invalid 'cik' in payload ({exc}).") from exc

    expected_norm = normalize_cik(expected_cik)
    if response_cik != expected_norm:
        raise ValueError(
            f"CIK mismatch in submissions response: requested CIK {expected_norm}, got CIK {response_cik}."
        )


def validate_ticker_payload(raw_data: Any) -> None:
    """Validate company ticker mapping before writing it to cache."""
    if not isinstance(raw_data, (dict, list)):
        raise ValueError("Malformed company tickers payload: expected a JSON object or array.")

    items = raw_data.values() if isinstance(raw_data, dict) else raw_data
    valid_count = 0
    for item in items:
        if isinstance(item, dict):
            ticker_val = item.get("ticker")
            cik_val = item.get("cik_str", item.get("cik"))
            if ticker_val and cik_val is not None:
                try:
                    normalize_cik(cik_val)
                    valid_count += 1
                except ValueError:
                    continue

    if valid_count == 0:
        raise ValueError("Malformed company tickers data: no valid ticker entries found.")


def load_company_tickers(
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Download or load cached company ticker mapping from SEC."""
    raw_dir = Path(cache_dir) if cache_dir is not None else RAW_DATA_DIR
    cache_file = raw_dir / "company_tickers.json"

    raw_data: Any = None
    if not force_refresh:
        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to read cached company tickers from {cache_file}: {exc}") from exc

    if raw_data is None:
        if session is None:
            session = make_sec_session()
        raw_data = sec_get_json("https://www.sec.gov/files/company_tickers.json", session=session)
        validate_ticker_payload(raw_data)
        _atomic_write_json(cache_file, raw_data)
    else:
        validate_ticker_payload(raw_data)

    mapping: dict[str, dict[str, str]] = {}
    items = raw_data.values() if isinstance(raw_data, dict) else raw_data
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker_val = item.get("ticker")
        cik_val = item.get("cik_str", item.get("cik"))
        title_val = item.get("title", item.get("company_name", ""))
        if ticker_val is not None and cik_val is not None:
            try:
                cik_norm = normalize_cik(cik_val)
            except ValueError:
                continue
            ticker_clean = str(ticker_val).strip().upper()
            mapping[ticker_clean] = {
                "cik": cik_norm,
                "ticker": ticker_clean,
                "company_name": str(title_val).strip(),
            }

    return mapping


def resolve_ticker(
    ticker: str,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Resolve a ticker symbol to its official company name and 10-digit CIK."""
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker symbol must not be empty.")

    normalized_ticker = ticker.strip().upper()
    tickers = load_company_tickers(session=session, cache_dir=cache_dir)
    if normalized_ticker not in tickers:
        raise KeyError(f"Unknown ticker symbol '{ticker}'. Not found in SEC company mapping.")

    return tickers[normalized_ticker]


def get_submissions(
    cik: str | int,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict:
    """Download or retrieve cached company submissions metadata from SEC."""
    cik_norm = normalize_cik(cik)
    raw_dir = Path(cache_dir) if cache_dir is not None else RAW_DATA_DIR
    cache_file = raw_dir / "submissions" / f"CIK{cik_norm}.json"

    if not force_refresh:
        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to read cached submissions file {cache_file}: {exc}") from exc

    if session is None:
        session = make_sec_session()

    url = f"https://data.sec.gov/submissions/CIK{cik_norm}.json"
    data = sec_get_json(url, session=session)
    validate_submissions_response(data, cik_norm)
    _atomic_write_json(cache_file, data)
    return data


def get_companyfacts(
    cik: str | int,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict:
    """Download or retrieve cached official company financial facts from SEC."""
    cik_norm = normalize_cik(cik)
    raw_dir = Path(cache_dir) if cache_dir is not None else RAW_DATA_DIR
    cache_file = raw_dir / "companyfacts" / f"CIK{cik_norm}.json"

    if not force_refresh:
        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to read cached companyfacts file {cache_file}: {exc}") from exc

    if session is None:
        session = make_sec_session()

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_norm}.json"
    data = sec_get_json(url, session=session)
    validate_companyfacts_response(data, cik_norm)
    _atomic_write_json(cache_file, data)
    return data


def main() -> None:
    """CLI entrypoint for verifying ticker resolution and company facts."""
    parser = argparse.ArgumentParser(description="SEC data client CLI")
    parser.add_argument("--ticker", required=True, help="Company ticker symbol (e.g. MSFT)")
    args = parser.parse_args()

    try:
        session = make_sec_session()
    except ValueError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        sys.exit(1)

    try:
        company = resolve_ticker(args.ticker, session=session)
    except Exception as exc:
        sys.stderr.write(f"Error resolving ticker '{args.ticker}': {exc}\n")
        sys.exit(1)

    cik_norm = normalize_cik(company["cik"])
    cache_file = RAW_DATA_DIR / "companyfacts" / f"CIK{cik_norm}.json"
    is_cached = cache_file.is_file()
    cache_status = "cached" if is_cached else "downloaded"

    try:
        facts = get_companyfacts(cik_norm, session=session)
    except Exception as exc:
        sys.stderr.write(f"Error fetching company facts for CIK {cik_norm}: {exc}\n")
        sys.exit(1)

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    concept_count = len(us_gaap) if isinstance(us_gaap, dict) else 0
    source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_norm}.json"

    print(f"Company: {company['company_name']}")
    print(f"Ticker: {company['ticker']}")
    print(f"CIK: {cik_norm}")
    print(f"Cache status: {cache_status}")
    print(f"US-GAAP concepts: {concept_count}")
    print(f"Source URL: {source_url}")


if __name__ == "__main__":
    main()
