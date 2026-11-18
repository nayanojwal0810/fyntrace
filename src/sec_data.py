"""SEC data client for retrieving company facts, submissions, and ticker mappings."""

import argparse
from datetime import datetime
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
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Rate limit delay to stay comfortably below SEC's 10 requests/sec limit
SEC_REQUEST_INTERVAL = 0.2
DEFAULT_TIMEOUT = 15.0

COLLECTION_CUTOFF_DATE = "2026-11-18"
TARGET_FISCAL_YEARS = [2023, 2024, 2025]

COMPANY_PANEL: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
    "Financial services": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH", "ABT"],
    "Energy and utilities": ["XOM", "CVX", "COP", "NEE", "DUK", "SO"],
    "Consumer": ["WMT", "COST", "PG", "KO", "PEP", "MCD"],
}

PANEL_TICKERS: dict[str, str] = {
    ticker: sector for sector, tickers in COMPANY_PANEL.items() for ticker in tickers
}

HISTORICAL_PANEL_REGISTRANTS: dict[str, dict[str, str]] = {
    "XOM": {
        "cik": "0000034088",
        "company_name": "EXXON MOBIL CORP",
    }
}

FINANCIAL_METRICS: dict[str, dict[str, Any]] = {
    "assets": {
        "tags": ["Assets"],
        "kind": "instant",
        "unit": "USD",
    },
    "liabilities": {
        "tags": ["Liabilities"],
        "kind": "instant",
        "unit": "USD",
    },
    "net_income": {
        "tags": ["NetIncomeLoss"],
        "kind": "duration",
        "unit": "USD",
    },
    "operating_income": {
        "tags": ["OperatingIncomeLoss"],
        "kind": "duration",
        "unit": "USD",
    },
    "revenue": {
        "tags": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ],
        "kind": "duration",
        "unit": "USD",
    },
}

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


def filing_url(cik: str | int, accession: str, primary_document: str) -> str:
    """Construct official SEC archive URL for a filing document."""
    cik_number = str(int(cik))
    accession_compact = accession.replace("-", "")
    doc_basename = Path(primary_document).name
    return f"https://www.sec.gov/Archives/edgar/data/{cik_number}/{accession_compact}/{doc_basename}"


def list_filings(
    cik: str | int,
    forms: list[str] | tuple[str, ...] | set[str] | None = None,
    as_of: str | None = None,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract filing records from SEC submissions matching requested forms and cutoff."""
    cik_norm = normalize_cik(cik)
    submissions = get_submissions(cik_norm, session=session, cache_dir=cache_dir)

    raw_dir = Path(cache_dir) if cache_dir is not None else RAW_DATA_DIR
    target_forms = set(forms) if forms is not None else None

    results: list[dict[str, Any]] = []

    def _process_filing_arrays(recent_dict: dict[str, Any]) -> None:
        accessions = recent_dict.get("accessionNumber", [])
        form_list = recent_dict.get("form", [])
        filing_dates = recent_dict.get("filingDate", [])
        report_dates = recent_dict.get("reportDate", [])
        primary_docs = recent_dict.get("primaryDocument", [])

        if not (isinstance(accessions, list) and isinstance(form_list, list)):
            return

        n = min(
            len(accessions),
            len(form_list),
            len(filing_dates) if isinstance(filing_dates, list) else 0,
            len(report_dates) if isinstance(report_dates, list) else 0,
            len(primary_docs) if isinstance(primary_docs, list) else 0,
        )

        for i in range(n):
            form_val = str(form_list[i]).strip()
            if target_forms is not None and form_val not in target_forms:
                continue

            filing_date_val = str(filing_dates[i]).strip() if filing_dates[i] else ""
            if as_of is not None and filing_date_val and filing_date_val > as_of:
                continue

            accn_val = str(accessions[i]).strip()
            report_date_val = str(report_dates[i]).strip() if report_dates[i] else ""
            doc_val = str(primary_docs[i]).strip() if primary_docs[i] else ""

            results.append(
                {
                    "accession_number": accn_val,
                    "form": form_val,
                    "filing_date": filing_date_val,
                    "report_date": report_date_val,
                    "primary_document": doc_val,
                    "cik": cik_norm,
                    "source_url": filing_url(cik_norm, accn_val, doc_val) if doc_val else "",
                }
            )

    filings_obj = submissions.get("filings", {})
    if isinstance(filings_obj, dict):
        recent = filings_obj.get("recent", {})
        if isinstance(recent, dict):
            _process_filing_arrays(recent)

        # Use older submission pages from filings.files only when necessary (e.g. recent does not reach 2023)
        filing_dates = recent.get("filingDate", []) if isinstance(recent, dict) else []
        oldest_recent = min(filing_dates) if filing_dates else ""
        if oldest_recent and oldest_recent > "2023-01-01":
            files_list = filings_obj.get("files", [])
            if isinstance(files_list, list) and files_list:
                for file_info in files_list:
                    if not isinstance(file_info, dict):
                        continue
                    filing_to = file_info.get("filingTo", "")
                    if filing_to and filing_to < "2023-01-01":
                        continue
                    name = file_info.get("name")
                    if not name:
                        continue
                    old_cache = raw_dir / "submissions" / name
                    old_data: Any = None
                    if old_cache.is_file():
                        try:
                            with open(old_cache, "r", encoding="utf-8") as f:
                                old_data = json.load(f)
                        except Exception:
                            pass
                    if old_data is None and session is not None:
                        try:
                            old_url = f"https://data.sec.gov/submissions/{name}"
                            old_data = sec_get_json(old_url, session=session)
                            _atomic_write_json(old_cache, old_data)
                        except Exception:
                            pass
                    if isinstance(old_data, dict):
                        _process_filing_arrays(old_data)

    return results


def extract_xbrl_10k_fiscal_years(companyfacts_data: dict) -> dict[str, int]:
    """Map Form 10-K accession numbers to their XBRL-reported fiscal years.

    Only unanimous fiscal years are mapped; ambiguous accessions are omitted.
    """
    accn_to_years: dict[str, set[int]] = {}
    facts = companyfacts_data.get("facts", {})
    if not isinstance(facts, dict):
        return {}

    for taxonomy in ("us-gaap", "dei"):
        tax_dict = facts.get(taxonomy, {})
        if not isinstance(tax_dict, dict):
            continue
        for concept_data in tax_dict.values():
            if not isinstance(concept_data, dict):
                continue
            units_dict = concept_data.get("units", {})
            if not isinstance(units_dict, dict):
                continue
            for rows in units_dict.values():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if row.get("form") == "10-K" and row.get("fp") == "FY":
                        accn = row.get("accn")
                        fy = row.get("fy")
                        if accn and isinstance(fy, int):
                            accn_to_years.setdefault(accn, set()).add(fy)

    confirmed_map: dict[str, int] = {}
    for accn, years in accn_to_years.items():
        if len(years) == 1:
            confirmed_map[accn] = next(iter(years))
    return confirmed_map


def select_annual_filings_for_company(
    cik: str | int,
    ticker: str,
    company_name: str,
    sector: str,
    target_years: list[int] = TARGET_FISCAL_YEARS,
    as_of: str = COLLECTION_CUTOFF_DATE,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Select 10-K filings for target fiscal years with confirmed/needs_review/missing status."""
    cik_norm = normalize_cik(cik)
    all_filings = list_filings(
        cik_norm,
        forms=["10-K", "10-K/A"],
        as_of=as_of,
        session=session,
        cache_dir=cache_dir,
    )

    try:
        facts_data = get_companyfacts(cik_norm, session=session, cache_dir=cache_dir)
        xbrl_fy_map = extract_xbrl_10k_fiscal_years(facts_data)
    except Exception:
        xbrl_fy_map = {}

    fy_to_10k: dict[int, list[tuple[dict[str, Any], str]]] = {}
    fy_to_10ka: dict[int, list[tuple[dict[str, Any], str]]] = {}

    for f in all_filings:
        accn = f["accession_number"]
        form_name = f["form"]
        r_date = f["report_date"]

        r_year: int | None = None
        if r_date and len(r_date) >= 4 and r_date[:4].isdigit():
            r_year = int(r_date[:4])

        if accn in xbrl_fy_map:
            assigned_fy = xbrl_fy_map[accn]
            if r_year is not None and abs(r_year - assigned_fy) <= 1:
                status = "confirmed" if form_name == "10-K" else "needs_review"
            else:
                status = "needs_review"
        elif r_year is not None:
            assigned_fy = r_year
            status = "needs_review"
        else:
            continue

        if form_name == "10-K":
            fy_to_10k.setdefault(assigned_fy, []).append((f, status))
        elif form_name == "10-K/A":
            fy_to_10ka.setdefault(assigned_fy, []).append((f, "needs_review"))

    manifest_records: list[dict[str, Any]] = []

    for year in target_years:
        k_candidates = fy_to_10k.get(year, [])
        ka_candidates = fy_to_10ka.get(year, [])

        if k_candidates:
            if len(k_candidates) == 1:
                filing, st = k_candidates[0]
                manifest_records.append(
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "cik": cik_norm,
                        "sector": sector,
                        "target_fiscal_year": year,
                        "fiscal_year_status": st,
                        "form": filing["form"],
                        "accession_number": filing["accession_number"],
                        "filing_date": filing["filing_date"],
                        "report_date": filing["report_date"],
                        "primary_document": filing["primary_document"],
                        "source_url": filing["source_url"],
                    }
                )
            else:
                filing, _ = k_candidates[0]
                manifest_records.append(
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "cik": cik_norm,
                        "sector": sector,
                        "target_fiscal_year": year,
                        "fiscal_year_status": "needs_review",
                        "form": filing["form"],
                        "accession_number": filing["accession_number"],
                        "filing_date": filing["filing_date"],
                        "report_date": filing["report_date"],
                        "primary_document": filing["primary_document"],
                        "source_url": filing["source_url"],
                    }
                )
        elif ka_candidates:
            filing, _ = ka_candidates[0]
            manifest_records.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": cik_norm,
                    "sector": sector,
                    "target_fiscal_year": year,
                    "fiscal_year_status": "needs_review",
                    "form": filing["form"],
                    "accession_number": filing["accession_number"],
                    "filing_date": filing["filing_date"],
                    "report_date": filing["report_date"],
                    "primary_document": filing["primary_document"],
                    "source_url": filing["source_url"],
                }
            )
        else:
            manifest_records.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": cik_norm,
                    "sector": sector,
                    "target_fiscal_year": year,
                    "fiscal_year_status": "missing",
                    "form": None,
                    "accession_number": None,
                    "filing_date": None,
                    "report_date": None,
                    "primary_document": None,
                    "source_url": None,
                }
            )

    return manifest_records


def collect_panel_data(
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    as_of: str = COLLECTION_CUTOFF_DATE,
) -> dict[str, Any]:
    """Process the 30-company panel, download/reuse SEC JSON data, and build filing manifest."""
    if session is None:
        session = make_sec_session()

    raw_dir = Path(cache_dir) if cache_dir is not None else RAW_DATA_DIR
    save_manifest_path = Path(output_path) if output_path is not None else (PROCESSED_DATA_DIR / "filing_manifest.json")

    companies_processed = 0
    companies_successful = 0
    download_failures = 0
    manifest_records: list[dict[str, Any]] = []

    for sector, tickers in COMPANY_PANEL.items():
        for ticker in tickers:
            companies_processed += 1
            try:
                if ticker in HISTORICAL_PANEL_REGISTRANTS:
                    hist_info = HISTORICAL_PANEL_REGISTRANTS[ticker]
                    cik = hist_info["cik"]
                    company_name = hist_info["company_name"]
                else:
                    company = resolve_ticker(ticker, session=session, cache_dir=raw_dir)
                    cik = company["cik"]
                    company_name = company["company_name"]

                get_submissions(cik, session=session, cache_dir=raw_dir)
                get_companyfacts(cik, session=session, cache_dir=raw_dir)

                records = select_annual_filings_for_company(
                    cik=cik,
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector,
                    target_years=TARGET_FISCAL_YEARS,
                    as_of=as_of,
                    session=session,
                    cache_dir=raw_dir,
                )
                manifest_records.extend(records)
                companies_successful += 1
            except Exception:
                download_failures += 1
                for year in TARGET_FISCAL_YEARS:
                    manifest_records.append(
                        {
                            "ticker": ticker,
                            "company_name": ticker,
                            "cik": "",
                            "sector": sector,
                            "target_fiscal_year": year,
                            "fiscal_year_status": "missing",
                            "form": None,
                            "accession_number": None,
                            "filing_date": None,
                            "report_date": None,
                            "primary_document": None,
                            "source_url": None,
                        }
                    )

    confirmed_count = sum(1 for r in manifest_records if r["fiscal_year_status"] == "confirmed")
    needs_review_count = sum(1 for r in manifest_records if r["fiscal_year_status"] == "needs_review")
    missing_count = sum(1 for r in manifest_records if r["fiscal_year_status"] == "missing")
    annual_filings_identified = sum(1 for r in manifest_records if r.get("accession_number"))

    _atomic_write_json(save_manifest_path, manifest_records)

    summary = {
        "companies_processed": companies_processed,
        "companies_successful": companies_successful,
        "annual_filings_identified": annual_filings_identified,
        "confirmed_records": confirmed_count,
        "needs_review_records": needs_review_count,
        "missing_records": missing_count,
        "download_failures": download_failures,
        "manifest_path": str(save_manifest_path),
    }

    return summary


def load_filing_manifest(manifest_path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the filing manifest from disk.

    Validates:
    - Exactly 90 records.
    - Exactly 30 unique companies.
    - Three target fiscal years per company: 2023, 2024, 2025.
    - No duplicate company-year combinations.
    - Every selected filing has a valid accession number, CIK, filing date, and report date.
    - All records have 'confirmed' status.
    """
    path = Path(manifest_path) if manifest_path is not None else (PROCESSED_DATA_DIR / "filing_manifest.json")
    if not path.is_file():
        raise FileNotFoundError(f"Filing manifest file not found at {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to read filing manifest from {path}: {exc}") from exc

    if not isinstance(records, list):
        raise ValueError("Invalid filing manifest: expected a JSON array.")

    if len(records) != 90:
        raise ValueError(f"Filing manifest must contain exactly 90 records, got {len(records)}.")

    seen_companies: set[str] = set()
    seen_company_years: set[tuple[str, int]] = set()
    company_years: dict[str, set[int]] = {}

    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("Manifest record must be a JSON object.")
        ticker = rec.get("ticker")
        fy = rec.get("target_fiscal_year")
        status = rec.get("fiscal_year_status")
        accn = rec.get("accession_number")
        cik = rec.get("cik")
        filing_date = rec.get("filing_date")
        report_date = rec.get("report_date")

        if not ticker or not isinstance(ticker, str):
            raise ValueError("Manifest record missing valid 'ticker'.")
        if not isinstance(fy, int):
            raise ValueError(f"Manifest record for '{ticker}' missing valid integer 'target_fiscal_year'.")
        if status != "confirmed":
            raise ValueError(f"Manifest record for '{ticker}' FY{fy} status is '{status}', expected 'confirmed'.")
        if not accn or not isinstance(accn, str):
            raise ValueError(f"Manifest record for '{ticker}' FY{fy} missing valid 'accession_number'.")
        if not cik:
            raise ValueError(f"Manifest record for '{ticker}' FY{fy} missing valid 'cik'.")
        if not filing_date or not isinstance(filing_date, str):
            raise ValueError(f"Manifest record for '{ticker}' FY{fy} missing valid 'filing_date'.")
        if not report_date or not isinstance(report_date, str):
            raise ValueError(f"Manifest record for '{ticker}' FY{fy} missing valid 'report_date'.")

        pair = (ticker, fy)
        if pair in seen_company_years:
            raise ValueError(f"Duplicate company-year in manifest: {pair}")
        seen_company_years.add(pair)
        seen_companies.add(ticker)
        company_years.setdefault(ticker, set()).add(fy)

    if len(seen_companies) != 30:
        raise ValueError(f"Filing manifest must contain exactly 30 unique companies, got {len(seen_companies)}.")

    for ticker, years in company_years.items():
        if years != {2023, 2024, 2025}:
            raise ValueError(
                f"Company '{ticker}' does not have target fiscal years 2023, 2024, 2025: got {sorted(years)}."
            )

    return records


def select_financial_fact(
    manifest_record: dict[str, Any],
    metric_name: str,
    facts_data: dict[str, Any],
    as_of: str = COLLECTION_CUTOFF_DATE,
) -> dict[str, Any]:
    """Select and normalize a single financial fact for a company filing record."""
    if metric_name not in FINANCIAL_METRICS:
        raise ValueError(f"Unknown metric '{metric_name}'. Must be one of {list(FINANCIAL_METRICS.keys())}.")

    metric_config = FINANCIAL_METRICS[metric_name]
    candidate_tags: list[str] = metric_config["tags"]
    kind: str = metric_config["kind"]
    target_unit: str = metric_config["unit"]

    base_record: dict[str, Any] = {
        "ticker": manifest_record["ticker"],
        "company_name": manifest_record["company_name"],
        "cik": manifest_record["cik"],
        "sector": manifest_record["sector"],
        "fiscal_year": manifest_record["target_fiscal_year"],
        "metric": metric_name,
        "status": "missing",
        "value": None,
        "unit": target_unit,
        "taxonomy": "us-gaap",
        "tag": None,
        "period_start": None,
        "period_end": manifest_record.get("report_date"),
        "filed_date": manifest_record.get("filing_date"),
        "form": manifest_record.get("form") or "10-K",
        "accession": manifest_record.get("accession_number"),
        "source_url": manifest_record.get("source_url") or "",
        "selection_policy": "original_filing",
        "reason": None,
    }

    facts_dict = facts_data.get("facts", {}) if isinstance(facts_data, dict) else {}
    us_gaap = facts_dict.get("us-gaap", {}) if isinstance(facts_dict, dict) else {}
    if not isinstance(us_gaap, dict):
        base_record["reason"] = "No US-GAAP concepts in company facts"
        return base_record

    target_accn = manifest_record.get("accession_number")
    target_report_date = manifest_record.get("report_date")

    candidates: list[dict[str, Any]] = []

    for tag in candidate_tags:
        concept_data = us_gaap.get(tag)
        if not isinstance(concept_data, dict):
            continue
        units_dict = concept_data.get("units", {})
        if not isinstance(units_dict, dict):
            continue
        rows = units_dict.get(target_unit, [])
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("form") != "10-K":
                continue
            if row.get("accn") != target_accn:
                continue
            row_filed = row.get("filed")
            if row_filed and as_of and str(row_filed) > as_of:
                continue
            row_end = row.get("end")
            if not row_end or row_end != target_report_date:
                continue

            row_start = row.get("start")
            if kind == "duration":
                if not row_start:
                    continue
                try:
                    d_start = datetime.strptime(row_start, "%Y-%m-%d")
                    d_end = datetime.strptime(row_end, "%Y-%m-%d")
                    duration_days = (d_end - d_start).days
                except (ValueError, TypeError):
                    continue
                if not (330 <= duration_days <= 400):
                    continue
            elif kind == "instant":
                # For instant metrics, start date is not required
                pass

            val = row.get("val")
            if val is None:
                continue

            candidates.append(
                {
                    "tag": tag,
                    "val": val,
                    "start": row_start if kind == "duration" else None,
                    "end": row_end,
                    "filed": row_filed or manifest_record.get("filing_date"),
                    "accn": row.get("accn"),
                }
            )

    # Remove duplicate equivalent records (same tag, value, period, accession)
    unique_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for c in candidates:
        key = (c["tag"], c["val"], c["start"], c["end"], c["accn"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_candidates.append(c)

    if not unique_candidates:
        base_record["status"] = "missing"
        base_record["reason"] = f"No matching 10-K facts found for tags: {candidate_tags}"
        return base_record

    distinct_vals = {c["val"] for c in unique_candidates}
    if len(distinct_vals) == 1:
        # All candidates agree on value!
        # Pick preferred candidate according to candidate_tags order
        unique_candidates.sort(key=lambda x: candidate_tags.index(x["tag"]))
        chosen = unique_candidates[0]
        base_record["status"] = "found"
        base_record["value"] = chosen["val"]
        base_record["tag"] = chosen["tag"]
        base_record["period_start"] = chosen["start"]
        base_record["period_end"] = chosen["end"]
        base_record["filed_date"] = chosen["filed"]
        base_record["reason"] = None
        return base_record

    # Multiple candidates disagree on value
    base_record["status"] = "ambiguous"
    vals_summary = {c["tag"]: c["val"] for c in unique_candidates}
    base_record["reason"] = f"Conflicting candidate values found: {vals_summary}"
    return base_record


def normalize_panel_facts(
    manifest_path: Path | None = None,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    as_of: str = COLLECTION_CUTOFF_DATE,
) -> dict[str, Any]:
    """Normalize annual financial facts for panel companies and write facts.jsonl."""
    raw_dir = Path(cache_dir) if cache_dir is not None else RAW_DATA_DIR
    out_file = Path(output_path) if output_path is not None else (PROCESSED_DATA_DIR / "facts.jsonl")

    manifest = load_filing_manifest(manifest_path)

    # Cache companyfacts in-memory during extraction
    loaded_facts: dict[str, dict[str, Any]] = {}

    all_fact_records: list[dict[str, Any]] = []
    unique_tickers: set[str] = set()

    for manifest_rec in manifest:
        ticker = manifest_rec["ticker"]
        unique_tickers.add(ticker)
        cik = manifest_rec["cik"]

        if cik not in loaded_facts:
            facts_file = raw_dir / "companyfacts" / f"CIK{cik}.json"
            if facts_file.is_file():
                try:
                    with open(facts_file, "r", encoding="utf-8") as f:
                        loaded_facts[cik] = json.load(f)
                except Exception as exc:
                    raise ValueError(f"Failed to read company facts from {facts_file}: {exc}") from exc
            else:
                loaded_facts[cik] = get_companyfacts(cik, cache_dir=raw_dir)

        facts_data = loaded_facts[cik]

        for metric_name in FINANCIAL_METRICS:
            fact_rec = select_financial_fact(manifest_rec, metric_name, facts_data, as_of=as_of)
            all_fact_records.append(fact_rec)

    # Write out_file atomically
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_file.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for rec in all_fact_records:
                f.write(json.dumps(rec) + "\n")
        os.replace(temp_path, out_file)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    found_count = sum(1 for r in all_fact_records if r["status"] == "found")
    missing_count = sum(1 for r in all_fact_records if r["status"] == "missing")
    ambiguous_count = sum(1 for r in all_fact_records if r["status"] == "ambiguous")

    counts_by_metric: dict[str, dict[str, int]] = {}
    for metric_name in FINANCIAL_METRICS:
        m_records = [r for r in all_fact_records if r["metric"] == metric_name]
        counts_by_metric[metric_name] = {
            "found": sum(1 for r in m_records if r["status"] == "found"),
            "missing": sum(1 for r in m_records if r["status"] == "missing"),
            "ambiguous": sum(1 for r in m_records if r["status"] == "ambiguous"),
        }

    return {
        "companies_processed": len(unique_tickers),
        "fiscal_year_records_processed": len(manifest),
        "total_metric_records": len(all_fact_records),
        "found_records": found_count,
        "missing_records": missing_count,
        "ambiguous_records": ambiguous_count,
        "counts_by_metric": counts_by_metric,
        "output_path": str(out_file),
    }


def main() -> None:
    """CLI entrypoint for verifying ticker resolution and company facts or panel collection."""
    parser = argparse.ArgumentParser(description="SEC data client CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Company ticker symbol (e.g. MSFT)")
    group.add_argument(
        "--collect-panel",
        action="store_true",
        help="Collect data and generate annual filing manifest for the 30-company panel",
    )
    group.add_argument(
        "--normalize-facts",
        action="store_true",
        help="Normalize annual financial facts from cached SEC data",
    )
    args = parser.parse_args()

    if args.normalize_facts:
        try:
            summary = normalize_panel_facts()
        except Exception as exc:
            sys.stderr.write(f"Error normalizing financial facts: {exc}\n")
            sys.exit(1)

        print(f"Companies processed: {summary['companies_processed']}")
        print(f"Fiscal-year records processed: {summary['fiscal_year_records_processed']}")
        print(f"Total metric records: {summary['total_metric_records']}")
        print(f"Found records: {summary['found_records']}")
        print(f"Missing records: {summary['missing_records']}")
        print(f"Ambiguous records: {summary['ambiguous_records']}")
        print("Counts by metric:")
        for metric, counts in summary["counts_by_metric"].items():
            print(f"  {metric}: {counts['found']} found, {counts['missing']} missing, {counts['ambiguous']} ambiguous")
        return

    try:
        session = make_sec_session()
    except ValueError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        sys.exit(1)

    if args.collect_panel:
        try:
            summary = collect_panel_data(session=session)
        except Exception as exc:
            sys.stderr.write(f"Error during panel collection: {exc}\n")
            sys.exit(1)

        print(f"Companies processed: {summary['companies_processed']}")
        print(f"Companies successful: {summary['companies_successful']}")
        print(f"Annual filings identified: {summary['annual_filings_identified']}")
        print(f"Confirmed fiscal-year records: {summary['confirmed_records']}")
        print(f"Records needing review: {summary['needs_review_records']}")
        print(f"Missing records: {summary['missing_records']}")
        print(f"Download failures: {summary['download_failures']}")
        return

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
