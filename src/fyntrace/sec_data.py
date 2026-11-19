"""SEC data collection, validation, and financial fact normalization."""

import argparse
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import dotenv
import requests

from fyntrace.config import (
    ALL_TICKERS,
    COMPANY_PANEL,
    SUPPORTED_FISCAL_YEARS,
    get_company,
    is_supported_ticker,
    is_supported_year,
)

# Approved SEC domain endpoints
SEC_WWW_HOST = "www.sec.gov"
SEC_DATA_HOST = "data.sec.gov"
APPROVED_SEC_HOSTS = {SEC_WWW_HOST, SEC_DATA_HOST}

TICKER_MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
HISTORICAL_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/{filename}"
COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_doc}"
)

# Historical registrant mapping for corporate restructurings
HISTORICAL_REGISTRANTS: Mapping[str, Mapping[str, str]] = {
    "XOM": {
        "cik": "0000034088",
        "name": "EXXON MOBIL CORP",
    }
}

# Standard financial metrics and candidate concepts
SUPPORTED_METRICS = ("assets", "liabilities", "net_income", "operating_income", "revenue")
REVENUE_CANDIDATE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)

METRIC_CONFIG: Mapping[str, Mapping[str, Any]] = {
    "assets": {
        "kind": "instant",
        "tags": ("Assets",),
    },
    "liabilities": {
        "kind": "instant",
        "tags": ("Liabilities",),
    },
    "net_income": {
        "kind": "duration",
        "tags": ("NetIncomeLoss",),
    },
    "operating_income": {
        "kind": "duration",
        "tags": ("OperatingIncomeLoss",),
    },
    "revenue": {
        "kind": "duration",
        "tags": REVENUE_CANDIDATE_TAGS,
    },
}

# Default directory structure
DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"


class SecAccessError(Exception):
    """Raised when SEC access is rejected or persistent errors occur."""


def validate_user_agent(user_agent: str | None) -> str:
    """Validate that SEC User-Agent complies with SEC fair access requirements.

    Must contain genuine identification and email contact information.
    """
    if not user_agent or not isinstance(user_agent, str):
        raise ValueError("SEC_USER_AGENT is missing or empty. Please set a valid User-Agent.")

    ua = user_agent.strip()
    if len(ua) < 10:
        raise ValueError("SEC_USER_AGENT is too short to contain genuine contact information.")

    lower = ua.lower()
    placeholders = (
        "yourrealname",
        "your-real-email",
        "example.com",
        "placeholder",
        "sample",
        "user@email.com",
        "name@domain.com",
        "<",
        ">",
    )
    for p in placeholders:
        if p in lower:
            raise ValueError(f"SEC_USER_AGENT contains placeholder text ('{p}'). A truthful User-Agent is required.")

    # Require an email-like substring: at least one '@' and a '.' after '@'
    if "@" not in ua:
        raise ValueError("SEC_USER_AGENT must include a contact email address.")
    email_part = ua.split("@", 1)[1]
    if "." not in email_part:
        raise ValueError("SEC_USER_AGENT must include a valid contact email domain.")

    return ua


def get_sec_user_agent() -> str:
    """Read and validate SEC User-Agent from environment or local .env."""
    dotenv.load_dotenv()
    ua = os.environ.get("SEC_USER_AGENT")
    return validate_user_agent(ua)


class SecRateLimiter:
    """Sequential request rate limiter adhering to SEC fair access rules (~0.4-0.5s delay)."""

    def __init__(self, min_delay_sec: float = 0.45):
        self.min_delay_sec = min_delay_sec
        self.last_request_time: float = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay_sec:
            time.sleep(self.min_delay_sec - elapsed)
        self.last_request_time = time.time()


_RATE_LIMITER = SecRateLimiter()


def validate_sec_url(url: str) -> None:
    """Ensure URL strictly uses HTTPS and approved SEC hosts."""
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only HTTPS is permitted.")
    if parsed.netloc.lower() not in APPROVED_SEC_HOSTS:
        raise ValueError(f"Invalid host '{parsed.netloc}'. Only approved SEC domains are permitted.")
    if ".." in parsed.path:
        raise ValueError(f"URL path traversal detected in '{parsed.path}'.")


def make_sec_session(user_agent: str | None = None) -> requests.Session:
    """Create a configured SEC HTTP session with headers, timeouts, and rate limiting."""
    if user_agent is None:
        user_agent = get_sec_user_agent()
    else:
        user_agent = validate_user_agent(user_agent)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def sec_request(
    session: requests.Session,
    url: str,
    stream: bool = False,
    max_retries: int = 3,
    timeout: tuple[float, float] = (5.0, 30.0),
) -> requests.Response:
    """Execute an HTTP request to the SEC with rate limiting, timeouts, and retries."""
    validate_sec_url(url)

    for attempt in range(1, max_retries + 1):
        _RATE_LIMITER.wait()
        try:
            resp = session.get(url, stream=stream, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == max_retries:
                raise SecAccessError(f"Connection failure after {max_retries} attempts: {exc}") from exc
            time.sleep(1.0 * attempt)
            continue

        if resp.status_code == 200:
            return resp
        elif resp.status_code == 403:
            raise SecAccessError("SEC access denied (HTTP 403). Verify User-Agent contact information.")
        elif resp.status_code == 404:
            raise FileNotFoundError(f"Requested SEC resource not found (HTTP 404): {url}")
        elif resp.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                raise SecAccessError(f"SEC server returned HTTP {resp.status_code} after {max_retries} retries.")
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_sec = max(1.0, float(retry_after))
                except ValueError:
                    sleep_sec = 2.0 * attempt
            else:
                sleep_sec = 2.0 * attempt
            time.sleep(sleep_sec)
        else:
            raise SecAccessError(f"Unexpected HTTP {resp.status_code} response from SEC endpoint.")

    raise SecAccessError(f"Failed to fetch {url} after {max_retries} attempts.")


def normalize_cik(cik: str | int) -> str:
    """Normalize CIK to 10-digit zero-padded string."""
    digits = re.sub(r"\D", "", str(cik))
    if not digits:
        raise ValueError(f"Invalid CIK value '{cik}'. Must contain numbers.")
    if len(digits) > 10:
        raise ValueError(f"CIK '{cik}' exceeds 10 digits.")
    return digits.zfill(10)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a file safely using atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".tmp_{path.name}_{os.getpid()}_{time.time_ns()}"
    try:
        temp_path.write_bytes(data)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON data to a file safely using atomic replacement."""
    payload = json.dumps(data, indent=2) + "\n"
    _atomic_write_bytes(path, payload.encode("utf-8"))


def load_company_tickers(
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Download or load cached SEC company tickers mapping."""
    target_dir = raw_dir or DEFAULT_RAW_DIR
    cache_file = target_dir / "company_tickers.json"

    if not force_refresh and cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    resp = sec_request(session, TICKER_MAPPING_URL)
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Invalid format received from company_tickers.json.")

    _atomic_write_json(cache_file, data)
    return data


def resolve_ticker(ticker: str, mapping: Mapping[str, Any]) -> dict[str, str]:
    """Resolve a ticker symbol to company metadata using official SEC mapping.

    Returns dict with ticker, cik (10-digit zero-padded), name, and sector.
    """
    if not is_supported_ticker(ticker):
        raise ValueError(f"Ticker '{ticker}' is not supported in the configured company panel.")

    company = get_company(ticker)
    sym = company.ticker

    # Check historical registrants first
    if sym in HISTORICAL_REGISTRANTS:
        hist = HISTORICAL_REGISTRANTS[sym]
        return {
            "ticker": sym,
            "cik": hist["cik"],
            "name": hist["name"],
            "sector": company.sector,
        }

    # Search in SEC mapping
    for entry in mapping.values():
        if isinstance(entry, dict) and entry.get("ticker", "").upper() == sym:
            cik_int = entry.get("cik_str", entry.get("cik"))
            if cik_int is not None:
                cik_padded = normalize_cik(cik_int)
                return {
                    "ticker": sym,
                    "cik": cik_padded,
                    "name": str(entry.get("title", "")).strip(),
                    "sector": company.sector,
                }

    raise KeyError(f"Ticker '{ticker}' could not be resolved from SEC mapping.")


def get_submissions(
    cik: str | int,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load cached company submissions or download from SEC."""
    cik_10 = normalize_cik(cik)
    target_dir = raw_dir or DEFAULT_RAW_DIR
    cache_file = target_dir / "submissions" / f"CIK{cik_10}.json"

    if not force_refresh and cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "filings" in data and "recent" in data["filings"]:
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik_10)
    resp = sec_request(session, url)
    data = resp.json()

    if not isinstance(data, dict) or "filings" not in data or "recent" not in data["filings"]:
        raise ValueError(f"Incomplete submissions payload received for CIK {cik_10}.")

    _atomic_write_json(cache_file, data)
    return data


def get_historical_submissions(
    filename: str,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load cached historical submissions file or download from SEC."""
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise ValueError(f"Unsafe historical submissions filename '{filename}'.")

    target_dir = raw_dir or DEFAULT_RAW_DIR
    cache_file = target_dir / "submissions" / safe_name

    if not force_refresh and cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "accessionNumber" in data:
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    url = HISTORICAL_SUBMISSIONS_URL_TEMPLATE.format(filename=safe_name)
    resp = sec_request(session, url)
    data = resp.json()
    _atomic_write_json(cache_file, data)
    return data


def get_companyfacts(
    cik: str | int,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load cached company facts or download from SEC."""
    cik_10 = normalize_cik(cik)
    target_dir = raw_dir or DEFAULT_RAW_DIR
    cache_file = target_dir / "companyfacts" / f"CIK{cik_10}.json"

    if not force_refresh and cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "facts" in data and "us-gaap" in data["facts"]:
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    url = COMPANYFACTS_URL_TEMPLATE.format(cik=cik_10)
    resp = sec_request(session, url)
    data = resp.json()

    if not isinstance(data, dict) or "facts" not in data:
        raise ValueError(f"Incomplete companyfacts payload received for CIK {cik_10}.")

    _atomic_write_json(cache_file, data)
    return data


def infer_fiscal_year_from_filing(report_date_str: str, filing_date_str: str, fye: str | None = None) -> int:
    """Determine reporting fiscal year from report date and filing date.

    Handles 52/53-week years, January year ends (NVDA, WMT), and calendar year ends.
    """
    rd = datetime.strptime(report_date_str, "%Y-%m-%d").date()

    # For 52/53 week companies whose fiscal year ends in the first week of January (e.g. JNJ 2023-01-01),
    # the reporting year belongs to the prior calendar year.
    if rd.month == 1 and rd.day <= 7:
        return rd.year - 1

    return rd.year


def build_archive_url(cik: str, accession: str, primary_doc: str) -> str:
    """Construct official SEC archive URL for primary document."""
    cik_no_zeros = str(int(re.sub(r"\D", "", cik)))
    accession_clean = accession.strip()
    if not re.match(r"^\d{10}-\d{2}-\d{6}$", accession_clean):
        raise ValueError(f"Invalid accession number format '{accession}'.")

    safe_doc = Path(primary_doc).name
    if safe_doc != primary_doc or ".." in primary_doc or "/" in primary_doc or "\\" in primary_doc:
        raise ValueError(f"Unsafe primary document path '{primary_doc}'.")

    accession_no_dashes = accession_clean.replace("-", "")
    return ARCHIVE_URL_TEMPLATE.format(
        cik_no_zeros=cik_no_zeros,
        accession_no_dashes=accession_no_dashes,
        primary_doc=safe_doc,
    )


def _scan_aligned_arrays_for_10k(
    records: Mapping[str, list[Any]],
    fiscal_year: int,
    cik: str,
    as_of: date,
    fye: str | None,
) -> list[dict[str, Any]]:
    """Scan submissions aligned arrays for Form 10-K matching fiscal_year."""
    forms = records.get("form", [])
    reports = records.get("reportDate", [])
    filings = records.get("filingDate", [])
    accns = records.get("accessionNumber", [])
    docs = records.get("primaryDocument", [])

    candidates: list[dict[str, Any]] = []
    for form, rep, fd_str, acc, doc in zip(forms, reports, filings, accns, docs):
        if form != "10-K":
            continue
        try:
            fd = datetime.strptime(fd_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if fd > as_of:
            continue

        try:
            inferred_fy = infer_fiscal_year_from_filing(rep, fd_str, fye)
        except Exception:
            continue

        if inferred_fy == fiscal_year:
            source_url = build_archive_url(cik, acc, doc)
            candidates.append(
                {
                    "cik": normalize_cik(cik),
                    "fiscal_year": fiscal_year,
                    "form": "10-K",
                    "accession": acc,
                    "filing_date": fd_str,
                    "report_date": rep,
                    "primary_document": doc,
                    "source_url": source_url,
                }
            )
    return candidates


def select_annual_filing(
    submissions: dict[str, Any],
    fiscal_year: int,
    as_of_date: str | None = None,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Select the verified Form 10-K filing for the specified fiscal year.

    Raises:
        ValueError: If multiple conflicting candidates exist.
    """
    if not is_supported_year(fiscal_year):
        raise ValueError(f"Fiscal year {fiscal_year} is not supported. Supported: {SUPPORTED_FISCAL_YEARS}")

    cik = normalize_cik(submissions.get("cik", ""))
    fye = submissions.get("fiscalYearEnd")

    as_of = date.today() if as_of_date is None else datetime.strptime(as_of_date, "%Y-%m-%d").date()

    recent = submissions.get("filings", {}).get("recent", {})
    candidates = _scan_aligned_arrays_for_10k(recent, fiscal_year, cik, as_of, fye)

    if not candidates:
        # Check historical submission files if available
        files = submissions.get("filings", {}).get("files", [])
        for file_meta in files:
            file_name = file_meta.get("name")
            if not file_name:
                continue
            hist_sub = get_historical_submissions(file_name, session=session, raw_dir=raw_dir)
            candidates = _scan_aligned_arrays_for_10k(hist_sub, fiscal_year, cik, as_of, fye)
            if candidates:
                break

    if not candidates:
        return None

    if len(candidates) > 1:
        # Check if identical accession (duplicate record)
        accns = {c["accession"] for c in candidates}
        if len(accns) > 1:
            raise ValueError(f"Ambiguous 10-K filings found for CIK {cik} FY{fiscal_year}: {accns}")

    return candidates[0]


def download_filing(
    filing: Mapping[str, Any],
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force_refresh: bool = False,
) -> tuple[Path, str]:
    """Download primary HTML document of Form 10-K and return local path and SHA-256."""
    target_dir = (raw_dir or DEFAULT_RAW_DIR) / "filings"
    target_dir.mkdir(parents=True, exist_ok=True)

    ticker = filing.get("ticker", "SEC")
    fy = filing["fiscal_year"]
    acc_clean = filing["accession"].replace("-", "")
    filename = f"{ticker}_{fy}_{acc_clean}.html"
    local_path = target_dir / filename

    if not force_refresh and local_path.is_file():
        content = local_path.read_bytes()
        if len(content) > 1000 and (b"<html" in content.lower() or b"<!doctype" in content.lower()):
            sha = hashlib.sha256(content).hexdigest()
            return local_path, sha

    if session is None:
        session = make_sec_session()

    url = filing["source_url"]
    validate_sec_url(url)
    resp = sec_request(session, url)
    content = resp.content

    if len(content) < 500:
        raise ValueError(f"Downloaded filing HTML is suspiciously small ({len(content)} bytes).")

    if b"Access Denied" in content and resp.status_code != 200:
        raise SecAccessError("Received Access Denied page when downloading filing HTML.")

    _atomic_write_bytes(local_path, content)
    sha = hashlib.sha256(content).hexdigest()
    return local_path, sha


def normalize_annual_fact(
    companyfacts: Mapping[str, Any],
    filing: Mapping[str, Any],
    metric: str,
) -> dict[str, Any] | None:
    """Normalize a financial metric fact from SEC Company Facts for a specific 10-K filing.

    Returns the normalized fact record or None if unavailable/ambiguous.
    """
    if metric not in METRIC_CONFIG:
        raise ValueError(f"Unsupported metric '{metric}'. Supported: {SUPPORTED_METRICS}")

    config = METRIC_CONFIG[metric]
    kind = config["kind"]
    candidate_tags = config["tags"]

    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return None

    accn = filing["accession"]
    report_date = filing["report_date"]
    cik = normalize_cik(companyfacts.get("cik", filing.get("cik", "")))

    selected_facts: list[dict[str, Any]] = []

    for tag in candidate_tags:
        if tag not in us_gaap:
            continue
        concept_data = us_gaap[tag]
        units = concept_data.get("units", {}).get("USD", [])
        for u in units:
            if u.get("accn") != accn:
                continue
            if u.get("form") != "10-K":
                continue
            if u.get("end") != report_date:
                continue

            # Check instant vs duration
            if kind == "duration":
                start = u.get("start")
                end = u.get("end")
                if not start or not end:
                    continue
                try:
                    d_start = datetime.strptime(start, "%Y-%m-%d").date()
                    d_end = datetime.strptime(end, "%Y-%m-%d").date()
                    duration_days = (d_end - d_start).days
                except Exception:
                    continue

                # Annual period duration verification (330 to 400 days)
                if not (330 <= duration_days <= 400):
                    continue

            val = u.get("val")
            if val is None:
                continue

            selected_facts.append(
                {
                    "taxonomy": "us-gaap",
                    "tag": tag,
                    "val": val,
                    "unit": "USD",
                    "start": u.get("start"),
                    "end": u.get("end"),
                    "filed": u.get("filed", filing.get("filing_date")),
                }
            )

    if not selected_facts:
        return None

    # For revenue candidate tags: check consistency
    if metric == "revenue":
        # Group by tag
        by_tag: dict[str, Any] = {}
        for f in selected_facts:
            by_tag[f["tag"]] = f["val"]

        unique_vals = set(by_tag.values())
        if len(unique_vals) > 1:
            # Conflicting candidate values -> ambiguous
            return None

        # Pick highest priority tag
        chosen_fact = None
        for tag in candidate_tags:
            for f in selected_facts:
                if f["tag"] == tag:
                    chosen_fact = f
                    break
            if chosen_fact:
                break
    else:
        # Assets, liabilities, net_income, operating_income
        vals = {f["val"] for f in selected_facts}
        if len(vals) > 1:
            return None
        chosen_fact = selected_facts[0]

    if not chosen_fact:
        return None

    raw_val = chosen_fact["val"]
    # Integer preservation
    if isinstance(raw_val, (int, float)):
        if isinstance(raw_val, float) and raw_val.is_integer():
            final_val: int | str = int(raw_val)
        elif isinstance(raw_val, int):
            final_val = raw_val
        else:
            final_val = str(Decimal(str(raw_val)))
    else:
        final_val = str(raw_val)

    # Compute companyfacts sha256 if possible
    cf_bytes = json.dumps(companyfacts, sort_keys=True).encode("utf-8")
    cf_sha256 = hashlib.sha256(cf_bytes).hexdigest()

    return {
        "ticker": filing.get("ticker", ""),
        "cik": cik,
        "company_name": filing.get("company_name", companyfacts.get("entityName", "")),
        "sector": filing.get("sector", ""),
        "fiscal_year": filing["fiscal_year"],
        "metric": metric,
        "taxonomy": chosen_fact["taxonomy"],
        "tag": chosen_fact["tag"],
        "value": final_val,
        "unit": "USD",
        "period_start": chosen_fact.get("start"),
        "period_end": chosen_fact.get("end"),
        "form": "10-K",
        "accession": accn,
        "filed_date": chosen_fact.get("filed"),
        "source_url": filing["source_url"],
        "selection_policy": "original_filing",
        "companyfacts_sha256": cf_sha256,
    }


def collect_data(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    as_of_date: str | None = None,
    cache_dir: Path | None = None,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """Execute complete SEC data collection, normalization, and manifest generation."""
    target_tickers = [t.strip().upper() for t in tickers] if tickers else list(ALL_TICKERS)
    target_years = sorted(years) if years else list(SUPPORTED_FISCAL_YEARS)

    for t in target_tickers:
        if not is_supported_ticker(t):
            raise ValueError(f"Unsupported ticker '{t}'. Supported: {ALL_TICKERS}")
    for y in target_years:
        if not is_supported_year(y):
            raise ValueError(f"Unsupported year {y}. Supported: {SUPPORTED_FISCAL_YEARS}")

    as_of = date.today().isoformat() if as_of_date is None else as_of_date

    raw_dir = cache_dir or DEFAULT_RAW_DIR
    processed_dir = (raw_dir.parent / "processed") if cache_dir else DEFAULT_PROCESSED_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    session = make_sec_session(user_agent)

    # 1. Load ticker mapping
    ticker_mapping = load_company_tickers(session=session, raw_dir=raw_dir)

    company_identities: list[dict[str, str]] = []
    manifest_filings: list[dict[str, Any]] = []
    all_facts: list[dict[str, Any]] = []

    successful_filings_count = 0
    failed_filings_count = 0

    # 2. Iterate companies in configured panel order
    for company in COMPANY_PANEL:
        if company.ticker not in target_tickers:
            continue

        resolved = resolve_ticker(company.ticker, ticker_mapping)
        company_identities.append(resolved)

        cik = resolved["cik"]
        name = resolved["name"]
        sector = resolved["sector"]

        try:
            sub = get_submissions(cik, session=session, raw_dir=raw_dir)
            cf = get_companyfacts(cik, session=session, raw_dir=raw_dir)
        except Exception as exc:
            for y in target_years:
                manifest_filings.append(
                    {
                        "ticker": company.ticker,
                        "fiscal_year": y,
                        "status": "failed",
                        "local_path": None,
                        "source_url": None,
                        "accession": None,
                        "filing_date": None,
                        "period_end_date": None,
                        "primary_document": None,
                        "form": "10-K",
                        "file_sha256": None,
                        "selected_metrics": [],
                        "unavailable_metrics": {m: str(exc) for m in SUPPORTED_METRICS},
                        "errors": [str(exc)],
                    }
                )
                failed_filings_count += 1
            continue

        for y in target_years:
            try:
                filing_meta = select_annual_filing(sub, y, as_of_date=as_of, session=session, raw_dir=raw_dir)
            except Exception as exc:
                manifest_filings.append(
                    {
                        "ticker": company.ticker,
                        "fiscal_year": y,
                        "status": "ambiguous",
                        "local_path": None,
                        "source_url": None,
                        "accession": None,
                        "filing_date": None,
                        "period_end_date": None,
                        "primary_document": None,
                        "form": "10-K",
                        "file_sha256": None,
                        "selected_metrics": [],
                        "unavailable_metrics": {m: "ambiguous filing" for m in SUPPORTED_METRICS},
                        "errors": [str(exc)],
                    }
                )
                failed_filings_count += 1
                continue

            if not filing_meta:
                manifest_filings.append(
                    {
                        "ticker": company.ticker,
                        "fiscal_year": y,
                        "status": "missing",
                        "local_path": None,
                        "source_url": None,
                        "accession": None,
                        "filing_date": None,
                        "period_end_date": None,
                        "primary_document": None,
                        "form": "10-K",
                        "file_sha256": None,
                        "selected_metrics": [],
                        "unavailable_metrics": {m: "filing not found in SEC submissions" for m in SUPPORTED_METRICS},
                        "errors": [],
                    }
                )
                failed_filings_count += 1
                continue

            # Populate company context
            filing_meta["ticker"] = company.ticker
            filing_meta["company_name"] = name
            filing_meta["sector"] = sector

            try:
                local_file, file_sha = download_filing(filing_meta, session=session, raw_dir=raw_dir)
            except Exception as exc:
                manifest_filings.append(
                    {
                        "ticker": company.ticker,
                        "fiscal_year": y,
                        "status": "failed",
                        "local_path": None,
                        "source_url": filing_meta["source_url"],
                        "accession": filing_meta["accession"],
                        "filing_date": filing_meta["filing_date"],
                        "period_end_date": filing_meta["report_date"],
                        "primary_document": filing_meta["primary_document"],
                        "form": "10-K",
                        "file_sha256": None,
                        "selected_metrics": [],
                        "unavailable_metrics": {m: f"HTML download failed: {exc}" for m in SUPPORTED_METRICS},
                        "errors": [f"HTML download failed: {exc}"],
                    }
                )
                failed_filings_count += 1
                continue

            selected_metrics = []
            unavailable_metrics = {}

            for m in SUPPORTED_METRICS:
                fact = normalize_annual_fact(cf, filing_meta, m)
                if fact:
                    selected_metrics.append(m)
                    all_facts.append(fact)
                else:
                    unavailable_metrics[m] = "concept unavailable or ambiguous in SEC Company Facts"

            # Local path relative to workspace if under DEFAULT_DATA_DIR
            try:
                display_path = str(local_file.as_posix())
            except Exception:
                display_path = str(local_file)

            manifest_filings.append(
                {
                    "ticker": company.ticker,
                    "fiscal_year": y,
                    "status": "downloaded",
                    "local_path": display_path,
                    "source_url": filing_meta["source_url"],
                    "accession": filing_meta["accession"],
                    "filing_date": filing_meta["filing_date"],
                    "period_end_date": filing_meta["report_date"],
                    "primary_document": filing_meta["primary_document"],
                    "form": "10-K",
                    "file_sha256": file_sha,
                    "selected_metrics": selected_metrics,
                    "unavailable_metrics": unavailable_metrics,
                    "errors": [],
                }
            )
            successful_filings_count += 1

    # Write facts.jsonl atomically
    facts_file = processed_dir / "facts.jsonl"
    facts_lines = [json.dumps(fact) + "\n" for fact in all_facts]
    _atomic_write_bytes(facts_file, "".join(facts_lines).encode("utf-8"))

    # Write manifest.json atomically
    manifest_file = processed_dir / "manifest.json"
    manifest_data = {
        "collection_timestamp": datetime.now().isoformat(),
        "as_of_date": as_of,
        "requested_company_count": len(target_tickers),
        "requested_fiscal_years": target_years,
        "sec_sources": {
            "ticker_mapping_url": TICKER_MAPPING_URL,
            "submissions_url_template": SUBMISSIONS_URL_TEMPLATE,
            "companyfacts_url_template": COMPANYFACTS_URL_TEMPLATE,
            "archive_url_template": ARCHIVE_URL_TEMPLATE,
        },
        "company_identities": company_identities,
        "summary": {
            "total_requested_filings": len(target_tickers) * len(target_years),
            "successful_filings": successful_filings_count,
            "failed_filings": failed_filings_count,
            "facts_normalized": len(all_facts),
        },
        "filings": manifest_filings,
    }
    _atomic_write_json(manifest_file, manifest_data)

    return manifest_data


def get_fact(
    ticker: str,
    metric: str,
    fiscal_year: int,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """Retrieve a previously validated fact from local processed facts.jsonl.

    Raises:
        KeyError: If fact is not found.
    """
    sym = ticker.strip().upper()
    met = metric.strip().lower()
    p_dir = processed_dir or DEFAULT_PROCESSED_DIR
    facts_file = p_dir / "facts.jsonl"

    if not facts_file.is_file():
        raise FileNotFoundError(f"Processed facts file not found at {facts_file}. Run collection first.")

    with facts_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ticker") == sym and rec.get("metric") == met and rec.get("fiscal_year") == fiscal_year:
                return rec

    raise KeyError(f"No validated fact found for {sym} {met} FY{fiscal_year}.")


def get_filing(
    ticker: str,
    fiscal_year: int,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """Retrieve source and local file metadata for an already collected filing from manifest.json.

    Raises:
        KeyError: If filing is not found.
    """
    sym = ticker.strip().upper()
    p_dir = processed_dir or DEFAULT_PROCESSED_DIR
    manifest_file = p_dir / "manifest.json"

    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest file not found at {manifest_file}. Run collection first.")

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    for filing in data.get("filings", []):
        if filing.get("ticker") == sym and filing.get("fiscal_year") == fiscal_year:
            return filing

    raise KeyError(f"No filing record found for {sym} FY{fiscal_year} in manifest.")


def main() -> None:
    """CLI entry point for sec_data module."""
    parser = argparse.ArgumentParser(description="Fyntrace SEC data collection and fact lookup.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect SEC data and normalize facts.")
    collect_parser.add_argument("--tickers", nargs="+", help="Specific tickers to collect.")
    collect_parser.add_argument("--years", nargs="+", type=int, help="Specific fiscal years to collect.")
    collect_parser.add_argument("--as-of-date", help="As-of cutoff date (YYYY-MM-DD).")

    fact_parser = subparsers.add_parser("fact", help="Retrieve one validated fact from local cache.")
    fact_parser.add_argument("--ticker", required=True, help="Company ticker symbol.")
    fact_parser.add_argument("--metric", required=True, help="Financial metric name.")
    fact_parser.add_argument("--year", required=True, type=int, help="Fiscal year.")

    args = parser.parse_args()

    if args.command == "collect":
        try:
            res = collect_data(tickers=args.tickers, years=args.years, as_of_date=args.as_of_date)
            summary = res["summary"]
            tot = summary["total_requested_filings"]
            succ = summary["successful_filings"]
            fail = summary["failed_filings"]
            facts_count = summary["facts_normalized"]
            print(
                f"Collection complete: {succ}/{tot} filings downloaded, {fail} failed, {facts_count} facts normalized."
            )
            if succ < tot:
                print("Notice: Collection was partial.")
                sys.exit(1)
        except Exception as exc:
            print(f"Collection failed: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "fact":
        try:
            fact = get_fact(ticker=args.ticker, metric=args.metric, fiscal_year=args.year)
            print(
                f"{fact['ticker']} FY{fact['fiscal_year']} {fact['metric']}: {fact['value']} {fact['unit']} "
                f"(Tag: {fact['tag']}, Form: {fact['form']}, Accn: {fact['accession']})"
            )
        except Exception as exc:
            print(f"Fact lookup failed: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
