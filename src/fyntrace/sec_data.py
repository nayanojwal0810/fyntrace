"""SEC data collection, validation, and financial fact normalization."""

import argparse
from collections import defaultdict
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

DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"


class SecAccessError(Exception):
    """Raised when SEC access is rejected or persistent block errors occur."""


class AmbiguousFilingError(Exception):
    """Raised when multiple conflicting filing candidates exist for a fiscal year."""


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


def validate_filing_html(content: bytes) -> None:
    """Validate raw SEC filing HTML content for presence, size, markup, and error markers.

    Raises:
        ValueError: If content is empty, too small, or not HTML.
        SecAccessError: If content contains an SEC error, block, or rate-limit page.
    """
    if not content:
        raise ValueError("Filing content is empty.")
    if len(content) < 1000:
        raise ValueError(f"Filing content is suspiciously small ({len(content)} bytes).")

    # Check leading bytes for HTML markup
    head = content[:8192].lower()
    if not (b"<html" in head or b"<!doctype" in head or b"<xml" in head or b"<document" in head):
        raise ValueError("Content does not appear to be valid HTML or SEC filing markup.")

    # Check leading bytes for error/block page markers
    error_patterns = (
        b"access denied",
        b"your request has been blocked",
        b"automated tool",
        b"unauthorized request",
        b"rate limit exceeded",
        b"too many requests",
        b"403 forbidden",
        b"sec.gov - page not found",
    )
    for pattern in error_patterns:
        if pattern in head:
            marker_str = pattern.decode("latin1", "replace")
            raise SecAccessError(f"SEC block or error page detected in response: {marker_str}")


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
                raise ConnectionError(f"Connection failure after {max_retries} attempts: {exc}") from exc
            time.sleep(1.0 * attempt)
            continue

        if resp.status_code == 200:
            return resp
        elif resp.status_code == 403:
            resp.close()
            raise SecAccessError("SEC access denied (HTTP 403). Verify User-Agent contact information.")
        elif resp.status_code == 404:
            resp.close()
            raise FileNotFoundError(f"Requested SEC resource not found (HTTP 404): {url}")
        elif resp.status_code in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After")
            resp.close()
            if attempt == max_retries:
                if resp.status_code == 429:
                    raise SecAccessError(f"SEC rate limit exceeded (HTTP 429) after {max_retries} retries.")
                raise SecAccessError(f"SEC server returned HTTP {resp.status_code} after {max_retries} retries.")
            if retry_after:
                try:
                    sleep_sec = max(1.0, float(retry_after))
                except ValueError:
                    sleep_sec = 2.0 * attempt
            else:
                sleep_sec = 2.0 * attempt
            time.sleep(sleep_sec)
        else:
            code = resp.status_code
            resp.close()
            raise SecAccessError(f"Unexpected HTTP {code} response from SEC endpoint.")

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
            content = cache_file.read_bytes()
            data = json.loads(content)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    resp = sec_request(session, TICKER_MAPPING_URL)
    content = resp.content
    try:
        data = json.loads(content)
        if not isinstance(data, dict) or not data:
            raise ValueError("Invalid format or empty payload in company_tickers.json.")
    except Exception as exc:
        raise ValueError(f"Failed to parse company_tickers.json response: {exc}") from exc

    _atomic_write_bytes(cache_file, content)
    return data


def resolve_ticker(ticker: str, mapping: Mapping[str, Any]) -> dict[str, str]:
    """Resolve a ticker symbol to company metadata using official SEC mapping.

    Returns dict with ticker, cik (10-digit zero-padded), name, and sector.
    """
    if not is_supported_ticker(ticker):
        raise ValueError(f"Ticker '{ticker}' is not supported in the configured company panel.")

    company = get_company(ticker)
    sym = company.ticker

    if sym in HISTORICAL_REGISTRANTS:
        hist = HISTORICAL_REGISTRANTS[sym]
        return {
            "ticker": sym,
            "cik": hist["cik"],
            "name": hist["name"],
            "sector": company.sector,
        }

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
            content = cache_file.read_bytes()
            data = json.loads(content)
            if (
                isinstance(data, dict)
                and normalize_cik(data.get("cik", "")) == cik_10
                and "filings" in data
                and "recent" in data["filings"]
            ):
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik_10)
    resp = sec_request(session, url)
    content = resp.content
    try:
        data = json.loads(content)
        if (
            not isinstance(data, dict)
            or normalize_cik(data.get("cik", "")) != cik_10
            or "filings" not in data
            or "recent" not in data["filings"]
        ):
            raise ValueError(f"Incomplete submissions payload received for CIK {cik_10}.")
    except Exception as exc:
        raise ValueError(f"Failed to parse submissions response for CIK {cik_10}: {exc}") from exc

    _atomic_write_bytes(cache_file, content)
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
            content = cache_file.read_bytes()
            data = json.loads(content)
            if isinstance(data, dict) and "accessionNumber" in data:
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    url = HISTORICAL_SUBMISSIONS_URL_TEMPLATE.format(filename=safe_name)
    resp = sec_request(session, url)
    content = resp.content
    try:
        data = json.loads(content)
        if not isinstance(data, dict) or "accessionNumber" not in data:
            raise ValueError(f"Incomplete historical submissions payload in {filename}.")
    except Exception as exc:
        raise ValueError(f"Failed to parse historical submissions {filename}: {exc}") from exc

    _atomic_write_bytes(cache_file, content)
    return data


def get_companyfacts(
    cik: str | int,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load cached company facts or download from SEC.

    Stores the byte-exact SHA-256 in the returned dictionary under '_file_sha256'.
    """
    cik_10 = normalize_cik(cik)
    target_dir = raw_dir or DEFAULT_RAW_DIR
    cache_file = target_dir / "companyfacts" / f"CIK{cik_10}.json"

    if not force_refresh and cache_file.is_file():
        try:
            content = cache_file.read_bytes()
            data = json.loads(content)
            if (
                isinstance(data, dict)
                and normalize_cik(data.get("cik", "")) == cik_10
                and "facts" in data
                and "us-gaap" in data["facts"]
            ):
                data["_file_sha256"] = hashlib.sha256(content).hexdigest()
                return data
        except Exception:
            pass

    if session is None:
        session = make_sec_session()

    url = COMPANYFACTS_URL_TEMPLATE.format(cik=cik_10)
    resp = sec_request(session, url)
    content = resp.content
    try:
        data = json.loads(content)
        if (
            not isinstance(data, dict)
            or normalize_cik(data.get("cik", "")) != cik_10
            or "facts" not in data
            or "us-gaap" not in data["facts"]
        ):
            raise ValueError(f"Incomplete companyfacts payload received for CIK {cik_10}.")
    except Exception as exc:
        raise ValueError(f"Failed to parse companyfacts response for CIK {cik_10}: {exc}") from exc

    _atomic_write_bytes(cache_file, content)
    data["_file_sha256"] = hashlib.sha256(content).hexdigest()
    return data


def infer_fiscal_year_from_filing(report_date_str: str, filing_date_str: str, fye: str | None = None) -> int:
    """Determine reporting fiscal year from period-end date, filing date, and company FYE.

    Validates that reportDate aligns with the expected fiscal-year-end pattern.
    """
    rd = datetime.strptime(report_date_str, "%Y-%m-%d").date()
    fd = datetime.strptime(filing_date_str, "%Y-%m-%d").date()

    if fd < rd:
        raise ValueError(f"Filing date {filing_date_str} cannot precede report date {report_date_str}.")

    # Parse fiscal-year-end metadata
    fye_month = int(fye[:2]) if fye and len(fye) >= 2 else 12
    fye_day = int(fye[2:]) if fye and len(fye) >= 4 else 31

    # 1. Calendar year or 52/53-week ending first week of January (e.g. JNJ, fye="0103")
    if fye_month == 12 or (fye_month == 1 and fye_day <= 7):
        if rd.month == 1 and rd.day <= 7:
            # 52/53-week ending in first week of January belongs to prior calendar year
            return rd.year - 1
        elif rd.month in (11, 12):
            return rd.year
        else:
            raise ValueError(f"Report date {report_date_str} does not align with December fiscal year end ({fye}).")

    # 2. January fiscal year end (e.g. NVDA, WMT, fye="0131")
    elif fye_month == 1:
        if rd.month == 12 and rd.day >= 20:
            return rd.year + 1
        elif rd.month in (1, 2):
            return rd.year
        else:
            raise ValueError(f"Report date {report_date_str} does not align with January fiscal year end ({fye}).")

    # 3. Mid-year fiscal year ends (e.g. June for MSFT, August/September for COST/AAPL)
    else:
        allowed_months = {fye_month, (fye_month - 1) or 12, (fye_month % 12) + 1}
        if rd.month not in allowed_months:
            raise ValueError(f"Report date {report_date_str} does not align with fiscal year end ({fye}).")
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
    """Scan submissions aligned arrays for Form 10-K matching fiscal_year.

    Validates that all required arrays have identical lengths.
    """
    forms = records.get("form", [])
    reports = records.get("reportDate", [])
    filings = records.get("filingDate", [])
    accns = records.get("accessionNumber", [])
    docs = records.get("primaryDocument", [])

    n = len(forms)
    if not (len(reports) == n and len(filings) == n and len(accns) == n and len(docs) == n):
        raise ValueError(
            f"Mismatched aligned submission arrays: form={len(forms)}, reportDate={len(reports)}, "
            f"filingDate={len(filings)}, accessionNumber={len(accns)}, primaryDocument={len(docs)}"
        )

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
        AmbiguousFilingError: If multiple conflicting candidates exist.
        ValueError: If fiscal year is unsupported or arrays are corrupt.
    """
    if not is_supported_year(fiscal_year):
        raise ValueError(f"Fiscal year {fiscal_year} is not supported. Supported: {SUPPORTED_FISCAL_YEARS}")

    cik = normalize_cik(submissions.get("cik", ""))
    fye = submissions.get("fiscalYearEnd")

    as_of = date.today() if as_of_date is None else datetime.strptime(as_of_date, "%Y-%m-%d").date()

    recent = submissions.get("filings", {}).get("recent", {})
    candidates = _scan_aligned_arrays_for_10k(recent, fiscal_year, cik, as_of, fye)

    if not candidates:
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
        accns = {c["accession"] for c in candidates}
        if len(accns) > 1:
            raise AmbiguousFilingError(f"Ambiguous 10-K filings found for CIK {cik} FY{fiscal_year}: {accns}")

    return candidates[0]


def download_filing(
    filing: Mapping[str, Any],
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force_refresh: bool = False,
    expected_sha256: str | None = None,
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
        try:
            validate_filing_html(content)
            sha = hashlib.sha256(content).hexdigest()
            if expected_sha256 and sha != expected_sha256:
                raise ValueError(f"Cached filing SHA mismatch: expected {expected_sha256}, got {sha}")
            return local_path, sha
        except Exception:
            if session is None:
                raise

    if session is None:
        session = make_sec_session()

    url = filing["source_url"]
    validate_sec_url(url)
    resp = sec_request(session, url)
    content = resp.content

    validate_filing_html(content)

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

            # Instant vs duration validation
            if kind == "instant":
                # Instant metrics (assets, liabilities) must NOT contain a start date
                if u.get("start") is not None:
                    continue
            elif kind == "duration":
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

    # Handle multiple matching entries / candidate tags
    if metric == "revenue":
        by_tag: dict[str, Any] = {}
        for f in selected_facts:
            by_tag[f["tag"]] = f["val"]

        unique_vals = set(by_tag.values())
        if len(unique_vals) > 1:
            return None  # Conflicting candidate values -> ambiguous

        # Select deterministically according to candidate_tags priority
        chosen_fact = None
        for tag in candidate_tags:
            for f in selected_facts:
                if f["tag"] == tag:
                    chosen_fact = f
                    break
            if chosen_fact:
                break
    else:
        unique_vals = {f["val"] for f in selected_facts}
        if len(unique_vals) > 1:
            return None  # Conflicting values -> ambiguous
        chosen_fact = selected_facts[0]

    if not chosen_fact:
        return None

    raw_val = chosen_fact["val"]
    if isinstance(raw_val, (int, float)):
        if isinstance(raw_val, float) and raw_val.is_integer():
            final_val: int | str = int(raw_val)
        elif isinstance(raw_val, int):
            final_val = raw_val
        else:
            final_val = str(Decimal(str(raw_val)))
    else:
        final_val = str(raw_val)

    # Use actual cached file SHA-256 if available, or compute byte hash
    cf_sha256 = companyfacts.get("_file_sha256")
    if not cf_sha256:
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
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Execute complete SEC data collection, normalization, and manifest generation.

    Preserves unrelated existing valid records and prevents partial runs from deleting data.
    """
    target_tickers = list(dict.fromkeys(t.strip().upper() for t in tickers)) if tickers else list(ALL_TICKERS)
    target_years = sorted(set(years)) if years else list(SUPPORTED_FISCAL_YEARS)

    for t in target_tickers:
        if not is_supported_ticker(t):
            raise ValueError(f"Unsupported ticker '{t}'. Supported: {ALL_TICKERS}")
    for y in target_years:
        if not is_supported_year(y):
            raise ValueError(f"Unsupported year {y}. Supported: {SUPPORTED_FISCAL_YEARS}")

    requested_pairs = {(t, y) for t in target_tickers for y in target_years}
    as_of = date.today().isoformat() if as_of_date is None else as_of_date

    raw_dir = cache_dir or DEFAULT_RAW_DIR
    processed_dir = (raw_dir.parent / "processed") if cache_dir else DEFAULT_PROCESSED_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load existing manifest and facts to preserve unrelated coverage
    manifest_file = processed_dir / "manifest.json"
    facts_file = processed_dir / "facts.jsonl"

    existing_filings_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    existing_identities: dict[str, dict[str, str]] = {}
    if manifest_file.is_file():
        try:
            m_prev = json.loads(manifest_file.read_text(encoding="utf-8"))
            for f_rec in m_prev.get("filings", []):
                key = (f_rec["ticker"], f_rec["fiscal_year"])
                existing_filings_by_pair[key] = f_rec
            for id_rec in m_prev.get("company_identities", []):
                existing_identities[id_rec["ticker"]] = id_rec
        except Exception:
            pass

    existing_facts_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if facts_file.is_file():
        try:
            with facts_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        fact_rec = json.loads(line)
                        key = (fact_rec["ticker"], fact_rec["fiscal_year"])
                        existing_facts_by_pair[key].append(fact_rec)
        except Exception:
            pass

    if session is None:
        session = make_sec_session(user_agent)

    run_successful = 0
    run_failed = 0
    run_facts = 0
    stop_access_error: str | None = None

    # 2. Load ticker mapping
    try:
        ticker_mapping = load_company_tickers(session=session, raw_dir=raw_dir)
    except SecAccessError as exc:
        stop_access_error = str(exc)
        ticker_mapping = {}
        for t in target_tickers:
            for y in target_years:
                pair = (t, y)
                if pair not in existing_filings_by_pair or existing_filings_by_pair[pair].get("status") != "downloaded":
                    existing_filings_by_pair[pair] = {
                        "ticker": t,
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
                        "unavailable_metrics": {m: stop_access_error for m in SUPPORTED_METRICS},
                        "errors": [stop_access_error],
                    }
                run_failed += 1
    except Exception as exc:
        ticker_mapping = {}
        for t in target_tickers:
            for y in target_years:
                pair = (t, y)
                if pair not in existing_filings_by_pair or existing_filings_by_pair[pair].get("status") != "downloaded":
                    existing_filings_by_pair[pair] = {
                        "ticker": t,
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
                run_failed += 1

    # 3. Process requested companies
    for company in COMPANY_PANEL:
        if not ticker_mapping:
            break

        if company.ticker not in target_tickers:
            continue

        if stop_access_error:
            # SEC access denied: stop further network requests immediately
            for y in target_years:
                pair = (company.ticker, y)
                if pair not in existing_filings_by_pair or existing_filings_by_pair[pair].get("status") != "downloaded":
                    existing_filings_by_pair[pair] = {
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
                        "unavailable_metrics": {m: stop_access_error for m in SUPPORTED_METRICS},
                        "errors": [stop_access_error],
                    }
                run_failed += 1
            continue

        try:
            resolved = resolve_ticker(company.ticker, ticker_mapping)
            existing_identities[company.ticker] = resolved

            cik = resolved["cik"]
            name = resolved["name"]
            sector = resolved["sector"]

            sub = get_submissions(cik, session=session, raw_dir=raw_dir)
            cf = get_companyfacts(cik, session=session, raw_dir=raw_dir)
        except SecAccessError as exc:
            stop_access_error = str(exc)
            for y in target_years:
                pair = (company.ticker, y)
                if pair not in existing_filings_by_pair or existing_filings_by_pair[pair].get("status") != "downloaded":
                    existing_filings_by_pair[pair] = {
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
                run_failed += 1
            continue
        except Exception as exc:
            for y in target_years:
                pair = (company.ticker, y)
                if pair not in existing_filings_by_pair or existing_filings_by_pair[pair].get("status") != "downloaded":
                    existing_filings_by_pair[pair] = {
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
                run_failed += 1
            continue

        for y in target_years:
            pair = (company.ticker, y)
            try:
                filing_meta = select_annual_filing(sub, y, as_of_date=as_of, session=session, raw_dir=raw_dir)
            except AmbiguousFilingError as exc:
                existing_filings_by_pair[pair] = {
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
                    "unavailable_metrics": {m: "ambiguous filing candidates" for m in SUPPORTED_METRICS},
                    "errors": [str(exc)],
                }
                existing_facts_by_pair.pop(pair, None)
                run_failed += 1
                continue
            except SecAccessError as exc:
                stop_access_error = str(exc)
                existing_filings_by_pair[pair] = {
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
                run_failed += 1
                curr_idx = target_years.index(y)
                for rem_y in target_years[curr_idx + 1:]:
                    rpair = (company.ticker, rem_y)
                    if rpair not in existing_filings_by_pair or existing_filings_by_pair[rpair].get("status") != "downloaded":
                        existing_filings_by_pair[rpair] = {
                            "ticker": company.ticker,
                            "fiscal_year": rem_y,
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
                            "unavailable_metrics": {m: stop_access_error for m in SUPPORTED_METRICS},
                            "errors": [stop_access_error],
                        }
                    run_failed += 1
                break
            except Exception as exc:
                existing_filings_by_pair[pair] = {
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
                run_failed += 1
                continue

            if not filing_meta:
                existing_filings_by_pair[pair] = {
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
                existing_facts_by_pair.pop(pair, None)
                run_failed += 1
                continue

            filing_meta["ticker"] = company.ticker
            filing_meta["company_name"] = name
            filing_meta["sector"] = sector

            # Check expected sha from previous manifest if available
            prev_sha = existing_filings_by_pair.get(pair, {}).get("file_sha256")

            try:
                local_file, file_sha = download_filing(
                    filing_meta,
                    session=session,
                    raw_dir=raw_dir,
                    expected_sha256=prev_sha,
                )
            except SecAccessError as exc:
                stop_access_error = str(exc)
                existing_filings_by_pair[pair] = {
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
                    "unavailable_metrics": {m: str(exc) for m in SUPPORTED_METRICS},
                    "errors": [str(exc)],
                }
                run_failed += 1
                curr_idx = target_years.index(y)
                for rem_y in target_years[curr_idx + 1:]:
                    rpair = (company.ticker, rem_y)
                    if rpair not in existing_filings_by_pair or existing_filings_by_pair[rpair].get("status") != "downloaded":
                        existing_filings_by_pair[rpair] = {
                            "ticker": company.ticker,
                            "fiscal_year": rem_y,
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
                            "unavailable_metrics": {m: stop_access_error for m in SUPPORTED_METRICS},
                            "errors": [stop_access_error],
                        }
                    run_failed += 1
                break
            except Exception as exc:
                # If fresh download failed but we already had a valid downloaded file, preserve it
                if existing_filings_by_pair.get(pair, {}).get("status") == "downloaded":
                    run_failed += 1
                    continue
                existing_filings_by_pair[pair] = {
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
                run_failed += 1
                continue

            selected_metrics = []
            unavailable_metrics = {}
            pair_facts: list[dict[str, Any]] = []

            for m in SUPPORTED_METRICS:
                fact = normalize_annual_fact(cf, filing_meta, m)
                if fact:
                    selected_metrics.append(m)
                    pair_facts.append(fact)
                else:
                    unavailable_metrics[m] = "concept unavailable or ambiguous in SEC Company Facts"

            display_path = str(local_file.as_posix())

            existing_filings_by_pair[pair] = {
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
            existing_facts_by_pair[pair] = pair_facts
            run_successful += 1
            run_facts += len(pair_facts)

    # 4. Assemble stable, sorted manifest filings and facts
    ticker_order = {c.ticker: i for i, c in enumerate(COMPANY_PANEL)}

    all_filings: list[dict[str, Any]] = sorted(
        existing_filings_by_pair.values(),
        key=lambda f: (ticker_order.get(f["ticker"], 999), f["fiscal_year"]),
    )

    all_identities: list[dict[str, str]] = sorted(
        existing_identities.values(),
        key=lambda id_rec: ticker_order.get(id_rec["ticker"], 999),
    )

    all_facts: list[dict[str, Any]] = []
    for f_rec in all_filings:
        pair = (f_rec["ticker"], f_rec["fiscal_year"])
        all_facts.extend(existing_facts_by_pair.get(pair, []))

    # Write facts.jsonl atomically
    facts_lines = [json.dumps(fact) + "\n" for fact in all_facts]
    _atomic_write_bytes(facts_file, "".join(facts_lines).encode("utf-8"))

    # Write manifest.json atomically
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
        "company_identities": all_identities,
        "summary": {
            "total_requested_filings": len(requested_pairs),
            "successful_filings": run_successful,
            "failed_filings": run_failed,
            "facts_normalized": run_facts,
            "current_run": {
                "requested_filings": len(requested_pairs),
                "successful_filings": run_successful,
                "failed_filings": run_failed,
                "facts_normalized": run_facts,
            },
            "total_stored_filings": len(all_filings),
            "total_downloaded_filings": sum(1 for f in all_filings if f.get("status") == "downloaded"),
            "total_stored_facts": len(all_facts),
        },
        "filings": all_filings,
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
            current_run = res["summary"]["current_run"]
            tot = current_run["requested_filings"]
            succ = current_run["successful_filings"]
            fail = current_run["failed_filings"]
            facts_count = current_run["facts_normalized"]
            print(
                f"Collection complete: {succ}/{tot} filings downloaded, {fail} failed, {facts_count} facts normalized."
            )
            if succ < tot or fail > 0:
                print("Notice: Collection was partial or encountered failures.")
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
