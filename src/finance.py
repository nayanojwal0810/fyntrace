"""Financial calculations and company comparisons module."""

import json
import math
from pathlib import Path
from typing import Any

from src.sec_data import FINANCIAL_METRICS, PANEL_TICKERS, TARGET_FISCAL_YEARS

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FACTS_PATH = BASE_DIR / "data" / "processed" / "facts.jsonl"

SUPPORTED_TICKERS: set[str] = set(PANEL_TICKERS.keys())
SUPPORTED_YEARS: set[int] = set(TARGET_FISCAL_YEARS)
SUPPORTED_METRICS: set[str] = set(FINANCIAL_METRICS.keys())

ALLOWED_STATUSES = {"found", "missing", "ambiguous"}
REQUIRED_FIELDS = ("ticker", "fiscal_year", "metric", "status", "value", "unit")

RATIO_DEFINITIONS: dict[str, dict[str, str]] = {
    "operating_margin": {
        "display_name": "Operating margin",
        "numerator": "operating_income",
        "denominator": "revenue",
        "formula": "(operating_income / revenue) * 100",
    },
    "net_profit_margin": {
        "display_name": "Net profit margin",
        "numerator": "net_income",
        "denominator": "revenue",
        "formula": "(net_income / revenue) * 100",
    },
    "liabilities_to_assets": {
        "display_name": "Liabilities-to-assets",
        "numerator": "liabilities",
        "denominator": "assets",
        "formula": "(liabilities / assets) * 100",
    },
}


def load_facts(path: Path | None = None) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Load and validate financial fact records from a JSONL file.

    Indexes records by (ticker, fiscal_year, metric).
    Rejects duplicate keys, invalid values, and non-conforming records.
    """
    facts_path = Path(path) if path is not None else DEFAULT_FACTS_PATH
    if not facts_path.is_file():
        raise FileNotFoundError(f"Financial facts file not found at {facts_path}")

    facts: dict[tuple[str, int, str], dict[str, Any]] = {}
    try:
        with open(facts_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    record = json.loads(clean_line)
                except Exception as exc:
                    raise ValueError(f"Malformed JSON on line {line_no} in {facts_path}: {exc}") from exc

                if not isinstance(record, dict):
                    raise ValueError(f"Record on line {line_no} must be a JSON object.")

                for req_field in REQUIRED_FIELDS:
                    if req_field not in record:
                        raise ValueError(f"Missing required field '{req_field}' on line {line_no}.")

                ticker = record["ticker"]
                if not isinstance(ticker, str) or not ticker.strip():
                    raise ValueError(f"Invalid ticker on line {line_no}: {ticker!r}")
                ticker_clean = ticker.strip().upper()

                fy = record["fiscal_year"]
                if not isinstance(fy, int) or isinstance(fy, bool):
                    raise ValueError(f"Invalid fiscal_year on line {line_no}: {fy!r}")

                metric = record["metric"]
                if not isinstance(metric, str) or not metric.strip():
                    raise ValueError(f"Invalid metric on line {line_no}: {metric!r}")
                metric_clean = metric.strip().lower()

                key = (ticker_clean, fy, metric_clean)
                if key in facts:
                    raise ValueError(f"Duplicate record key {key} on line {line_no}.")

                status = record["status"]
                if status not in ALLOWED_STATUSES:
                    raise ValueError(f"Invalid status '{status}' on line {line_no}. Allowed: {ALLOWED_STATUSES}.")

                val = record.get("value")
                if status == "found":
                    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                        raise ValueError(f"Invalid financial value {val!r} for found record on line {line_no}.")
                    if record.get("unit") != "USD":
                        raise ValueError(f"Invalid unit '{record.get('unit')}' for found record on line {line_no}.")
                    for str_field in ("accession", "period_end", "source_url"):
                        field_val = record.get(str_field)
                        if not field_val or not isinstance(field_val, str) or not field_val.strip():
                            raise ValueError(f"Missing or invalid '{str_field}' for found record on line {line_no}.")
                else:
                    if val is not None:
                        raise ValueError(f"Value must be null for {status} record on line {line_no}, got {val!r}.")

                facts[key] = record
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Error reading file {facts_path}: {exc}") from exc

    return facts


def get_fact(
    ticker: str,
    fiscal_year: int,
    metric: str,
    facts: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Retrieve normalized financial fact record for a company, fiscal year, and metric."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"Invalid ticker: {ticker!r}")
    ticker_clean = ticker.strip().upper()
    if ticker_clean not in SUPPORTED_TICKERS:
        raise ValueError(f"Unsupported ticker: '{ticker}'. Must be one of supported panel companies.")

    if not isinstance(fiscal_year, int) or isinstance(fiscal_year, bool):
        raise ValueError(f"Invalid fiscal year: {fiscal_year!r}")
    if fiscal_year not in SUPPORTED_YEARS:
        raise ValueError(f"Unsupported fiscal year: {fiscal_year}. Supported years are {sorted(SUPPORTED_YEARS)}.")

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError(f"Invalid metric: {metric!r}")
    metric_clean = metric.strip().lower()
    if metric_clean not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: '{metric}'. Supported metrics are {sorted(SUPPORTED_METRICS)}.")

    if not isinstance(facts, dict):
        raise TypeError("facts must be a dictionary indexed by (ticker, fiscal_year, metric).")

    key = (ticker_clean, fiscal_year, metric_clean)
    if key not in facts:
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "fiscal_year": fiscal_year,
            "metric": metric_clean,
            "value": None,
            "reason": f"Financial fact for {ticker_clean} FY{fiscal_year} {metric_clean} is not in facts dataset.",
        }

    return dict(facts[key])


def calculate_growth(
    ticker: str,
    metric: str,
    start_year: int,
    end_year: int,
    facts: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Calculate percentage growth between start_year and end_year for a company metric."""
    if not isinstance(start_year, int) or isinstance(start_year, bool):
        raise ValueError(f"Invalid start_year: {start_year!r}")
    if not isinstance(end_year, int) or isinstance(end_year, bool):
        raise ValueError(f"Invalid end_year: {end_year!r}")
    if start_year >= end_year:
        raise ValueError(f"start_year ({start_year}) must be earlier than end_year ({end_year}).")

    start_fact = get_fact(ticker, start_year, metric, facts)
    end_fact = get_fact(ticker, end_year, metric, facts)

    ticker_clean = ticker.strip().upper() if isinstance(ticker, str) else ticker
    metric_clean = metric.strip().lower() if isinstance(metric, str) else metric

    if start_fact.get("status") != "found":
        reason = start_fact.get("reason") or f"FY{start_year} fact status is {start_fact.get('status')}"
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "metric": metric_clean,
            "start_year": start_year,
            "end_year": end_year,
            "reason": f"Starting year (FY{start_year}) data unavailable: {reason}",
        }

    if end_fact.get("status") != "found":
        reason = end_fact.get("reason") or f"FY{end_year} fact status is {end_fact.get('status')}"
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "metric": metric_clean,
            "start_year": start_year,
            "end_year": end_year,
            "reason": f"Ending year (FY{end_year}) data unavailable: {reason}",
        }

    unit_start = start_fact.get("unit")
    unit_end = end_fact.get("unit")
    if unit_start != unit_end:
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "metric": metric_clean,
            "start_year": start_year,
            "end_year": end_year,
            "reason": f"Incompatible units: FY{start_year} ({unit_start}) vs FY{end_year} ({unit_end}).",
        }

    old_val = start_fact["value"]
    new_val = end_fact["value"]

    if old_val <= 0:
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "metric": metric_clean,
            "start_year": start_year,
            "end_year": end_year,
            "reason": f"Starting value must be greater than zero for percentage change calculation (got {old_val}).",
        }

    formula = "((new_value - old_value) / old_value) * 100"
    raw_change = ((new_val - old_val) / old_val) * 100.0
    pct_change = round(raw_change, 4)

    return {
        "status": "success",
        "ticker": ticker_clean,
        "metric": metric_clean,
        "start_year": start_year,
        "end_year": end_year,
        "old_value": old_val,
        "new_value": new_val,
        "percentage_change": pct_change,
        "unit": unit_start,
        "formula": formula,
        "start_period_end": start_fact.get("period_end"),
        "end_period_end": end_fact.get("period_end"),
        "sources": [
            {
                "year": start_year,
                "accession": start_fact.get("accession"),
                "source_url": start_fact.get("source_url"),
            },
            {
                "year": end_year,
                "accession": end_fact.get("accession"),
                "source_url": end_fact.get("source_url"),
            },
        ],
    }


def calculate_ratio(
    ticker: str,
    fiscal_year: int,
    ratio_name: str,
    facts: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Calculate a standard financial ratio (operating margin, net profit margin, liabilities-to-assets)."""
    if not isinstance(ratio_name, str) or not ratio_name.strip():
        raise ValueError("ratio_name must be a non-empty string.")

    cleaned_name = ratio_name.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned_name not in RATIO_DEFINITIONS:
        raise ValueError(
            f"Unsupported ratio name: '{ratio_name}'. Supported ratios: {list(RATIO_DEFINITIONS.keys())}."
        )

    ratio_def = RATIO_DEFINITIONS[cleaned_name]
    num_metric = ratio_def["numerator"]
    den_metric = ratio_def["denominator"]

    num_fact = get_fact(ticker, fiscal_year, num_metric, facts)
    den_fact = get_fact(ticker, fiscal_year, den_metric, facts)

    ticker_clean = ticker.strip().upper() if isinstance(ticker, str) else ticker

    if num_fact.get("status") != "found":
        reason = num_fact.get("reason") or f"status is {num_fact.get('status')}"
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "fiscal_year": fiscal_year,
            "ratio": ratio_def["display_name"],
            "reason": f"Numerator '{num_metric}' data unavailable: {reason}",
        }

    if den_fact.get("status") != "found":
        reason = den_fact.get("reason") or f"status is {den_fact.get('status')}"
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "fiscal_year": fiscal_year,
            "ratio": ratio_def["display_name"],
            "reason": f"Denominator '{den_metric}' data unavailable: {reason}",
        }

    if num_fact.get("unit") != den_fact.get("unit"):
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "fiscal_year": fiscal_year,
            "ratio": ratio_def["display_name"],
            "reason": f"Incompatible units: '{num_metric}' ({num_fact.get('unit')}) vs '{den_metric}' ({den_fact.get('unit')}).",
        }

    num_val = num_fact["value"]
    den_val = den_fact["value"]

    if den_val <= 0:
        return {
            "status": "unavailable",
            "ticker": ticker_clean,
            "fiscal_year": fiscal_year,
            "ratio": ratio_def["display_name"],
            "reason": f"Denominator '{den_metric}' must be greater than zero (got {den_val}).",
        }

    raw_ratio = (num_val / den_val) * 100.0
    ratio_val = round(raw_ratio, 4)

    return {
        "status": "success",
        "ticker": ticker_clean,
        "fiscal_year": fiscal_year,
        "ratio": ratio_def["display_name"],
        "value": ratio_val,
        "unit": "percent",
        "formula": ratio_def["formula"],
        "numerator": {
            "metric": num_metric,
            "value": num_val,
            "unit": num_fact.get("unit"),
            "period_end": num_fact.get("period_end"),
            "tag": num_fact.get("tag"),
            "accession": num_fact.get("accession"),
            "source_url": num_fact.get("source_url"),
        },
        "denominator": {
            "metric": den_metric,
            "value": den_val,
            "unit": den_fact.get("unit"),
            "period_end": den_fact.get("period_end"),
            "tag": den_fact.get("tag"),
            "accession": den_fact.get("accession"),
            "source_url": den_fact.get("source_url"),
        },
        "sources": [
            {
                "metric": num_metric,
                "accession": num_fact.get("accession"),
                "source_url": num_fact.get("source_url"),
            },
            {
                "metric": den_metric,
                "accession": den_fact.get("accession"),
                "source_url": den_fact.get("source_url"),
            },
        ],
    }


def compare_companies(
    ticker_a: str,
    ticker_b: str,
    fiscal_year: int,
    metric: str,
    facts: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Compare a financial metric between two companies for a target fiscal year."""
    fact_a = get_fact(ticker_a, fiscal_year, metric, facts)
    fact_b = get_fact(ticker_b, fiscal_year, metric, facts)

    t_a = ticker_a.strip().upper() if isinstance(ticker_a, str) else ticker_a
    t_b = ticker_b.strip().upper() if isinstance(ticker_b, str) else ticker_b
    m_clean = metric.strip().lower() if isinstance(metric, str) else metric

    if fact_a.get("status") != "found":
        reason = fact_a.get("reason") or f"status is {fact_a.get('status')}"
        return {
            "status": "unavailable",
            "ticker_a": t_a,
            "ticker_b": t_b,
            "fiscal_year": fiscal_year,
            "metric": m_clean,
            "reason": f"Company {t_a} data unavailable for {m_clean} FY{fiscal_year}: {reason}",
        }

    if fact_b.get("status") != "found":
        reason = fact_b.get("reason") or f"status is {fact_b.get('status')}"
        return {
            "status": "unavailable",
            "ticker_a": t_a,
            "ticker_b": t_b,
            "fiscal_year": fiscal_year,
            "metric": m_clean,
            "reason": f"Company {t_b} data unavailable for {m_clean} FY{fiscal_year}: {reason}",
        }

    unit_a = fact_a.get("unit")
    unit_b = fact_b.get("unit")
    if unit_a != unit_b:
        return {
            "status": "unavailable",
            "ticker_a": t_a,
            "ticker_b": t_b,
            "fiscal_year": fiscal_year,
            "metric": m_clean,
            "reason": f"Incompatible units: {t_a} ({unit_a}) vs {t_b} ({unit_b}).",
        }

    val_a = fact_a["value"]
    val_b = fact_b["value"]
    diff = val_a - val_b

    notes: list[str] = []
    end_a = fact_a.get("period_end")
    end_b = fact_b.get("period_end")
    if end_a != end_b:
        notes.append(
            f"Reporting periods differ: {t_a} FY{fiscal_year} ended on {end_a}, while {t_b} FY{fiscal_year} ended on {end_b}. These are not identical calendar periods."
        )

    tag_a = fact_a.get("tag")
    tag_b = fact_b.get("tag")
    if m_clean == "revenue" and tag_a != tag_b:
        notes.append(
            f"Revenue accounting tags differ: {t_a} reported '{tag_a}', while {t_b} reported '{tag_b}'. Concepts may not be perfectly equivalent."
        )

    return {
        "status": "success",
        "fiscal_year": fiscal_year,
        "metric": m_clean,
        "unit": unit_a,
        "difference": diff,
        "company_a": {
            "ticker": t_a,
            "company_name": fact_a.get("company_name"),
            "value": val_a,
            "period_end": end_a,
            "tag": tag_a,
            "accession": fact_a.get("accession"),
            "source_url": fact_a.get("source_url"),
        },
        "company_b": {
            "ticker": t_b,
            "company_name": fact_b.get("company_name"),
            "value": val_b,
            "period_end": end_b,
            "tag": tag_b,
            "accession": fact_b.get("accession"),
            "source_url": fact_b.get("source_url"),
        },
        "notes": notes,
        "sources": [
            {
                "ticker": t_a,
                "accession": fact_a.get("accession"),
                "source_url": fact_a.get("source_url"),
            },
            {
                "ticker": t_b,
                "accession": fact_b.get("accession"),
                "source_url": fact_b.get("source_url"),
            },
        ],
    }
