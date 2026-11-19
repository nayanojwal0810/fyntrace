"""Automated tests for financial calculations and company comparisons."""

import json
import math
from pathlib import Path

import pytest

from src.finance import (
    calculate_growth,
    calculate_net_assets,
    calculate_ratio,
    compare_companies,
    get_fact,
    load_facts,
)


def make_fact_row(
    ticker: str = "MSFT",
    company_name: str = "MICROSOFT CORP",
    fy: int = 2024,
    metric: str = "revenue",
    status: str = "found",
    value: float | int | None = 245122000000,
    tag: str | None = "RevenueFromContractWithCustomerExcludingAssessedTax",
    end: str = "2024-06-30",
    unit: str = "USD",
    accn: str = "0000950170-24-087843",
    reason: str | None = None,
) -> dict:
    """Helper to construct a single valid normalized fact dictionary."""
    return {
        "ticker": ticker,
        "company_name": company_name,
        "cik": "0000789019",
        "sector": "Technology",
        "fiscal_year": fy,
        "metric": metric,
        "status": status,
        "value": value,
        "unit": unit,
        "taxonomy": "us-gaap",
        "tag": tag,
        "period_start": "2023-07-01" if metric in ("revenue", "net_income", "operating_income") else None,
        "period_end": end,
        "filed_date": "2024-07-30",
        "form": "10-K",
        "accession": accn,
        "source_url": f"https://www.sec.gov/{accn}",
        "selection_policy": "original_filing",
        "reason": reason,
    }


def test_1_load_facts_valid_jsonl(tmp_path: Path) -> None:
    """Test 1: Loading valid JSONL file."""
    rows = [
        make_fact_row(metric="revenue", value=200000),
        make_fact_row(metric="operating_income", status="missing", value=None, reason="Tag absent"),
        make_fact_row(metric="liabilities", status="ambiguous", value=None, reason="Multiple tags"),
    ]
    test_file = tmp_path / "facts.jsonl"
    test_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    facts = load_facts(test_file)
    assert len(facts) == 3
    assert ("MSFT", 2024, "revenue") in facts
    assert facts[("MSFT", 2024, "revenue")]["status"] == "found"
    assert facts[("MSFT", 2024, "revenue")]["value"] == 200000
    assert facts[("MSFT", 2024, "operating_income")]["status"] == "missing"
    assert facts[("MSFT", 2024, "operating_income")]["value"] is None
    assert facts[("MSFT", 2024, "liabilities")]["status"] == "ambiguous"
    assert facts[("MSFT", 2024, "liabilities")]["value"] is None


def test_2_duplicate_record_detection(tmp_path: Path) -> None:
    """Test 2: Duplicate record key raises ValueError."""
    rows = [
        make_fact_row(metric="revenue", value=200000),
        make_fact_row(metric="revenue", value=210000),
    ]
    test_file = tmp_path / "dup.jsonl"
    test_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_facts(test_file)
    assert "Duplicate" in str(exc.value)


@pytest.mark.parametrize(
    "bad_value",
    [
        "1000000",
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_3_invalid_financial_values(tmp_path: Path, bad_value: object) -> None:
    """Test 3: Reject non-finite, string, or boolean financial values for found records."""
    row = make_fact_row(value=bad_value)  # type: ignore[arg-type]
    test_file = tmp_path / "bad_val.jsonl"
    # Dump directly; NaN/Inf need default json handling
    if isinstance(bad_value, float) and (math.isnan(bad_value) or math.isinf(bad_value)):
        raw = json.dumps(row, allow_nan=True)
    else:
        raw = json.dumps(row)
    test_file.write_text(raw + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_facts(test_file)
    assert "Invalid financial value" in str(exc.value)


def test_3b_missing_or_ambiguous_with_non_null_value(tmp_path: Path) -> None:
    """Test 3b: Missing or ambiguous records must have null values."""
    row = make_fact_row(status="missing", value=100)
    test_file = tmp_path / "bad_missing.jsonl"
    test_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_facts(test_file)
    assert "Value must be null" in str(exc.value)


def test_4_valid_fact_lookup() -> None:
    """Test 4: Valid fact lookup preserves original fields."""
    row = make_fact_row(ticker="MSFT", fy=2024, metric="assets", value=512163000000)
    facts = {("MSFT", 2024, "assets"): row}

    res = get_fact("MSFT", 2024, "assets", facts)
    assert res["status"] == "found"
    assert res["value"] == 512163000000
    assert res["unit"] == "USD"
    assert res["accession"] == "0000950170-24-087843"
    assert res["period_end"] == "2024-06-30"

    # Case-insensitive checks
    res_lower = get_fact("msft", 2024, "ASSETS", facts)
    assert res_lower["value"] == 512163000000


def test_5_missing_and_ambiguous_fact_handling() -> None:
    """Test 5: Missing and ambiguous facts are preserved and never converted to zero."""
    row_missing = make_fact_row(ticker="SO", fy=2023, metric="net_income", status="missing", value=None, reason="No tag")
    row_ambiguous = make_fact_row(ticker="WMT", fy=2024, metric="revenue", status="ambiguous", value=None, reason="Conflicting tags")
    facts = {
        ("SO", 2023, "net_income"): row_missing,
        ("WMT", 2024, "revenue"): row_ambiguous,
    }

    res_m = get_fact("SO", 2023, "net_income", facts)
    assert res_m["status"] == "missing"
    assert res_m["value"] is None
    assert res_m["value"] != 0

    res_a = get_fact("WMT", 2024, "revenue", facts)
    assert res_a["status"] == "ambiguous"
    assert res_a["value"] is None
    assert res_a["value"] != 0

    # Fact not present in facts dictionary returns unavailable
    res_unavail = get_fact("AAPL", 2024, "revenue", facts)
    assert res_unavail["status"] == "unavailable"
    assert res_unavail["value"] is None


def test_6_revenue_growth_calculation() -> None:
    """Test 6: Revenue growth percentage calculation."""
    r23 = make_fact_row(ticker="MSFT", fy=2023, metric="revenue", value=200000, accn="accn-2023", end="2023-06-30")
    r24 = make_fact_row(ticker="MSFT", fy=2024, metric="revenue", value=250000, accn="accn-2024", end="2024-06-30")
    facts = {
        ("MSFT", 2023, "revenue"): r23,
        ("MSFT", 2024, "revenue"): r24,
    }

    res = calculate_growth("MSFT", "revenue", 2023, 2024, facts)
    assert res["status"] == "success"
    assert res["old_value"] == 200000
    assert res["new_value"] == 250000
    assert res["percentage_change"] == "25.0000"
    assert res["formula"] == "((new_value - old_value) / old_value) * 100"
    assert len(res["sources"]) == 2
    assert res["sources"][0]["year"] == 2023
    assert res["sources"][1]["year"] == 2024


def test_7_zero_and_negative_starting_values() -> None:
    """Test 7: Reject zero and negative starting values with an unavailable result."""
    r_zero = make_fact_row(ticker="MSFT", fy=2023, metric="net_income", value=0)
    r_neg = make_fact_row(ticker="MSFT", fy=2023, metric="net_income", value=-5000)
    r_pos = make_fact_row(ticker="MSFT", fy=2024, metric="net_income", value=10000)

    facts_zero = {("MSFT", 2023, "net_income"): r_zero, ("MSFT", 2024, "net_income"): r_pos}
    res_zero = calculate_growth("MSFT", "net_income", 2023, 2024, facts_zero)
    assert res_zero["status"] == "unavailable"
    assert "greater than zero" in res_zero["reason"]

    facts_neg = {("MSFT", 2023, "net_income"): r_neg, ("MSFT", 2024, "net_income"): r_pos}
    res_neg = calculate_growth("MSFT", "net_income", 2023, 2024, facts_neg)
    assert res_neg["status"] == "unavailable"
    assert "greater than zero" in res_neg["reason"]

    # Start year >= end year raises ValueError
    with pytest.raises(ValueError):
        calculate_growth("MSFT", "net_income", 2024, 2023, facts_neg)


def test_8_operating_and_net_profit_margins() -> None:
    """Test 8: Operating and net profit margins."""
    facts = {
        ("MSFT", 2024, "operating_income"): make_fact_row(metric="operating_income", value=50000),
        ("MSFT", 2024, "net_income"): make_fact_row(metric="net_income", value=40000),
        ("MSFT", 2024, "revenue"): make_fact_row(metric="revenue", value=200000),
    }

    op_res = calculate_ratio("MSFT", 2024, "operating_margin", facts)
    assert op_res["status"] == "success"
    assert op_res["value"] == "25.0000"
    assert op_res["ratio"] == "Operating margin"

    net_res = calculate_ratio("MSFT", 2024, "net_profit_margin", facts)
    assert net_res["status"] == "success"
    assert net_res["value"] == "20.0000"
    assert net_res["ratio"] == "Net profit margin"

    # Negative net income (net loss) allowed in margins
    facts_loss = dict(facts)
    facts_loss[("MSFT", 2024, "net_income")] = make_fact_row(metric="net_income", value=-20000)
    loss_res = calculate_ratio("MSFT", 2024, "net_profit_margin", facts_loss)
    assert loss_res["status"] == "success"
    assert loss_res["value"] == "-10.0000"


def test_9_liabilities_to_assets_calculation() -> None:
    """Test 9: Liabilities-to-assets ratio calculation."""
    facts = {
        ("MSFT", 2024, "liabilities"): make_fact_row(metric="liabilities", value=60000),
        ("MSFT", 2024, "assets"): make_fact_row(metric="assets", value=100000),
    }

    res = calculate_ratio("MSFT", 2024, "liabilities-to-assets", facts)
    assert res["status"] == "success"
    assert res["value"] == "60.0000"
    assert res["ratio"] == "Liabilities-to-assets"
    assert res["numerator"]["value"] == 60000
    assert res["denominator"]["value"] == 100000


def test_10_missing_inputs_in_ratio_calculations() -> None:
    """Test 10: Missing or ambiguous inputs return unavailable."""
    facts = {
        ("MSFT", 2024, "operating_income"): make_fact_row(metric="operating_income", status="missing", value=None),
        ("MSFT", 2024, "revenue"): make_fact_row(metric="revenue", value=200000),
        ("MSFT", 2024, "assets"): make_fact_row(metric="assets", value=0),  # non-positive denominator
        ("MSFT", 2024, "liabilities"): make_fact_row(metric="liabilities", value=50000),
    }

    # Missing numerator
    res_missing = calculate_ratio("MSFT", 2024, "operating_margin", facts)
    assert res_missing["status"] == "unavailable"
    assert "operating_income" in res_missing["reason"]

    # Non-positive denominator
    res_zero_den = calculate_ratio("MSFT", 2024, "liabilities_to_assets", facts)
    assert res_zero_den["status"] == "unavailable"
    assert "greater than zero" in res_zero_den["reason"]


def test_11_valid_company_comparison() -> None:
    """Test 11: Valid company comparison calculates difference."""
    facts = {
        ("AAPL", 2024, "assets"): make_fact_row(ticker="AAPL", company_name="Apple Inc.", end="2024-09-30", value=300000),
        ("MSFT", 2024, "assets"): make_fact_row(ticker="MSFT", company_name="MICROSOFT CORP", end="2024-09-30", value=200000),
    }

    res = compare_companies("AAPL", "MSFT", 2024, "assets", facts)
    assert res["status"] == "success"
    assert res["difference"] == 100000
    assert res["company_a"]["ticker"] == "AAPL"
    assert res["company_b"]["ticker"] == "MSFT"
    assert len(res["notes"]) == 0


def test_12_different_fiscal_year_end_dates() -> None:
    """Test 12: Note added when fiscal-year end dates differ."""
    facts = {
        ("AAPL", 2024, "assets"): make_fact_row(ticker="AAPL", company_name="Apple Inc.", end="2024-09-28", value=364980000000),
        ("MSFT", 2024, "assets"): make_fact_row(ticker="MSFT", company_name="MICROSOFT CORP", end="2024-06-30", value=512163000000),
    }

    res = compare_companies("AAPL", "MSFT", 2024, "assets", facts)
    assert res["status"] == "success"
    assert len(res["notes"]) >= 1
    assert any("Reporting periods differ" in n for n in res["notes"])
    assert any("not identical calendar periods" in n for n in res["notes"])


def test_13_different_revenue_accounting_tags() -> None:
    """Test 13: Note added when revenue accounting tags differ."""
    facts = {
        ("AAPL", 2024, "revenue"): make_fact_row(
            ticker="AAPL",
            company_name="Apple Inc.",
            tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            end="2024-09-30",
            value=390000000,
        ),
        ("CVX", 2024, "revenue"): make_fact_row(
            ticker="CVX",
            company_name="CHEVRON CORP",
            tag="Revenues",
            end="2024-09-30",
            value=200000000,
        ),
    }

    res = compare_companies("AAPL", "CVX", 2024, "revenue", facts)
    assert res["status"] == "success"
    assert len(res["notes"]) >= 1
    assert any("Revenue accounting tags differ" in n for n in res["notes"])


def test_14_unsupported_companies_years_and_metrics() -> None:
    """Test 14: Unsupported companies, years, and metrics raise ValueError."""
    facts: dict = {}

    with pytest.raises(ValueError) as exc1:
        get_fact("UNKNOWN_TICKER", 2024, "revenue", facts)
    assert "Unsupported ticker" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        get_fact("MSFT", 2018, "revenue", facts)
    assert "Unsupported fiscal year" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        get_fact("MSFT", 2024, "ebitda", facts)
    assert "Unsupported metric" in str(exc3.value)

    with pytest.raises(ValueError) as exc4:
        calculate_ratio("MSFT", 2024, "return_on_equity", facts)
    assert "Unsupported ratio" in str(exc4.value)


def test_15_decimal_precision_with_large_usd_numbers() -> None:
    """1. Decimal precision with large USD numbers preserves exact integer precision without float drift."""
    # 512,163,000,000 to 563,379,300,000 (growth: exactly 10.0000%)
    f23 = make_fact_row(ticker="MSFT", fy=2023, metric="assets", value=512163000000)
    f24 = make_fact_row(ticker="MSFT", fy=2024, metric="assets", value=563379300000)
    facts = {("MSFT", 2023, "assets"): f23, ("MSFT", 2024, "assets"): f24}

    res = calculate_growth("MSFT", "assets", 2023, 2024, facts)
    assert res["status"] == "success"
    assert res["percentage_change"] == "10.0000"


def test_16_growth_100_to_125_and_100_to_80() -> None:
    """2 & 3. 100 to 125 growth equals 25.0000%; 100 to 80 growth equals -20.0000%."""
    f100 = make_fact_row(ticker="MSFT", fy=2023, metric="revenue", value=100)
    f125 = make_fact_row(ticker="MSFT", fy=2024, metric="revenue", value=125)
    f80 = make_fact_row(ticker="MSFT", fy=2024, metric="revenue", value=80)

    res_pos = calculate_growth("MSFT", "revenue", 2023, 2024, {("MSFT", 2023, "revenue"): f100, ("MSFT", 2024, "revenue"): f125})
    assert res_pos["status"] == "success"
    assert res_pos["percentage_change"] == "25.0000"

    res_neg = calculate_growth("MSFT", "revenue", 2023, 2024, {("MSFT", 2023, "revenue"): f100, ("MSFT", 2024, "revenue"): f80})
    assert res_neg["status"] == "success"
    assert res_neg["percentage_change"] == "-20.0000"


def test_17_nonterminating_decimal_ratio_and_rounding() -> None:
    """4. Nonterminating decimal ratio rounds using ROUND_HALF_UP to 4 decimal places."""
    # 1 / 3 = 33.333333... -> 33.3333%
    facts = {
        ("MSFT", 2024, "net_income"): make_fact_row(metric="net_income", value=1),
        ("MSFT", 2024, "revenue"): make_fact_row(metric="revenue", value=3),
    }
    res = calculate_ratio("MSFT", 2024, "net_profit_margin", facts)
    assert res["status"] == "success"
    assert res["value"] == "33.3333"

    # 2 / 3 = 66.666666... -> 66.6667%
    facts_2 = {
        ("MSFT", 2024, "net_income"): make_fact_row(metric="net_income", value=2),
        ("MSFT", 2024, "revenue"): make_fact_row(metric="revenue", value=3),
    }
    res_2 = calculate_ratio("MSFT", 2024, "net_profit_margin", facts_2)
    assert res_2["status"] == "success"
    assert res_2["value"] == "66.6667"


def test_18_mismatched_reporting_periods_in_ratios() -> None:
    """9. Mismatched annual reporting periods within the same year return unavailable."""
    facts = {
        ("MSFT", 2024, "net_income"): make_fact_row(metric="net_income", value=40000, end="2024-06-30"),
        ("MSFT", 2024, "revenue"): make_fact_row(metric="revenue", value=200000, end="2024-07-31"),
    }
    res = calculate_ratio("MSFT", 2024, "net_profit_margin", facts)
    assert res["status"] == "unavailable"
    assert "Reporting period mismatch" in res["reason"]


def test_19_incompatible_revenue_concepts_in_growth() -> None:
    """10. Revenue concept tag changes between growth years return unavailable."""
    f23 = make_fact_row(ticker="WMT", fy=2023, metric="revenue", value=600000, tag="Revenues")
    f24 = make_fact_row(ticker="WMT", fy=2024, metric="revenue", value=650000, tag="SalesRevenueNet")
    facts = {("WMT", 2023, "revenue"): f23, ("WMT", 2024, "revenue"): f24}

    res = calculate_growth("WMT", "revenue", 2023, 2024, facts)
    assert res["status"] == "unavailable"
    assert "concept changed" in res["reason"]


def test_20_bank_versus_industrial_revenue_comparison() -> None:
    """11. Bank net interest revenue compared to commercial sales returns not_comparable."""
    facts = {
        ("JPM", 2024, "revenue"): make_fact_row(
            ticker="JPM",
            company_name="JPMORGAN CHASE & CO",
            tag="RevenuesNetOfInterestExpense",
            value=160000000000,
        ),
        ("MSFT", 2024, "revenue"): make_fact_row(
            ticker="MSFT",
            company_name="MICROSOFT CORP",
            tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            value=245000000000,
        ),
    }
    res = compare_companies("JPM", "MSFT", 2024, "revenue", facts)
    assert res["status"] == "not_comparable"
    assert "materially incompatible" in res["reason"]
    assert res["company_a"]["ticker"] == "JPM"
    assert res["company_b"]["ticker"] == "MSFT"
    assert len(res["sources"]) == 2


def test_21_valid_net_assets_calculation() -> None:
    """13. Valid net assets = assets - liabilities with integer precision and complete sourcing."""
    facts = {
        ("MSFT", 2024, "assets"): make_fact_row(ticker="MSFT", metric="assets", value=512163000000),
        ("MSFT", 2024, "liabilities"): make_fact_row(ticker="MSFT", metric="liabilities", value=243686000000),
    }
    res = calculate_net_assets("MSFT", 2024, facts)
    assert res["status"] == "success"
    assert res["value"] == 268477000000
    assert res["formula"] == "assets - liabilities"
    assert res["inputs"]["assets"] == 512163000000
    assert res["inputs"]["liabilities"] == 243686000000
    assert len(res["sources"]) == 2


def test_22_missing_liabilities_net_assets_abstention() -> None:
    """14. Missing liabilities returns unavailable without synthesizing data."""
    facts = {
        ("AMZN", 2024, "assets"): make_fact_row(ticker="AMZN", metric="assets", value=500000000000),
        ("AMZN", 2024, "liabilities"): make_fact_row(ticker="AMZN", metric="liabilities", status="missing", value=None),
    }
    res = calculate_net_assets("AMZN", 2024, facts)
    assert res["status"] == "unavailable"
    assert "Liabilities data unavailable" in res["reason"]


def test_23_json_serialization_of_all_public_results() -> None:
    """16. All public result dictionaries are strictly JSON serializable."""
    facts = {
        ("MSFT", 2023, "revenue"): make_fact_row(ticker="MSFT", fy=2023, metric="revenue", value=200000),
        ("MSFT", 2024, "revenue"): make_fact_row(ticker="MSFT", fy=2024, metric="revenue", value=250000),
        ("MSFT", 2024, "net_income"): make_fact_row(ticker="MSFT", fy=2024, metric="net_income", value=50000),
        ("MSFT", 2024, "assets"): make_fact_row(ticker="MSFT", fy=2024, metric="assets", value=500000),
        ("MSFT", 2024, "liabilities"): make_fact_row(ticker="MSFT", fy=2024, metric="liabilities", value=200000),
        ("AAPL", 2024, "assets"): make_fact_row(ticker="AAPL", fy=2024, metric="assets", value=350000),
    }

    g_res = calculate_growth("MSFT", "revenue", 2023, 2024, facts)
    r_res = calculate_ratio("MSFT", 2024, "net_profit_margin", facts)
    c_res = compare_companies("MSFT", "AAPL", 2024, "assets", facts)
    na_res = calculate_net_assets("MSFT", 2024, facts)

    for item in (g_res, r_res, c_res, na_res):
        serialized = json.dumps(item)
        assert isinstance(serialized, str)


def test_24_jpm_revenues_vs_msft_revenues_not_comparable() -> None:
    """JPM tagged Revenues vs MSFT tagged Revenues must be not_comparable based on sector identity."""
    facts = {
        ("JPM", 2024, "revenue"): make_fact_row(
            ticker="JPM",
            company_name="JPMORGAN CHASE & CO",
            tag="Revenues",
            value=177556000000,
        ),
        ("MSFT", 2024, "revenue"): make_fact_row(
            ticker="MSFT",
            company_name="MICROSOFT CORP",
            tag="Revenues",
            value=245122000000,
        ),
    }
    res = compare_companies("JPM", "MSFT", 2024, "revenue", facts)
    assert res["status"] == "not_comparable"
    assert "materially incompatible" in res["reason"]
    assert "financial services" in res["reason"]
    assert res["company_a"]["ticker"] == "JPM"
    assert res["company_b"]["ticker"] == "MSFT"
    assert res["company_a"]["tag"] == "Revenues"
    assert res["company_b"]["tag"] == "Revenues"
    assert len(res["sources"]) == 2


def test_25_gs_vs_wmt_revenue_not_comparable() -> None:
    """GS tagged RevenuesNetOfInterestExpense vs WMT tagged Revenues must be not_comparable."""
    facts = {
        ("GS", 2024, "revenue"): make_fact_row(
            ticker="GS",
            company_name="GOLDMAN SACHS GROUP INC",
            tag="RevenuesNetOfInterestExpense",
            value=53500000000,
        ),
        ("WMT", 2024, "revenue"): make_fact_row(
            ticker="WMT",
            company_name="WALMART INC",
            tag="Revenues",
            value=648125000000,
        ),
    }
    res = compare_companies("GS", "WMT", 2024, "revenue", facts)
    assert res["status"] == "not_comparable"
    assert "materially incompatible" in res["reason"]


def test_26_jpm_vs_msft_assets_comparison_succeeds() -> None:
    """JPM vs MSFT assets comparison succeeds normally across sectors."""
    facts = {
        ("JPM", 2024, "assets"): make_fact_row(
            ticker="JPM",
            company_name="JPMORGAN CHASE & CO",
            tag="Assets",
            value=4169974000000,
        ),
        ("MSFT", 2024, "assets"): make_fact_row(
            ticker="MSFT",
            company_name="MICROSOFT CORP",
            tag="Assets",
            value=512163000000,
        ),
    }
    res = compare_companies("JPM", "MSFT", 2024, "assets", facts)
    assert res["status"] == "success"
    assert res["difference"] == 4169974000000 - 512163000000
    assert res["company_a"]["ticker"] == "JPM"
    assert res["company_b"]["ticker"] == "MSFT"


def test_27_two_banks_revenue_comparison_logic() -> None:
    """Bank-vs-bank revenue comparison inspects accounting concepts and notes tag differences."""
    facts = {
        ("JPM", 2024, "revenue"): make_fact_row(
            ticker="JPM",
            company_name="JPMORGAN CHASE & CO",
            tag="Revenues",
            value=177556000000,
        ),
        ("GS", 2024, "revenue"): make_fact_row(
            ticker="GS",
            company_name="GOLDMAN SACHS GROUP INC",
            tag="RevenuesNetOfInterestExpense",
            value=53500000000,
        ),
    }
    res = compare_companies("JPM", "GS", 2024, "revenue", facts)
    assert res["status"] == "success"
    assert res["difference"] == 177556000000 - 53500000000
    assert any("financial institutions reporting bank net revenue" in n for n in res["notes"])


def test_28_unsupported_revenue_tag_returns_not_comparable() -> None:
    """Companies using unrecognized or non-revenue tags are not declared comparable."""
    facts = {
        ("MSFT", 2024, "revenue"): make_fact_row(ticker="MSFT", tag="RevenueFromContractWithCustomerExcludingAssessedTax", value=200000),
        ("AAPL", 2024, "revenue"): make_fact_row(ticker="AAPL", tag="UnrecognizedCustomTag", value=250000),
    }
    res = compare_companies("MSFT", "AAPL", 2024, "revenue", facts)
    assert res["status"] == "not_comparable"
    assert "cannot be established as comparable" in res["reason"]


def test_29_fractional_monetary_rejection_in_load_facts(tmp_path: Path) -> None:
    """load_facts rejects fractional monetary values rather than silently truncating."""
    row = make_fact_row(value=100.75)
    f = tmp_path / "fractional.jsonl"
    f.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_facts(f)
    assert "fractional" in str(exc.value).lower()


def test_30_fractional_monetary_abstention_in_calculations() -> None:
    """Direct calculation calls abstain when given fractional monetary values without silent truncation."""
    facts = {
        ("MSFT", 2024, "assets"): make_fact_row(metric="assets", value=500.5),
        ("MSFT", 2024, "liabilities"): make_fact_row(metric="liabilities", value=200),
    }
    res_na = calculate_net_assets("MSFT", 2024, facts)
    assert res_na["status"] == "unavailable"
    assert "fractional" in res_na["reason"].lower()

    facts_comp = {
        ("MSFT", 2024, "assets"): make_fact_row(ticker="MSFT", metric="assets", value=500.75),
        ("AAPL", 2024, "assets"): make_fact_row(ticker="AAPL", metric="assets", value=300),
    }
    res_comp = compare_companies("MSFT", "AAPL", 2024, "assets", facts_comp)
    assert res_comp["status"] == "unavailable"
    assert "fractional" in res_comp["reason"].lower()


def test_31_negative_net_assets_preserved() -> None:
    """Negative net assets (where liabilities exceed assets) is preserved as an exact negative integer."""
    facts = {
        ("MSFT", 2024, "assets"): make_fact_row(metric="assets", value=100000000),
        ("MSFT", 2024, "liabilities"): make_fact_row(metric="liabilities", value=150000000),
    }
    res = calculate_net_assets("MSFT", 2024, facts)
    assert res["status"] == "success"
    assert res["value"] == -50000000
    assert isinstance(res["value"], int)


def test_32_msft_fy2024_net_assets_exact_value() -> None:
    """Verify MSFT FY2024 net assets calculation using a synthetic regression fixture (assets - liabilities)."""
    accn = "0000950170-24-087843"
    facts = {
        ("MSFT", 2024, "assets"): make_fact_row(
            ticker="MSFT",
            company_name="MICROSOFT CORP",
            fy=2024,
            metric="assets",
            status="found",
            value=512163000000,
            tag="Assets",
            end="2024-06-30",
            unit="USD",
            accn=accn,
        ),
        ("MSFT", 2024, "liabilities"): make_fact_row(
            ticker="MSFT",
            company_name="MICROSOFT CORP",
            fy=2024,
            metric="liabilities",
            status="found",
            value=243686000000,
            tag="Liabilities",
            end="2024-06-30",
            unit="USD",
            accn=accn,
        ),
    }
    res = calculate_net_assets("MSFT", 2024, facts)
    assert res["status"] == "success"
    assert res["value"] == 268477000000
    assert isinstance(res["value"], int)
    assert res["unit"] == "USD"
    assert res["formula"] == "assets - liabilities"
    assert res["period_end"] == "2024-06-30"
    assert res["inputs"]["assets"] == 512163000000
    assert res["inputs"]["liabilities"] == 243686000000
    assert len(res["sources"]) == 2
    assert res["sources"][0]["metric"] == "assets"
    assert res["sources"][0]["accession"] == accn
    assert res["sources"][1]["metric"] == "liabilities"
    assert res["sources"][1]["accession"] == accn

    # Verify JSON serializability
    serialized = json.dumps(res)
    assert json.loads(serialized) == res


