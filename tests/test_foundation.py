"""Cross-module end-to-end integration tests for Fyntrace foundations.

Covers:
- Workflow A: SEC Manifest -> Normalized Fact -> Fact Lookup -> Sourced Result
- Workflow B: Normalized Facts -> Period/Unit Validation -> Decimal Growth -> Exact Sourced Output
- Workflow C: Multi-Company Facts -> Semantic Accounting Check -> Comparison / Non-comparable Result
- Workflow D: HTML Filing Document -> Canonical Text -> Chunks -> BM25 Index -> Filtered Retrieval
- Workflow E: Failure Modes and Abstention (No Invention of Missing Financial Facts or Filing Passages)
"""

from decimal import Decimal
import json
from pathlib import Path
import pytest

from src.sec_data import (
    filing_url,
    select_financial_fact,
)
from src.finance import (
    calculate_growth,
    calculate_net_assets,
    calculate_ratio,
    compare_companies,
    get_fact,
    load_facts,
)
from src.filings import (
    extract_sections,
    extract_visible_text,
    make_chunks,
)
from src.retrieval import (
    build_index,
    search_evidence,
)


SYNTHETIC_MANIFEST_RECORD_MSFT_2024 = {
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

SYNTHETIC_COMPANYFACTS_MSFT = {
    "cik": 789019,
    "entityName": "MICROSOFT CORP",
    "facts": {
        "us-gaap": {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "val": 512163000000,
                            "form": "10-K",
                            "fp": "FY",
                            "fy": 2024,
                            "accn": "0000950170-24-087843",
                            "end": "2024-06-30",
                            "filed": "2024-07-30",
                        }
                    ]
                }
            },
            "Liabilities": {
                "units": {
                    "USD": [
                        {
                            "val": 243686000000,
                            "form": "10-K",
                            "fp": "FY",
                            "fy": 2024,
                            "accn": "0000950170-24-087843",
                            "end": "2024-06-30",
                            "filed": "2024-07-30",
                        }
                    ]
                }
            },
        }
    },
}


def test_workflow_a_manifest_to_fact_lookup(tmp_path: Path) -> None:
    """Workflow A: Manifest -> normalized fact -> load_facts -> get_fact -> source-preserving result."""
    fact_rec = select_financial_fact(
        SYNTHETIC_MANIFEST_RECORD_MSFT_2024,
        "assets",
        SYNTHETIC_COMPANYFACTS_MSFT,
    )
    assert fact_rec["status"] == "found"
    assert fact_rec["value"] == 512163000000
    assert fact_rec["unit"] == "USD"
    assert fact_rec["accession"] == "0000950170-24-087843"
    assert fact_rec["source_url"] == SYNTHETIC_MANIFEST_RECORD_MSFT_2024["source_url"]

    # Write to facts.jsonl and reload through load_facts
    facts_file = tmp_path / "facts.jsonl"
    facts_file.write_text(json.dumps(fact_rec) + "\n", encoding="utf-8")

    loaded_index = load_facts(facts_file)
    result = get_fact("MSFT", 2024, "assets", loaded_index)
    assert result["status"] == "found"
    assert result["value"] == 512163000000
    assert result["accession"] == "0000950170-24-087843"
    assert result["source_url"] == SYNTHETIC_MANIFEST_RECORD_MSFT_2024["source_url"]


def test_workflow_b_growth_calculation_precision() -> None:
    """Workflow B: Two annual facts -> period/unit validation -> Decimal calculation -> exact sourced output."""
    facts = {
        ("MSFT", 2023, "revenue"): {
            "ticker": "MSFT",
            "company_name": "MICROSOFT CORP",
            "fiscal_year": 2023,
            "metric": "revenue",
            "status": "found",
            "value": 211915000000,
            "unit": "USD",
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "period_end": "2023-06-30",
            "accession": "0000950170-23-035122",
            "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
            "selection_policy": "original_filing",
        },
        ("MSFT", 2024, "revenue"): {
            "ticker": "MSFT",
            "company_name": "MICROSOFT CORP",
            "fiscal_year": 2024,
            "metric": "revenue",
            "status": "found",
            "value": 245122000000,
            "unit": "USD",
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "period_end": "2024-06-30",
            "accession": "0000950170-24-087843",
            "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
            "selection_policy": "original_filing",
        },
    }

    growth = calculate_growth("MSFT", "revenue", 2023, 2024, facts)
    assert growth["status"] == "success"
    # (245,122,000,000 - 211,915,000,000) / 211,915,000,000 * 100 = 15.66996... -> 15.6700%
    assert growth["percentage_change"] == "15.6700"
    assert growth["old_value"] == 211915000000
    assert growth["new_value"] == 245122000000
    assert len(growth["sources"]) == 2
    assert growth["sources"][0]["accession"] == "0000950170-23-035122"
    assert growth["sources"][1]["accession"] == "0000950170-24-087843"


def test_workflow_c_company_comparison_and_semantic_abstention() -> None:
    """Workflow C: Reported facts -> semantic comparability check -> comparison or not_comparable result."""
    facts = {
        ("AAPL", 2024, "assets"): {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "fiscal_year": 2024,
            "metric": "assets",
            "status": "found",
            "value": 364980000000,
            "unit": "USD",
            "period_end": "2024-09-28",
            "tag": "Assets",
            "accession": "0000320193-24-000106",
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000106/aapl-20240928.htm",
            "selection_policy": "original_filing",
        },
        ("MSFT", 2024, "assets"): {
            "ticker": "MSFT",
            "company_name": "MICROSOFT CORP",
            "fiscal_year": 2024,
            "metric": "assets",
            "status": "found",
            "value": 512163000000,
            "unit": "USD",
            "period_end": "2024-06-30",
            "tag": "Assets",
            "accession": "0000950170-24-087843",
            "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
            "selection_policy": "original_filing",
        },
        ("JPM", 2024, "revenue"): {
            "ticker": "JPM",
            "company_name": "JPMORGAN CHASE & CO",
            "fiscal_year": 2024,
            "metric": "revenue",
            "status": "found",
            "value": 158104000000,
            "unit": "USD",
            "period_end": "2024-12-31",
            "tag": "RevenuesNetOfInterestExpense",
            "accession": "0000019617-25-000270",
            "source_url": "https://www.sec.gov/Archives/edgar/data/19617/000001961725000270/jpm-20241231.htm",
            "selection_policy": "original_filing",
        },
        ("MSFT", 2024, "revenue"): {
            "ticker": "MSFT",
            "company_name": "MICROSOFT CORP",
            "fiscal_year": 2024,
            "metric": "revenue",
            "status": "found",
            "value": 245122000000,
            "unit": "USD",
            "period_end": "2024-06-30",
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "accession": "0000950170-24-087843",
            "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
            "selection_policy": "original_filing",
        },
    }

    # 1. Valid assets comparison
    comp_assets = compare_companies("AAPL", "MSFT", 2024, "assets", facts)
    assert comp_assets["status"] == "success"
    assert comp_assets["difference"] == 364980000000 - 512163000000
    assert any("Reporting periods differ" in n for n in comp_assets["notes"])

    # 2. Bank vs industrial revenue comparison -> not_comparable
    comp_rev = compare_companies("JPM", "MSFT", 2024, "revenue", facts)
    assert comp_rev["status"] == "not_comparable"
    assert "materially incompatible" in comp_rev["reason"]
    assert comp_rev["company_a"]["ticker"] == "JPM"
    assert comp_rev["company_b"]["ticker"] == "MSFT"


def test_workflow_d_filing_evidence_to_retrieval() -> None:
    """Workflow D: Synthetic fixture HTML -> canonical text -> source chunks -> BM25 index -> filtered retrieval."""
    synthetic_html = """
    <html>
      <head><title>Form 10-K MSFT</title></head>
      <body>
        <h1>PART I</h1>
        <h2>ITEM 1. BUSINESS</h2>
        <p>Microsoft enables digital transformation for the era of an intelligent cloud and an intelligent edge.</p>
        <p>Our mission is to empower every person and every organization on the planet to achieve more.</p>

        <h2>ITEM 1A. RISK FACTORS</h2>
        <p>Our operations face intense competition across global markets and emerging technologies.</p>
        <p>Adverse changes in cybersecurity risks, data protection regulations, or network infrastructure could harm operations.</p>
        <p>Security vulnerabilities could lead to service disruptions and loss of enterprise client confidence.</p>
      </body>
    </html>
    """
    canonical_text = extract_visible_text(synthetic_html)
    sections = extract_sections(canonical_text)

    chunks = make_chunks(
        canonical_text,
        SYNTHETIC_MANIFEST_RECORD_MSFT_2024,
        "d" * 64,
        sections=sections,
    )
    assert len(chunks) > 0

    for ch in chunks:
        # Offsets must match canonical text exactly
        assert canonical_text[ch["char_start"]:ch["char_end"]] == ch["text"]

    index = build_index(chunks)
    search_res = search_evidence(
        query="cybersecurity data protection security vulnerabilities",
        tickers=["MSFT"],
        fiscal_years=[2024],
        index=index,
        top_k=3,
    )
    assert search_res["status"] == "success"
    assert len(search_res["results"]) >= 1
    top_hit = search_res["results"][0]
    assert top_hit["ticker"] == "MSFT"
    assert top_hit["fiscal_year"] == 2024
    assert top_hit["accession"] == "0000950170-24-087843"
    assert "cybersecurity" in top_hit["text"].lower()


def test_workflow_e_failure_and_abstention_invariants() -> None:
    """Workflow E: Failure and abstention - missing facts, invalid inputs, or absent filings do not invent data."""
    facts = {
        ("MSFT", 2024, "assets"): {
            "ticker": "MSFT",
            "company_name": "MICROSOFT CORP",
            "fiscal_year": 2024,
            "metric": "assets",
            "status": "found",
            "value": 512163000000,
            "unit": "USD",
            "period_end": "2024-06-30",
            "accession": "0000950170-24-087843",
            "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
            "selection_policy": "original_filing",
        },
        ("MSFT", 2024, "liabilities"): {
            "ticker": "MSFT",
            "company_name": "MICROSOFT CORP",
            "fiscal_year": 2024,
            "metric": "liabilities",
            "status": "missing",
            "value": None,
            "unit": "USD",
            "period_end": "2024-06-30",
            "accession": "0000950170-24-087843",
            "source_url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
            "selection_policy": "original_filing",
        },
    }

    # 1. Missing fact lookup abstention
    res_fact = get_fact("MSFT", 2024, "liabilities", facts)
    assert res_fact["status"] == "missing"
    assert res_fact["value"] is None

    # 2. Net assets calculation abstains when liabilities are missing (no data synthesis)
    res_net_assets = calculate_net_assets("MSFT", 2024, facts)
    assert res_net_assets["status"] == "unavailable"
    assert "Liabilities data unavailable" in res_net_assets["reason"]

    # 3. Growth calculation abstains on non-positive base
    facts_zero_base = {
        ("MSFT", 2023, "revenue"): dict(facts[("MSFT", 2024, "assets")], fiscal_year=2023, metric="revenue", value=0),
        ("MSFT", 2024, "revenue"): dict(facts[("MSFT", 2024, "assets")], fiscal_year=2024, metric="revenue", value=100),
    }
    res_growth = calculate_growth("MSFT", "revenue", 2023, 2024, facts_zero_base)
    assert res_growth["status"] == "unavailable"
    assert "greater than zero" in res_growth["reason"]

    # 4. Filtered retrieval abstains when target company has no indexed filing
    empty_index = build_index([])
    res_retrieval = search_evidence("any query", ["MSFT"], [2024], empty_index)
    assert res_retrieval["status"] == "no_results"
    assert len(res_retrieval["results"]) == 0
