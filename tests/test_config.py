"""Tests for Fyntrace configuration, company panel, and supported coverage."""

import re
import pytest

from fyntrace.config import (
    ALL_SECTORS,
    ALL_TICKERS,
    COMPANY_PANEL,
    MAX_PLANNED_FILINGS,
    SECTOR_COMPANIES,
    SUPPORTED_FISCAL_YEARS,
    TICKER_TO_COMPANY,
    Company,
    get_companies_by_sector,
    get_company,
    is_supported_ticker,
    is_supported_year,
)


def test_total_companies_count():
    """Verify the panel contains exactly 15 companies."""
    assert len(COMPANY_PANEL) == 15
    assert len(ALL_TICKERS) == 15
    assert len(TICKER_TO_COMPANY) == 15


def test_company_uniqueness():
    """Verify all 15 company tickers and names are unique."""
    tickers = [c.ticker for c in COMPANY_PANEL]
    names = [c.name for c in COMPANY_PANEL]

    assert len(set(tickers)) == 15, "Tickers must be unique"
    assert len(set(names)) == 15, "Company names must be unique"


def test_sector_counts():
    """Verify exactly 5 sectors, each containing exactly 3 companies."""
    assert len(SECTOR_COMPANIES) == 5
    assert len(ALL_SECTORS) == 5

    expected_sectors = {"Technology", "Finance", "Healthcare", "Energy", "Consumer"}
    assert set(ALL_SECTORS) == expected_sectors
    assert set(SECTOR_COMPANIES.keys()) == expected_sectors

    for sector, tickers in SECTOR_COMPANIES.items():
        assert len(tickers) == 3, f"Sector '{sector}' must contain exactly 3 companies"


def test_expected_tickers_per_sector():
    """Verify each sector matches the specified 15 U.S. public companies."""
    expected = {
        "Technology": ("AAPL", "MSFT", "NVDA"),
        "Finance": ("JPM", "BAC", "GS"),
        "Healthcare": ("JNJ", "PFE", "ABBV"),
        "Energy": ("XOM", "CVX", "COP"),
        "Consumer": ("WMT", "COST", "MCD"),
    }
    assert dict(SECTOR_COMPANIES) == expected

    for sector, expected_tickers in expected.items():
        sector_companies = get_companies_by_sector(sector)
        assert tuple(c.ticker for c in sector_companies) == expected_tickers
        for c in sector_companies:
            assert c.sector == sector


def test_ticker_validity():
    """Verify every ticker conforms to standard U.S. exchange ticker syntax."""
    ticker_pattern = re.compile(r"^[A-Z]{1,5}$")
    for c in COMPANY_PANEL:
        assert ticker_pattern.match(c.ticker), f"Invalid ticker syntax: {c.ticker}"
        assert c.ticker.isupper(), f"Ticker must be uppercase: {c.ticker}"
        assert c.name.strip(), f"Company name must be non-empty for {c.ticker}"
        assert c.sector.strip(), f"Sector must be non-empty for {c.ticker}"


def test_supported_fiscal_years():
    """Verify initial filing coverage spans FY2023 through FY2025."""
    assert SUPPORTED_FISCAL_YEARS == (2023, 2024, 2025)
    assert len(SUPPORTED_FISCAL_YEARS) == 3


def test_max_planned_filings():
    """Verify planned maximum Form 10-K filings is 45 (15 companies * 3 years)."""
    assert MAX_PLANNED_FILINGS == 45
    assert MAX_PLANNED_FILINGS == len(COMPANY_PANEL) * len(SUPPORTED_FISCAL_YEARS)


def test_is_supported_ticker():
    """Verify ticker support checking with valid and invalid inputs."""
    # Positive cases (including case insensitivity and whitespace stripping)
    assert is_supported_ticker("AAPL") is True
    assert is_supported_ticker("msft") is True
    assert is_supported_ticker(" nvda ") is True
    assert is_supported_ticker("JPM") is True
    assert is_supported_ticker("wmt") is True

    # Negative cases
    assert is_supported_ticker("UNKNOWN") is False
    assert is_supported_ticker("GOOGL") is False
    assert is_supported_ticker("") is False
    assert is_supported_ticker("   ") is False
    assert is_supported_ticker(None) is False
    assert is_supported_ticker(123) is False


def test_is_supported_year():
    """Verify fiscal year support checking."""
    assert is_supported_year(2023) is True
    assert is_supported_year(2024) is True
    assert is_supported_year(2025) is True

    assert is_supported_year(2020) is False
    assert is_supported_year(2022) is False
    assert is_supported_year(2026) is False
    assert is_supported_year("2024") is False
    assert is_supported_year(None) is False


def test_get_company():
    """Verify company lookup by ticker."""
    apple = get_company("AAPL")
    assert apple.ticker == "AAPL"
    assert apple.name == "Apple Inc."
    assert apple.sector == "Technology"

    # Case-insensitive lookup
    jpm = get_company("  jpm  ")
    assert jpm.ticker == "JPM"
    assert jpm.sector == "Finance"

    # Unsupported ticker raises KeyError
    with pytest.raises(KeyError, match="Unsupported ticker 'DIS'"):
        get_company("DIS")


def test_get_companies_by_sector():
    """Verify retrieval of companies by sector."""
    energy_companies = get_companies_by_sector("Energy")
    assert len(energy_companies) == 3
    assert [c.ticker for c in energy_companies] == ["XOM", "CVX", "COP"]

    with pytest.raises(KeyError, match="Unknown sector 'Industrial'"):
        get_companies_by_sector("Industrial")


def test_company_dataclass_frozen():
    """Verify Company instances are immutable."""
    co = get_company("NVDA")
    with pytest.raises(Exception):
        co.ticker = "NEW"
