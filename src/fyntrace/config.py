"""Configuration and company panel definition for Fyntrace."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Company:
    """Represents a public company in the Fyntrace coverage panel."""

    ticker: str
    name: str
    sector: str


# Supported fiscal years for initial filing coverage
SUPPORTED_FISCAL_YEARS: tuple[int, ...] = (2023, 2024, 2025)

# Sector-to-ticker mapping for exactly 15 companies across 5 sectors
SECTOR_COMPANIES: Mapping[str, tuple[str, ...]] = {
    "Technology": ("AAPL", "MSFT", "NVDA"),
    "Finance": ("JPM", "BAC", "GS"),
    "Healthcare": ("JNJ", "PFE", "ABBV"),
    "Energy": ("XOM", "CVX", "COP"),
    "Consumer": ("WMT", "COST", "MCD"),
}

# Detailed company panel definitions
COMPANY_PANEL: tuple[Company, ...] = (
    # Technology
    Company(ticker="AAPL", name="Apple Inc.", sector="Technology"),
    Company(ticker="MSFT", name="Microsoft Corporation", sector="Technology"),
    Company(ticker="NVDA", name="NVIDIA Corporation", sector="Technology"),
    # Finance
    Company(ticker="JPM", name="JPMorgan Chase & Co.", sector="Finance"),
    Company(ticker="BAC", name="Bank of America Corporation", sector="Finance"),
    Company(ticker="GS", name="The Goldman Sachs Group, Inc.", sector="Finance"),
    # Healthcare
    Company(ticker="JNJ", name="Johnson & Johnson", sector="Healthcare"),
    Company(ticker="PFE", name="Pfizer Inc.", sector="Healthcare"),
    Company(ticker="ABBV", name="AbbVie Inc.", sector="Healthcare"),
    # Energy
    Company(ticker="XOM", name="Exxon Mobil Corporation", sector="Energy"),
    Company(ticker="CVX", name="Chevron Corporation", sector="Energy"),
    Company(ticker="COP", name="ConocoPhillips", sector="Energy"),
    # Consumer
    Company(ticker="WMT", name="Walmart Inc.", sector="Consumer"),
    Company(ticker="COST", name="Costco Wholesale Corporation", sector="Consumer"),
    Company(ticker="MCD", name="McDonald's Corporation", sector="Consumer"),
)

ALL_TICKERS: tuple[str, ...] = tuple(c.ticker for c in COMPANY_PANEL)
ALL_SECTORS: tuple[str, ...] = tuple(SECTOR_COMPANIES.keys())
TICKER_TO_COMPANY: Mapping[str, Company] = {c.ticker: c for c in COMPANY_PANEL}

# Planned maximum number of Form 10-K filings across the panel
MAX_PLANNED_FILINGS: int = len(COMPANY_PANEL) * len(SUPPORTED_FISCAL_YEARS)


def is_supported_ticker(ticker: str) -> bool:
    """Check if ticker is in the supported company panel."""
    if not isinstance(ticker, str):
        return False
    return ticker.strip().upper() in TICKER_TO_COMPANY


def is_supported_year(year: int) -> bool:
    """Check if fiscal year is supported in the initial coverage."""
    return year in SUPPORTED_FISCAL_YEARS


def get_company(ticker: str) -> Company:
    """Retrieve company metadata by ticker symbol.

    Raises:
        KeyError: If ticker is not supported.
    """
    cleaned = ticker.strip().upper() if isinstance(ticker, str) else ""
    if cleaned not in TICKER_TO_COMPANY:
        raise KeyError(f"Unsupported ticker '{ticker}'. Supported: {', '.join(ALL_TICKERS)}")
    return TICKER_TO_COMPANY[cleaned]


def get_companies_by_sector(sector: str) -> tuple[Company, ...]:
    """Retrieve all companies belonging to a given sector.

    Raises:
        KeyError: If sector is not recognized.
    """
    if sector not in SECTOR_COMPANIES:
        raise KeyError(f"Unknown sector '{sector}'. Known sectors: {', '.join(ALL_SECTORS)}")
    return tuple(c for c in COMPANY_PANEL if c.sector == sector)
