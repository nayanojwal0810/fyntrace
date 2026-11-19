# Fyntrace

Verifiable financial analysis and source-grounded evidence retrieval over SEC filings.

## Status

Rebuild in progress on branch `rebuild/clean-start`.

## Overview

Fyntrace extracts, normalizes, and analyzes SEC Form 10-K filings with verifiable citations to official SEC archives. The system is designed to provide:

- Accurate financial fact extraction and normalized metric calculations
- Verifiable source citations grounded in original SEC Form 10-K filings
- High-precision evidence retrieval across multi-year annual reports
- Deterministic, auditable accounting comparisons and growth metrics

## Company Coverage

The initial panel covers 15 U.S. public companies across 5 sectors for fiscal years 2023–2025:

- **Technology**: AAPL, MSFT, NVDA
- **Finance**: JPM, BAC, GS
- **Healthcare**: JNJ, PFE, ABBV
- **Energy**: XOM, CVX, COP
- **Consumer**: WMT, COST, MCD

## Project Setup

```bash
# Clone and install in editable mode
pip install -e ".[dev]"

# Run tests
pytest
```
