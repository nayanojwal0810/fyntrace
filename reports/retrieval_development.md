# Fyntrace — Retrieval Development Evaluation Report

## 1. Overview and Corpus Coverage

This report presents the verified evaluation of the Fyntrace BM25 retrieval system against the audited 100-question development benchmark (`evaluation/dev_questions.jsonl`).

The evaluation measures whether the BM25 retrieval engine returns supporting evidence passages in the top 5 results for targeted financial research queries grounded in genuine SEC Form 10-K filings.

### Corpus Summary
- **Companies covered**: 15 companies across 5 sectors:
  - Technology: AAPL, MSFT, NVDA
  - Finance: JPM, BAC, GS
  - Healthcare: JNJ, PFE, ABBV
  - Energy: XOM, CVX, COP
  - Consumer: WMT, COST, MCD
- **Fiscal years covered**: FY2023, FY2024, FY2025 (45 total annual filings).
- **Total indexed chunks**: 26,766 chunks (13,830 narrative text chunks, 12,936 table chunks).
- **Corpus SHA-256**: `2c8c27ab823450f4efaa1729cf58792091ad41641dda7adf78a1e48abeb9eddf`.
- **Retrieval implementation**: SQLite FTS5 with built-in BM25 ranking, external content table indexing, and SQL filtering on ticker and fiscal year.

---

## 2. Dataset Quality Audit

All 100 development questions were individually audited against canonical Form 10-K source filings and indexed chunks.

### Summary Audit Counts
- **Questions audited**: 100
- **Questions accepted unchanged**: 81
- **Gold labels corrected**: 18 (reassigned from non-informative or misattributed chunks to exact substantive evidence)
- **Questions replaced**: 1 (DEV0060: replaced uninformative Item 1B 5-character fragment with Item 1C Cybersecurity governance disclosure for ABBV FY2025)
- **Evidence excerpts corrected**: 77 (eliminated mid-word cutoffs, truncated headings, and incomplete phrases to ensure grammatically complete clauses or table rows)
- **Difficulty distribution verified**: 20 Easy, 40 Medium, 40 Hard
- **Unresolved questions**: 0
- **Audited dataset SHA-256**: `01d0721cb5bbbe5dab372c2427cc18882a38f358a9656a31505963ac56705b51`

---

## 3. Verified Retrieval Performance

Evaluation was conducted using `top_k = 5` with exact metadata filtering on company ticker and fiscal year.

| Metric | Original Baseline | Audited Benchmark | Delta |
|---|---:|---:|---:|
| **Recall@5** | 0.3700 (37.0%) | **0.4500** (45.0%) | +0.0800 |
| **Hit@5** | 0.3700 (37.0%) | **0.4500** (45.0%) | +0.0800 |
| **MRR@5** | 0.2305 | **0.2888** | +0.0583 |
| **Filter Accuracy** | 100.0% (0 / 500) | **100.0%** (0 / 500) | 0.0% |
| **Median Search Latency** | 144.40 ms | **66.74 ms** | -77.66 ms |
| **95th Percentile Latency (P95)** | 294.47 ms | **118.16 ms** | -176.31 ms |
| **Average Search Latency** | 149.73 ms | **75.53 ms** | -74.20 ms |
| **Total Missed Questions** | 63 / 100 | **55 / 100** | -8 |

> **Note on Metric Changes**: The increase from 37.0% to 45.0% Recall@5 reflects the correction of 18 misattributed gold labels and 1 unsupported question in the evaluation dataset. The SQLite FTS5 BM25 retrieval code, ranking formula, and tokenization remained completely unchanged. Search latencies depend on the local execution environment.

---

## 4. Performance Breakdowns

### By Question Difficulty

| Difficulty | Questions | Recall@5 | Hit@5 | MRR@5 | Missed in Top 5 |
|---|---:|---:|---:|---:|---:|
| **Easy** | 20 | 0.5500 | 0.5500 | 0.3325 | 9 |
| **Medium** | 40 | 0.4500 | 0.4500 | 0.2967 | 22 |
| **Hard** | 40 | 0.4000 | 0.4000 | 0.2592 | 24 |

Easy questions with direct keyword overlap achieved 55.0% recall, while hard questions requiring conceptual synthesis or complex phrasing yielded 40.0% recall.

### By Sector

| Sector | Questions | Recall@5 | Hit@5 | MRR@5 | Missed in Top 5 |
|---|---:|---:|---:|---:|---:|
| **Consumer** | 20 | 0.5500 | 0.5500 | 0.3517 | 9 |
| **Energy** | 20 | 0.4500 | 0.4500 | 0.3058 | 11 |
| **Technology** | 20 | 0.4500 | 0.4500 | 0.3100 | 11 |
| **Finance** | 20 | 0.4000 | 0.4000 | 0.2558 | 12 |
| **Healthcare** | 20 | 0.4000 | 0.4000 | 0.2208 | 12 |

Consumer and Energy sectors demonstrated stronger recall due to distinctive operational terminology (e.g., store formats, Permian assets, brand names). Healthcare and Finance exhibited lower recall due to diverse clinical and regulatory terminology.

### By Question Category

| Category | Questions | Recall@5 | Hit@5 | MRR@5 | Missed in Top 5 |
|---|---:|---:|---:|---:|---:|
| **Other Disclosures** | 10 | 0.7000 | 0.7000 | 0.4750 | 3 |
| **Management Discussion** | 25 | 0.6400 | 0.6400 | 0.4713 | 9 |
| **Business Operations** | 20 | 0.4500 | 0.4500 | 0.2892 | 11 |
| **Risk Factors** | 25 | 0.4000 | 0.4000 | 0.2013 | 15 |
| **Financial Tables** | 20 | 0.1500 | 0.1500 | 0.0767 | 17 |

Management Discussion and Other Disclosures performed well under BM25 lexical matching. Financial Tables showed the lowest recall (15.0%), reflecting an inherent limitation of keyword-based retrieval on sparse numerical data.

---

## 5. Important Findings and Failure Analysis

1. **Financial-Table Retrieval Weakness**: Tabular chunks achieved only a 15.0% hit rate (17 misses out of 20). Pure BM25 struggles with numerical tables because financial statement chunks contain concise row headers and dense numbers without the natural language phrasing present in analyst queries.
2. **Lexical Mismatch in Risk Disclosures**: Risk factor disclosures often share generic legal boilerplate across filings. Keyword queries match multiple loosely related risks (e.g., market risk, liquidity, credit risks) that score higher than the specific target passage.
3. **Alternative Valid Evidence**: Qualitative review revealed that in several missed questions, retrieved passages contained substantive alternative discussion of the query topic. However, the official benchmark strictly measures exact gold chunk IDs.
4. **Evaluation Limitations**: The benchmark evaluates single-chunk passage retrieval. Cross-filing synthesis, multi-hop reasoning, and complex table aggregation are not measured by this lexical retrieval stage.

---

## 6. Reproduction Commands

To reproduce dataset validation and evaluation on a clean workspace with the processed corpus:

```powershell
# 1. Validate dataset independently (public reproduction)
python -m fyntrace.retrieval_eval validate --questions evaluation/dev_questions.jsonl

# 2. Validate dataset with local audit records
python -m fyntrace.retrieval_eval validate --questions evaluation/dev_questions.jsonl --audit-path data/processed/retrieval_dev_audit.jsonl

# 3. Execute BM25 retrieval evaluation
python -m fyntrace.retrieval_eval run --questions evaluation/dev_questions.jsonl --top-k 5

# 4. Run test suite
python -m pytest -q
```
