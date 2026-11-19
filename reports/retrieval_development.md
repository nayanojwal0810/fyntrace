# Fyntrace — Retrieval Development Evaluation Report

## 1. Overview and Corpus Coverage

This report presents the baseline evaluation of the Fyntrace BM25 retrieval system evaluated against 100 independently verified development questions grounded in genuine SEC Form 10-K annual reports.

The evaluation measures whether the BM25 retrieval engine returns supporting evidence passages in the top 5 results for targeted financial research queries.

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
- **Retrieval implementation**: SQLite FTS5 using the built-in BM25 ranking function, external content table indexing, and fully parameterized SQL filtering for ticker, fiscal year, accession, section, and chunk type.

---

## 2. Evaluation Dataset Design

The development dataset (`evaluation/dev_questions.jsonl`) contains exactly 100 verified question objects frozen prior to evaluation.

### Distribution Requirements
- **Sector distribution**: Exactly 20 questions per sector across all 5 sectors.
- **Filing coverage**: Every one of the 45 company/fiscal year filings contributes at least 2 questions.
- **Category distribution**:
  - Business operations: 20 questions
  - Risk factors: 25 questions
  - Management discussion: 25 questions
  - Financial tables: 20 questions
  - Other supported disclosures: 10 questions
- **Difficulty distribution**:
  - Easy: 20 questions
  - Medium: 40 questions
  - Hard: 40 questions

### Gold Evidence Selection
All gold chunk IDs and exact evidence excerpts were established independently by inspecting the source filing canonical text and stored chunks prior to scoring. Each question has verified character offsets, source URLs, and accession numbers.

---

## 3. Overall Retrieval Performance

The primary evaluation measures Top-5 retrieval quality (`top_k = 5`).

| Metric | Result | Target Benchmark |
|---|---:|---:|
| **Recall@5** | **0.3700** (37.0%) | 0.9000 |
| **Hit@5** | **0.3700** (37.0%) | — |
| **MRR@5** | **0.2305** | 0.7500 |
| **Filter Accuracy** | **100.0%** (0 / 500 violations) | 100.0% |
| **Median Search Latency** | **144.40 ms** | — |
| **95th Percentile Latency (P95)** | **294.47 ms** | — |
| **Average Search Latency** | **149.73 ms** | — |
| **Total Missed Questions** | **63 / 100** | — |

---

## 4. Performance Breakdowns

### By Question Difficulty

| Difficulty | Questions | Recall@5 | Hit@5 | MRR@5 | Missed in Top 5 |
|---|---:|---:|---:|---:|---:|
| **Easy** | 20 | 0.4000 | 0.4000 | 0.2225 | 12 |
| **Medium** | 40 | 0.4250 | 0.4250 | 0.2933 | 23 |
| **Hard** | 40 | 0.3000 | 0.3000 | 0.1717 | 28 |

Medium questions achieved slightly higher recall than easy questions due to rich topic keywords matching narrative paragraphs, while hard questions had the lowest recall (30.0%) as they used nuanced financial terminology that diverged lexically from the filing text.

### By Sector

| Sector | Questions | Recall@5 | Hit@5 | MRR@5 | Missed in Top 5 |
|---|---:|---:|---:|---:|---:|
| **Technology** | 20 | 0.4000 | 0.4000 | 0.2475 | 12 |
| **Finance** | 20 | 0.3000 | 0.3000 | 0.1558 | 14 |
| **Healthcare** | 20 | 0.2000 | 0.2000 | 0.1417 | 16 |
| **Energy** | 20 | 0.4500 | 0.4500 | 0.3058 | 11 |
| **Consumer** | 20 | 0.5000 | 0.5000 | 0.3017 | 10 |

Consumer and Energy sectors demonstrated higher recall due to domain-specific product lines and segment terms (e.g., Permian, Walmart Connect, Kirkland Signature, Upstream), whereas Healthcare had lower recall due to dense medical terminology and diverse pipeline disclosures.

### By Question Category

| Category | Questions | Recall@5 | Hit@5 | MRR@5 | Missed in Top 5 |
|---|---:|---:|---:|---:|---:|
| **Business Operations** | 20 | 0.4500 | 0.4500 | 0.2917 | 11 |
| **Management Discussion** | 25 | 0.4000 | 0.4000 | 0.2393 | 15 |
| **Risk Factors** | 25 | 0.3600 | 0.3600 | 0.1880 | 16 |
| **Other Disclosures** | 10 | 0.6000 | 0.6000 | 0.5000 | 4 |
| **Financial Tables** | 20 | 0.1500 | 0.1500 | 0.0767 | 17 |

Financial tables experienced the lowest recall (15.0%) across all categories. Financial tables predominantly consist of numerical amounts and brief row headers, providing very sparse lexical overlap with natural language analyst queries.

### By Fiscal Year

| Fiscal Year | Questions | Recall@5 | Hit@5 | MRR@5 | Missed in Top 5 |
|---|---:|---:|---:|---:|---:|
| **FY2023** | 33 | 0.4242 | 0.4242 | 0.2621 | 19 |
| **FY2024** | 34 | 0.3235 | 0.3235 | 0.1784 | 23 |
| **FY2025** | 33 | 0.3636 | 0.3636 | 0.2525 | 21 |

---

## 5. Retrieval Failure Analysis

Analysis of the 63 missed questions revealed several primary failure modes inherent to pure lexical BM25 matching:

1. **Table-Heavy Content & Numeric Queries**:
   - Financial statement tables contain column labels and numbers without explanatory sentences. A query asking for segment revenue or operating earnings often matches narrative paragraphs containing the phrase rather than the actual tabular line item.
2. **Lexical Vocabulary Mismatch**:
   - Natural analyst questions use terms like "advisory revenues" or "unresolved comments" when the filing uses "Financial Advisory net revenues" or "None" under Item 1B.
3. **Pervasive Common Financial Terms**:
   - Terms like "net income", "operating costs", and "commercial operations" appear across dozens of chunks in a single filing. BM25 ranks chunks with repetitive term occurrences rather than the specific contextual definition sought.
4. **Section Misplacement & Exhibit Schedules**:
   - Certain financial disclosures are placed within Exhibit 15 or financial footnotes rather than core Item 7 or Item 8, competing against other chunks with similar terminology.

---

## 6. Analysis of Prior JPMorgan Search Examples

Two specific queries from earlier retrieval checks returned chunks labeled under `Item 15 - Exhibits and Financial Statement Schedules`:

### Query 1: JPM FY2024 — `net interest income investment banking`
- **Top Result Chunk ID**: `JPM_2024_10K_000001961725000270_00217`
- **Section**: `Item 15 - Exhibits and Financial Statement Schedules` (Type: table)
- **Top Result Text Excerpt**: `Investment banking fees | | | $ | 8,910 | | | | | $ | 6,519`
- **Assessment**:
  - The retrieved passage is a **legitimate financial table** containing the exact revenue figures for Investment Banking fees and Net interest income.
  - In JPMorgan Chase's 10-K, the Consolidated Financial Statements and Notes are located under Item 15 rather than duplicating them under Item 8. Thus, Item 15 is a **legitimate location containing relevant financial evidence**, combined with a **section-labeling property of the filing's table of contents**.

### Query 2: JPM FY2023 — `credit loss provisions deposits`
- **Top Result Chunk ID**: `JPM_2023_10K_000001961724000225_01376`
- **Section**: `Item 15 - Exhibits and Financial Statement Schedules` (Type: text)
- **Top Result Text Excerpt**: `In connection with the First Republic acquisition, the Firm and the FDIC entered into two shared-loss agreements with respect to certain loans...`
- **Assessment**:
  - While Chunk 5 (`...00351`) contains the actual provision for credit losses and deposits income statement data, the top result (`...01376`) matched words ("loss", "provisions") in a First Republic shared-loss note.
  - This outcome represents a **weak ranking result / lexical mismatch**, where BM25 scored an acquisition note higher than the primary financial statement schedule.

---

## 7. Current System Limitations

1. **Pure Lexical Search**: Lexical BM25 lacks semantic understanding and cannot bridge synonym differences between question phrasing and document wording.
2. **Table Retrieval Deficiencies**: Tabular rows without contextual narrative lack sufficient term density to outscore narrative discussions.
3. **No Cross-Chunk Synthesis**: Questions requiring multi-chunk or multi-table synthesis cannot be satisfied by single-chunk retrieval.

---

## 8. Reproduction Commands

To reproduce the validation and evaluation locally:

```powershell
# 1. Build the retrieval index
python -m fyntrace.retrieval build

# 2. Validate the 100-question development dataset
python -m fyntrace.retrieval_eval validate --questions evaluation/dev_questions.jsonl

# 3. Execute the full evaluation
python -m fyntrace.retrieval_eval run --questions evaluation/dev_questions.jsonl --top-k 5

# 4. Run test suites
python -m pytest -q
```
