# Claim Verification System

An AI-powered insurance claim verification system that analyzes customer claim conversations and submitted images to determine whether a damage claim is supported, contradicted, or requires further review.

The system supports:

- Car damage claims
- Laptop damage claims
- Package damage claims

It uses Gemini 2.5 Flash for claim understanding and visual damage assessment while combining rule-based decision logic, evidence validation, and historical risk analysis.

---

## Features

- Multi-image claim analysis
- Automated claim extraction from customer conversations
- Visual damage detection using Gemini Vision
- Evidence requirement validation
- Historical user risk flag analysis
- Rule-based and VLM-based decision strategies
- Automatic retry and rate-limit handling
- Incremental autosave during batch processing
- Evaluation framework with operational metrics
- Structured output matching the required submission schema

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Run on the Full Dataset

```bash
cd code
python main.py
```

The pipeline reads `dataset/claims.csv`, processes each claim, and generates:

```text
code/output.csv
```

The output follows the exact required schema and is automatically saved periodically to avoid losing progress during long runs or API rate limits.

### Useful Commands

```bash
python main.py --strategy vlm_judgment
```

Uses the VLM-based judgment strategy instead of the default rule-based strategy.

```bash
python main.py --limit 20
```

Runs only the first 20 rows for testing.

```bash
python main.py --input ../dataset/sample_claims.csv --output sample_predictions.csv
```

Runs the pipeline on a custom input file.

---

## Evaluation

```bash
cd code
python evaluation/evaluate.py
```

The evaluation framework compares both decision strategies:

- rule_based
- vlm_judgment

It generates:

```text
evaluation/evaluation_report.md
```

The report includes:

- Accuracy metrics (when ground-truth labels are available)
- Model call statistics
- Token usage
- Runtime
- Retry counts
- Estimated operational cost
- TPM/RPM observations

If the sample dataset does not contain expected labels, the evaluation still runs and reports operational metrics.

---

## System Architecture

```text
Claims CSV
     │
     ▼
Claim Parser
     │
     ▼
Vision Analyzer
     │
     ▼
Evidence Checker
     │
     ▼
Risk Analyzer
     │
     ▼
Decision Engine
     │
     ▼
Output CSV
```

---

## Project Structure

### config.py

Single source of truth for:

- Output schema
- Allowed enum values
- Dataset paths
- Model configuration

### model_client.py

Provides:

- Gemini API integration
- Exponential backoff retries
- JSON extraction
- Usage tracking
- Cost estimation

### claim_parser.py

Extracts:

- claimed_issue
- claimed_part
- summary

from customer conversations.

### vision_analyzer.py

- Processes every submitted image
- Detects visible damage
- Determines severity
- Selects the strongest supporting image
- Supports multi-image claims

### evidence_checker.py

Validates whether submitted evidence satisfies the minimum claim requirements.

### risk_analyzer.py

Looks up historical user risk indicators from the user history dataset.

### decision_engine.py

Provides two strategies:

#### rule_based (default)

- Fast
- Deterministic
- Explainable
- Low cost

#### vlm_judgment

- Additional reasoning step using Gemini
- Better for ambiguous cases
- Higher cost and latency

### main.py

Coordinates the complete claim verification workflow with:

- Error isolation
- Autosave
- Batch processing

### evaluation/evaluate.py

Compares decision strategies and generates evaluation reports.

---

## Example Output

| user_id  | claim_object | claim_status           | issue_type |
| -------- | ------------ | ---------------------- | ---------- |
| user_005 | car          | supported              | dent       |
| user_004 | car          | contradicted           | none       |
| user_008 | car          | not_enough_information | unknown    |

---

## Design Decisions

- No hardcoded answers or dataset-specific labels
- Multi-image claim support
- Graceful degradation during API failures
- Retry-based handling of transient Gemini errors
- Structured, explainable decision making
- Modular architecture for easy experimentation and extension

---

## Future Improvements

- Migration to the latest Google GenAI SDK
- Better damage localization
- Confidence scoring
- Human-review workflow for uncertain claims
- Fine-tuned domain-specific vision models

---

## Submission Notes

- Processes claim conversations and image evidence end-to-end.
- Supports multiple images per claim.
- Includes evaluation utilities and strategy comparison framework.
- Generates output.csv in the required submission schema.
- Designed to degrade gracefully under API failures and rate limits.
