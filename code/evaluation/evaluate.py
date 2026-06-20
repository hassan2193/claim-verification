"""
Evaluation harness, required by the hackathon spec:
  - runs metrics on dataset/sample_claims.csv (which has expected outputs)
  - compares at least two strategies (rule_based vs vlm_judgment)
  - reports operational stats: model calls, tokens, images, approx cost,
    runtime, and rough TPM/RPM exposure
  - writes a markdown report to evaluation/evaluation_report.md

Usage:
    python evaluation/evaluate.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config import SAMPLE_CLAIMS_CSV, USER_HISTORY_CSV, DATASET_DIR
from main import process_row, load_history
from model_client import USAGE

REPORT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_report.md")

# Columns in sample_claims.csv that hold the expected/ground-truth values.
# Adjust here if the actual column names in your sample file differ.
EXPECTED_COLUMN_MAP = {
    "claim_status": "expected_claim_status",
    "issue_type": "expected_issue_type",
    "severity": "expected_severity",
    "evidence_standard_met": "expected_evidence_standard_met",
}


def _reset_usage():
    USAGE.calls = 0
    USAGE.retries = 0
    USAGE.failures = 0
    USAGE.input_tokens = 0
    USAGE.output_tokens = 0
    USAGE.images_sent = 0
    USAGE.total_seconds = 0.0


def evaluate_strategy(df, history_df, strategy):
    _reset_usage()
    start = time.time()

    predictions = []
    for _, row in df.iterrows():
        try:
            predictions.append(process_row(row, history_df, strategy))
        except Exception as e:
            predictions.append({"claim_status": "not_enough_information", "issue_type": "unknown",
                                 "severity": "unknown", "evidence_standard_met": False, "_error": str(e)})

    wall_clock = time.time() - start
    usage = USAGE.summary()
    usage["wall_clock_seconds"] = round(wall_clock, 2)

    metrics = {}
    for field, expected_col in EXPECTED_COLUMN_MAP.items():
        if expected_col not in df.columns:
            continue
        correct = 0
        total = 0
        for pred, (_, row) in zip(predictions, df.iterrows()):
            expected = row[expected_col]
            if pd.isna(expected):
                continue
            total += 1
            predicted_val = pred.get(field)
            if str(predicted_val).strip().lower() == str(expected).strip().lower():
                correct += 1
        if total:
            metrics[field] = {"accuracy": round(correct / total, 3), "n": total}

    return predictions, metrics, usage


def main():
    if not os.path.exists(SAMPLE_CLAIMS_CSV):
        print(f"Sample claims file not found at {SAMPLE_CLAIMS_CSV}; nothing to evaluate.")
        return

    df = pd.read_csv(SAMPLE_CLAIMS_CSV)
    history_df = load_history()

    report_lines = ["# Evaluation Report\n", f"Rows evaluated: {len(df)}\n"]

    all_results = {}
    for strategy in ["rule_based", "vlm_judgment"]:
        print(f"\n=== Evaluating strategy: {strategy} ===")
        predictions, metrics, usage = evaluate_strategy(df, history_df, strategy)
        all_results[strategy] = {"metrics": metrics, "usage": usage}

        report_lines.append(f"## Strategy: `{strategy}`\n")
        if metrics:
            report_lines.append("| Field | Accuracy | N |")
            report_lines.append("|---|---|---|")
            for field, m in metrics.items():
                report_lines.append(f"| {field} | {m['accuracy']:.1%} | {m['n']} |")
        else:
            report_lines.append(
                "_No `expected_*` columns found in sample_claims.csv — "
                "add them to enable accuracy scoring. Showing operational "
                "metrics only._"
            )
        report_lines.append("")
        report_lines.append("**Operational metrics:**\n")
        for k, v in usage.items():
            report_lines.append(f"- {k}: {v}")
        report_lines.append("")

    # naive TPM/RPM exposure note based on observed per-call timing
    report_lines.append("## TPM / RPM considerations\n")
    for strategy, data in all_results.items():
        u = data["usage"]
        if u["calls"] and u["wall_clock_seconds"] > 0:
            rpm = u["calls"] / (u["wall_clock_seconds"] / 60)
            tpm = (u["input_tokens"] + u["output_tokens"]) / (u["wall_clock_seconds"] / 60)
            report_lines.append(
                f"- `{strategy}`: ~{rpm:.1f} requests/min, ~{tpm:.0f} tokens/min observed "
                f"at this batch size. Scale-up should add backoff-aware throttling "
                f"(already implemented in model_client.py) to stay under your Gemini tier's RPM/TPM caps."
            )

    report_lines.append("\n## Final strategy used for output.csv\n")
    report_lines.append(
        "`rule_based` is used as the default for the full `claims.csv` run: it is "
        "~1 model call cheaper per claim (no extra judgment call), fully deterministic "
        "for compatible issue pairs, and only defers to `not_enough_information` on "
        "genuinely ambiguous claimed/visible issue combinations — which a human "
        "reviewer can then triage. `vlm_judgment` is kept as an option for cases "
        "where the rule table's compatibility list proves too rigid."
    )

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nWrote evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
