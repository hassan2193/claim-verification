"""
Main entry point. Processes every row of dataset/claims.csv and writes
output.csv with the exact required schema/column order.

Usage:
    python main.py                  # full run, default strategy
    python main.py --strategy vlm_judgment
    python main.py --limit 20       # quick smoke test on first 20 rows
    python main.py --input ../dataset/sample_claims.csv --output sample_predictions.csv
"""

import argparse
import os
import sys

import pandas as pd

from config import (
    CLAIMS_CSV,
    DATASET_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    USER_HISTORY_CSV,
)
from claim_parser import extract_claim
from decision_engine import decide_claim
from evidence_checker import check_evidence_standard
from model_client import USAGE
from risk_analyzer import get_risk_flags
from vision_analyzer import analyze_images


def load_history():
    try:
        return pd.read_csv(USER_HISTORY_CSV)
    except FileNotFoundError:
        print(f"WARNING: {USER_HISTORY_CSV} not found; risk_flags will default to 'none'.")
        return None


def process_row(row, history_df, strategy):
    user_id = row["user_id"]
    claim_object = row["claim_object"]
    conversation = row["user_claim"]
    image_paths = [p for p in str(row["image_paths"]).split(";") if p.strip()]

    risk_flags = get_risk_flags(user_id, history_df)
    claim_result = extract_claim(conversation, claim_object)
    per_image, best = analyze_images(image_paths, DATASET_DIR)

    if best is None:
        best = {
            "issue_type": "unknown", "object_part": "unknown",
            "severity": "unknown", "damage_visible": False,
            "image_usable": False,
        }

    evidence_met, evidence_reason, valid_image = check_evidence_standard(
        claim_object, image_paths, per_image
    )

    claim_status, justification = decide_claim(claim_result, best, strategy=strategy)

    supporting_ids = [
        p["image_id"] for p in per_image
        if p.get("issue_type") == best.get("issue_type") and p.get("issue_type") not in ("none", "unknown")
    ]
    supporting_image_ids = ";".join(supporting_ids) if supporting_ids else "none"

    return {
        "user_id": user_id,
        "image_paths": row["image_paths"],
        "user_claim": conversation,
        "claim_object": claim_object,
        "evidence_standard_met": evidence_met,
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": risk_flags,
        "issue_type": best.get("issue_type", "unknown"),
        "object_part": best.get("object_part", "unknown"),
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": supporting_image_ids,
        "valid_image": valid_image,
        "severity": best.get("severity", "unknown"),
    }


def run(input_csv, output_csv, strategy, limit=None):
    df = pd.read_csv(input_csv)
    
    if limit:
        df = df.head(limit)
    history_df = load_history()

    print(f"Processing {len(df)} rows from {input_csv} with strategy='{strategy}'...")

    results = []
    for idx, row in df.iterrows():
        try:
            result = process_row(row, history_df, strategy)
        except Exception as e:  # never let one bad row kill the whole batch
            print(f"  Row {idx} ({row.get('user_id')}) failed: {e}")
            result = {
                "user_id": row.get("user_id"),
                "image_paths": row.get("image_paths"),
                "user_claim": row.get("user_claim"),
                "claim_object": row.get("claim_object"),
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"Processing error: {e}",
                "risk_flags": "none",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Row failed to process.",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown",
            }
        results.append(result)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(df):
            print(f"  ...{idx + 1}/{len(df)} done. Usage so far: {USAGE.summary()}")
            pd.DataFrame(results, columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)

    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    output_df.to_csv(output_csv, index=False)

    print(f"\nDone. Wrote {len(output_df)} rows to {output_csv}")
    print("Final usage summary:", USAGE.summary())
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=CLAIMS_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--strategy", default="rule_based", choices=["rule_based", "vlm_judgment"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY is not set in the environment / .env file.")

    run(args.input, args.output, args.strategy, args.limit)
