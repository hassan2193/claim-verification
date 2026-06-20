"""
Shared constants for the claim verification system.
Keep this as the single source of truth for schema / allowed values
so claim_parser, vision_analyzer, decision_engine, and evaluation
all agree with each other.
"""

import os

# ---- paths -----------------------------------------------------------
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CLAIMS_CSV = os.path.join(DATASET_DIR, "claims.csv")
SAMPLE_CLAIMS_CSV = os.path.join(DATASET_DIR, "sample_claims.csv")
USER_HISTORY_CSV = os.path.join(DATASET_DIR, "user_history.csv")
EVIDENCE_REQUIREMENTS_CSV = os.path.join(DATASET_DIR, "evidence_requirements.csv")
IMAGES_DIR = DATASET_DIR  # image_paths in the CSVs are relative to dataset/

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "output.csv")

# ---- output schema (exact order matters for submission) --------------
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# ---- allowed enum values ---------------------------------------------
ALLOWED_ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
]

ALLOWED_SEVERITY = ["none", "low", "medium", "high", "unknown"]

ALLOWED_CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

# claimed_issue / visible_issue pairs that count as a semantic match
# even when the literal strings differ (used by the rule-based decision engine)
COMPATIBLE_ISSUE_PAIRS = {
    ("dent", "broken_part"),
    ("scratch", "broken_part"),
    ("crack", "glass_shatter"),
    ("broken_part", "glass_shatter"),
    ("torn_packaging", "crushed_packaging"),
    ("missing_part", "broken_part"),
}

GEMINI_MODEL = "gemini-2.5-flash"
