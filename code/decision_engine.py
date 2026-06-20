"""
Decision engines that turn (claimed issue, visible issue evidence) into a
final claim_status. Two strategies are provided so they can be compared
head-to-head in evaluation/evaluate.py, per the hackathon's requirement
to compare at least two strategies/configs before picking a final one.

Strategy A — "rule_based": fast, deterministic, cheap. Compares the
  claimed_issue against the best image's issue_type using an explicit
  compatibility table (config.COMPATIBLE_ISSUE_PAIRS).

Strategy B — "vlm_judgment": one extra LLM call that's shown the claim
  summary + the structured vision findings (not the raw image again, to
  keep cost down) and asked to make the final call directly. Slower /
  costs more per claim, but can reason about ambiguous cases the rule
  table doesn't cover (e.g. claimed "scratch", visible "stain").
"""

from config import ALLOWED_CLAIM_STATUS, COMPATIBLE_ISSUE_PAIRS
from model_client import generate_json


def decide_rule_based(claim_result, vision_best):
    claimed_issue = claim_result.get("claimed_issue", "unknown")
    visible_issue = vision_best.get("issue_type", "unknown")

    if visible_issue == "unknown":
        return "not_enough_information", "Visible damage type could not be determined from the image(s)."

    if visible_issue == "none":
        return "contradicted", "No visible damage found in the submitted image(s), contradicting the claim."

    if claimed_issue == "unknown":
        return "not_enough_information", "Claim text did not specify a clear damage type to compare against."

    if claimed_issue == visible_issue:
        return "supported", f"Claimed {claimed_issue} matches visible {visible_issue} in the image."

    if (claimed_issue, visible_issue) in COMPATIBLE_ISSUE_PAIRS:
        return "supported", f"Claimed {claimed_issue} is consistent with visible {visible_issue} in the image."

    return (
        "not_enough_information",
        f"Claimed {claimed_issue} does not clearly match visible {visible_issue}; needs human review.",
    )


_JUDGMENT_PROMPT = """You are an insurance claim adjudicator. Decide whether the
photo evidence SUPPORTS, CONTRADICTS, or gives NOT_ENOUGH_INFORMATION for this claim.

Claim summary: {summary}
Claimed issue: {claimed_issue}
Claimed part: {claimed_part}

Image evidence (best image found): issue_type={issue_type}, object_part={object_part},
severity={severity}, damage_visible={damage_visible}

Return ONLY valid JSON in exactly this shape:
{{
  "claim_status": "supported" | "contradicted" | "not_enough_information",
  "claim_status_justification": "one concise sentence grounded in the evidence above"
}}
"""


def decide_vlm_judgment(claim_result, vision_best):
    prompt = _JUDGMENT_PROMPT.format(
        summary=claim_result.get("summary", ""),
        claimed_issue=claim_result.get("claimed_issue", "unknown"),
        claimed_part=claim_result.get("claimed_part", "unknown"),
        issue_type=vision_best.get("issue_type", "unknown"),
        object_part=vision_best.get("object_part", "unknown"),
        severity=vision_best.get("severity", "unknown"),
        damage_visible=vision_best.get("damage_visible", False),
    )
    try:
        result = generate_json(prompt)
        status = result.get("claim_status")
        if status not in ALLOWED_CLAIM_STATUS:
            status = "not_enough_information"
        justification = result.get("claim_status_justification", "Model judgment call.")
        return status, justification
    except Exception:
        return "not_enough_information", "Decision model call failed after retries."


STRATEGIES = {
    "rule_based": decide_rule_based,
    "vlm_judgment": decide_vlm_judgment,
}


def decide_claim(claim_result, vision_best, strategy="rule_based"):
    fn = STRATEGIES.get(strategy, decide_rule_based)
    return fn(claim_result, vision_best)
