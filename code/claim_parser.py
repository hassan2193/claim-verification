"""
Parses the user_claim chat transcript into a structured claim:
  claimed_issue, claimed_part, summary

Uses the allowed-value list from config.py so this stays in sync with
vision_analyzer.py and decision_engine.py.
"""

from config import ALLOWED_ISSUE_TYPES
from model_client import generate_json

_ALLOWED_TEXT = "\n".join(ALLOWED_ISSUE_TYPES)

_PROMPT_TEMPLATE = """You are an insurance claim parser.

Object Type: {claim_object}

Read the conversation below and extract the claim. Return ONLY valid JSON,
no commentary, no markdown fences.

Allowed claimed_issue values:
{allowed}

Rules:
- claimed_issue MUST be exactly one of the allowed values above.
- Normalize the user's language into the closest allowed value (e.g.
  "it's all scratched up" -> scratch, "won't turn on" + visible cracks ->
  use your judgement on the closest physical damage category).
- Do not invent free-text issue descriptions.
- claimed_part should be the specific part the user says is affected
  (e.g. "rear bumper", "screen", "left corner of the box"). Use "unknown"
  if not mentioned.
- summary should be one short sentence describing what the user claims happened.
- Only return "unknown" for claimed_issue if the conversation truly gives no
  hint of what kind of damage is being claimed.

Return JSON in exactly this shape:
{{
  "claimed_issue": "",
  "claimed_part": "",
  "summary": ""
}}

Conversation:
{conversation}
"""


def extract_claim(conversation, claim_object):
    prompt = _PROMPT_TEMPLATE.format(
        claim_object=claim_object,
        allowed=_ALLOWED_TEXT,
        conversation=conversation,
    )
    try:
        result = generate_json(prompt)
    except Exception:
        return {
            "claimed_issue": "unknown",
            "claimed_part": "unknown",
            "summary": "Could not parse claim from conversation (model error).",
        }

    # defensive normalization in case the model returns something odd
    if result.get("claimed_issue") not in ALLOWED_ISSUE_TYPES:
        result["claimed_issue"] = "unknown"
    result.setdefault("claimed_part", "unknown")
    result.setdefault("summary", "")
    return result


if __name__ == "__main__":
    sample = """
    Customer: Hi, I found new damage on my car.
    Customer: The rear bumper has a dent.
    """
    print(extract_claim(sample, "car"))
