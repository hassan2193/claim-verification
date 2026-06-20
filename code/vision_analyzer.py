"""
Analyzes submitted images for visible damage.

Supports multiple images per claim: each image is analyzed independently,
then aggregated into a single "best evidence" result (the image showing
the most severe / most specific damage wins), while keeping the per-image
breakdown so the decision engine and evaluation can use it.
"""

import os

from PIL import Image, UnidentifiedImageError

from config import ALLOWED_ISSUE_TYPES, ALLOWED_SEVERITY
from model_client import generate_json

_ALLOWED_ISSUES_TEXT = "\n".join(ALLOWED_ISSUE_TYPES)
_ALLOWED_SEVERITY_TEXT = "\n".join(ALLOWED_SEVERITY)

_PROMPT = f"""You are an insurance damage reviewer looking at ONE submitted photo.

Return ONLY valid JSON, no commentary, no markdown fences, in exactly this shape:
{{
  "issue_type": "",
  "object_part": "",
  "severity": "",
  "damage_visible": true,
  "image_usable": true,
  "usability_reason": ""
}}

Allowed issue_type values:
{_ALLOWED_ISSUES_TEXT}

Allowed severity values:
{_ALLOWED_SEVERITY_TEXT}

Rules:
- issue_type and severity MUST be exactly one of the allowed values above.
- damage_visible should be false if the object looks undamaged.
- image_usable should be false if the photo is too blurry, too dark, too
  cropped, or otherwise not usable for a damage assessment (set
  usability_reason accordingly). If unusable, still do your best guess on
  the other fields and use "unknown" where you cannot tell.
- object_part should name the specific part visible in the photo (e.g.
  "rear bumper", "laptop screen", "box corner").
"""

_SEVERITY_RANK = {"none": 0, "unknown": 1, "low": 2, "medium": 3, "high": 4}


def _safe_open_image(path):
    try:
        return Image.open(path)
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None


def analyze_image(image_path):
    """Analyze a single image. Returns a dict; on any failure (missing file,
    corrupt image, model error) returns a safe 'unknown' result rather than
    raising, so one bad image doesn't kill the whole batch."""
    img = _safe_open_image(image_path)
    if img is None:
        return {
            "issue_type": "unknown",
            "object_part": "unknown",
            "severity": "unknown",
            "damage_visible": False,
            "image_usable": False,
            "usability_reason": "Image file missing or unreadable.",
        }

    try:
        result = generate_json([_PROMPT, img], images=1)
    except Exception:
        return {
            "issue_type": "unknown",
            "object_part": "unknown",
            "severity": "unknown",
            "damage_visible": False,
            "image_usable": False,
            "usability_reason": "Model call failed after retries.",
        }

    if result.get("issue_type") not in ALLOWED_ISSUE_TYPES:
        result["issue_type"] = "unknown"
    if result.get("severity") not in ALLOWED_SEVERITY:
        result["severity"] = "unknown"
    result.setdefault("object_part", "unknown")
    result.setdefault("image_usable", True)
    result.setdefault("usability_reason", "")
    return result


def analyze_images(image_paths, dataset_root):
    """Analyze every image for a claim. Returns:
      per_image: list of {path, image_id, **analyze_image result}
      best: the per-image result judged most informative (highest severity,
            preferring usable images and non-"none"/"unknown" issue types)
    """
    per_image = []
    for rel_path in image_paths:
        full_path = os.path.join(dataset_root, rel_path.strip())
        analysis = analyze_image(full_path)
        image_id = os.path.splitext(os.path.basename(rel_path.strip()))[0]
        per_image.append({"path": rel_path.strip(), "image_id": image_id, **analysis})

    if not per_image:
        return [], None

    def score(entry):
        usable_bonus = 1 if entry.get("image_usable") else 0
        issue_bonus = 0 if entry["issue_type"] in ("none", "unknown") else 1
        return (usable_bonus, issue_bonus, _SEVERITY_RANK.get(entry["severity"], 0))

    best = max(per_image, key=score)
    return per_image, best
