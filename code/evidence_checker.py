"""
Checks whether the submitted image set meets the minimum evidence
requirements for a claim_object, using dataset/evidence_requirements.csv.

The schema of that file isn't fixed by the README, so this module is
defensive: it looks for a row matching claim_object and reads whatever
"min_images" / "minimum_images" / similar column it can find. If the
file or a matching row is missing, it falls back to a sane default
(at least 1 usable image showing visible damage) rather than crashing.
"""

import pandas as pd

from config import EVIDENCE_REQUIREMENTS_CSV

_DEFAULT_MIN_IMAGES = 1

_candidate_min_image_cols = [
    "min_images", "minimum_images", "min_image_count", "required_images",
]


def _load_requirements():
    try:
        return pd.read_csv(EVIDENCE_REQUIREMENTS_CSV)
    except FileNotFoundError:
        return None


_REQUIREMENTS_DF = _load_requirements()


def _min_images_for(claim_object):
    if _REQUIREMENTS_DF is None or "claim_object" not in _REQUIREMENTS_DF.columns:
        return _DEFAULT_MIN_IMAGES

    rows = _REQUIREMENTS_DF[_REQUIREMENTS_DF["claim_object"] == claim_object]
    if rows.empty:
        return _DEFAULT_MIN_IMAGES

    row = rows.iloc[0]
    for col in _candidate_min_image_cols:
        if col in row and pd.notna(row[col]):
            try:
                return int(row[col])
            except (ValueError, TypeError):
                continue
    return _DEFAULT_MIN_IMAGES


def check_evidence_standard(claim_object, image_paths, per_image_results):
    """Returns (evidence_standard_met: bool, reason: str, valid_image: bool)."""
    min_images = _min_images_for(claim_object)
    usable = [r for r in per_image_results if r.get("image_usable")]
    showing_damage = [r for r in usable if r.get("issue_type") not in ("none", "unknown")]

    valid_image = len(usable) > 0

    if not valid_image:
        return (
            False,
            "No submitted image was usable for review (missing, corrupt, or too low quality).",
            False,
        )

    if len(image_paths) < min_images:
        return (
            False,
            f"Only {len(image_paths)} image(s) submitted; this claim type requires at least {min_images}.",
            valid_image,
        )

    if not showing_damage:
        return (
            True,
            "Sufficient usable images submitted; none show visible damage, supporting a contradiction or no-issue finding.",
            valid_image,
        )

    return (
        True,
        f"{len(usable)} usable image(s) submitted, meeting the minimum evidence requirement of {min_images}.",
        valid_image,
    )
