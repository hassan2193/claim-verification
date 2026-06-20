"""
Thin wrapper around google.generativeai that adds:
  - retry with exponential backoff (handles 429 / transient errors)
  - robust JSON extraction from model responses
  - usage tracking (call count, tokens, images) so the evaluation
    report can compute approximate cost and TPM/RPM exposure.
"""

import json
import os
import re
import time

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from config import GEMINI_MODEL

_model = genai.GenerativeModel(GEMINI_MODEL)

# Pricing is approximate (USD per 1M tokens) for gemini-2.5-flash as of
# early 2026. Update these if pricing changes — only used for the
# operational-cost estimate in the evaluation report, not for billing.
PRICE_PER_M_INPUT_TOKENS = 0.30
PRICE_PER_M_OUTPUT_TOKENS = 2.50


class UsageTracker:
    """Accumulates call/token/image counts across a run for cost reporting."""

    def __init__(self):
        self.calls = 0
        self.retries = 0
        self.failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.images_sent = 0
        self.total_seconds = 0.0

    def record(self, usage_metadata, elapsed, images=0):
        self.calls += 1
        self.images_sent += images
        self.total_seconds += elapsed
        if usage_metadata is not None:
            self.input_tokens += getattr(usage_metadata, "prompt_token_count", 0) or 0
            self.output_tokens += getattr(usage_metadata, "candidates_token_count", 0) or 0

    def estimated_cost_usd(self):
        return (
            self.input_tokens / 1_000_000 * PRICE_PER_M_INPUT_TOKENS
            + self.output_tokens / 1_000_000 * PRICE_PER_M_OUTPUT_TOKENS
        )

    def summary(self):
        return {
            "calls": self.calls,
            "retries": self.retries,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "images_sent": self.images_sent,
            "total_seconds": round(self.total_seconds, 2),
            "avg_seconds_per_call": round(self.total_seconds / self.calls, 2) if self.calls else 0,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
        }


USAGE = UsageTracker()

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text):
    """Pull a JSON object out of a model response, tolerating code fences
    and stray commentary before/after the JSON block."""
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in model response: {text[:200]!r}")


def generate_json(parts, max_retries=4, base_delay=2.0, images=0):
    """Call the model with `parts` (a prompt string, or [prompt, image, ...]),
    retrying on transient failures, and return the parsed JSON dict.

    Raises the last exception if all retries are exhausted; callers should
    catch this and fall back to a safe default rather than crashing the run.
    """
    last_err = None
    for attempt in range(max_retries):
        start = time.time()
        try:
            response = _model.generate_content(parts)
            elapsed = time.time() - start
            USAGE.record(getattr(response, "usage_metadata", None), elapsed, images=images)
            return _extract_json(response.text)
        except Exception as e:  # noqa: BLE001 - we want to retry broadly here
            last_err = e
            USAGE.retries += 1
            # exponential backoff with jitter-free simplicity is fine for a
            # hackathon-scale batch job; real production code would jitter.
            sleep_for = base_delay * (2 ** attempt)
            time.sleep(sleep_for)
    USAGE.failures += 1
    raise last_err
