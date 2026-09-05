"""One prompt used unchanged by all challenger harnesses."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from .models import normalize_icp


SYSTEM_PROMPT = """You are a rigorous B2B account researcher. Find companies that fit the supplied ICP and have the REQUIRED recent intent. Use only the provided tools. Never rely on memory for a factual claim. Verify company fit, company stage, every required attribute, and each intent against public source content. Preserve exact source URLs and reject stale, ambiguous, homepage-only, or wrong-company evidence. Prefer direct company, job, regulatory, filing, or reputable news pages. Return at most the requested number, ranked best first. Explain fit and why-now in plain language useful to a salesperson. Do not invent missing facts. Call submit_companies exactly once when done."""


def build_prompt(icp: dict[str, Any], max_companies: int | None = None) -> str:
    normalized = normalize_icp(icp)
    limit = max(1, min(int(max_companies or 5), 5))
    required_geography = str(
        normalized.get("geography") or normalized.get("country") or ""
    ).strip()
    required_stage = str(normalized.get("company_stage") or "").strip()
    qualification_gate = "\n".join(
        (
            "Required qualification gate before intent research:",
            f"- Geography: {required_geography or 'not specified'}",
            f"- Company stage: {required_stage or 'not specified'}",
            "Before intent research, reject any candidate whose required geography or company stage is not verified.",
        )
    )
    raw_day = (
        os.environ.get("BAKEOFF_EVALUATION_DATE")
        or os.environ.get("LAB_ARENA_EVALUATION_DATE")
        or ""
    ).strip()
    evaluation_date = date.fromisoformat(raw_day) if raw_day else date.today()
    return (
        f"Evaluation date: {evaluation_date.isoformat()}\n"
        f"Return up to {limit} companies. A company without a verified required intent must not be returned.\n"
        f"{qualification_gate}\n"
        "The matched_icp_signal index refers to required_intents in this normalized ICP:\n"
        f"{json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "If company_stage is required, return the verified stage in company_stage. If "
        "required_attribute is present in the ICP, include a required_attribute object with "
        "the same requirement text, passed=true, one direct evidence URL, a supporting quote, "
        "and a short explanation. Do not return a company when either required fact cannot be "
        "verified. Every submitted evidence URL must be an exact public URL returned by a tool; "
        "never invent, edit, or infer a URL. Write each why_now as one plain, non-technical sentence "
        "that tells a salesperson what changed and why outreach is timely. Use one bounded funnel: "
        "start with one search_companies call and shortlist at most "
        f"{min(limit + 1, 6)} domains; repeat discovery only if it returns no usable domains. "
        "Use get_company_profile or one focused fit search only when discovery lacks required geography "
        "or stage evidence. Do not call get_company_events or run an intent search for a domain until "
        "that qualification gate passes. Research only the REQUIRED intent categories. For each qualified "
        "domain, try get_company_events OR search_web first and use the other only when the first has no "
        "usable evidence. Fetch only the best evidence URL. If its live fetch fails or returns no usable "
        "text, try one bounded alternate; if that also fails, omit the intent and company. Never convert a "
        "provider failure into verified evidence. Never repeat an equivalent query, domain lookup, or URL. "
        "Submit as soon as enough companies "
        "pass or the remaining budget cannot improve the result. Company homepages may support identity or "
        "fit, but cannot alone prove a dated intent event. Submit ordinary JSON matching the declared company schema."
    )
