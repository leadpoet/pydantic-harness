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
    required_attribute = str(normalized.get("required_attribute") or "").strip()
    raw_day = (
        os.environ.get("BAKEOFF_EVALUATION_DATE")
        or os.environ.get("LAB_ARENA_EVALUATION_DATE")
        or ""
    ).strip()
    evaluation_date = date.fromisoformat(raw_day) if raw_day else date.today()
    return (
        f"Evaluation date: {evaluation_date.isoformat()}\n"
        f"Return up to {limit} companies. A company without a verified required intent must not be returned.\n"
        "The matched_icp_signal index refers to intent_contract below. Preserve its "
        "index exactly. Research required=true rows first. Index 0 is the host's required "
        "primary intent; bonus evidence must never replace it. Do not treat a required=false "
        "bonus row as required:\n"
        f"{json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Required fit before submission:\n"
        f"- Geography: {required_geography or 'not specified'}\n"
        f"- Company stage: {required_stage or 'not specified'}\n"
        f"- Required attribute: {required_attribute or 'not specified'}\n"
        "Verify the ICP industry, geography, employee band, stage, and required attribute; "
        "omit any company with a missing or conflicting required fact. If company_stage is "
        "required, verify it independently and preserve the verified stage. Normalize only true "
        "synonyms of known labels such as Seed, Series A, Series B, Series C+, Private Equity, "
        "Public, or Bootstrapped. Series C+ includes verified Series C and later venture rounds. "
        "Use Private Equity only for verified private-equity ownership and Public only for a "
        "company with publicly listed shares. Never copy the requested stage when the evidence "
        "does not prove it. When the ICP lists employee buckets, return one exact listed bucket "
        "after harmless formatting normalization only; otherwise preserve the actual verified band. If "
        "required_attribute is present in the ICP, include a required_attribute object with "
        "the literal same requirement text, passed=true, one direct evidence URL, a supporting quote, "
        "and a short explanation. Do not return a company when either required fact cannot be "
        "verified. Keep fit discovery separate from event verification. For a narrow dated primary "
        "event, begin with one focused search_web news or jobs query; after a dated hit, profile its "
        "domain and fetch_page on the best URL before any more candidate search. Otherwise begin with "
        "search_companies using a short industry or business-type query plus structured fit filters; "
        "keep intent and a literal stage label out of that query. For Series C+, search for and verify "
        "Series C, Series D, or a later venture round, not the label alone. After empty discovery, "
        "change the query and loosen exactly one discovery filter or use one broad search_web fallback; "
        "never loosen final fit. Make at most two search_companies calls and three total candidate-finding "
        "calls before verifying an available plausible candidate. "
        f"Shortlist at most {min(limit + 2, 7)} domains and expand once only when fewer than "
        "the requested count remain after fit checks. A verified_example_company is input context, "
        "not an answer; never return it unless tool evidence independently qualifies it in this run. "
        "Use get_company_profile or one focused fit search only when discovery lacks a required fit "
        "fact. A profile database can be stale: when a current company page or LinkedIn company "
        "page contradicts it, use the current supported fact, not the older profile value. "
        "Never choose an employee band just because it appears in the ICP. "
        "For each qualified domain, try get_company_events or one focused search_web query for the "
        "primary intent and use the other only when the first has no usable evidence. Fetch the best "
        "evidence URL, "
        "with one alternate after a failed or unsupported page. Never repeat an equivalent query, "
        "domain lookup, or URL. Quote the fetched page, not a search-result snippet. The signal date "
        "must be the actual event or announcement date; never substitute a crawl, page-update, or search "
        "index date. Preserve event status: beta, preview, pilot, or a future announcement is not "
        "general availability. For an appointment, distinguish announcement from effective or start "
        "date; use the date of the claimed event and never treat a future start as completed. Industry, "
        "size, and general activity prove fit, not buying intent. In why_now, state the verified event, "
        "then one commercial implication clearly as a possibility. Separate inference from sourced "
        "fact and mention only the relevant ICP product_service. Never copy unrelated offerings or "
        "invent procurement, budget, demand, vendor evaluation, or purchase plans. Avoid benchmark, "
        "ICP match, scoring, or qualification jargon. "
        "Vague claims such as 'the company is growing' are insufficient. Submit as soon as enough "
        "companies pass or the remaining budget "
        "cannot improve the result. Company homepages may support identity or fit, but cannot alone prove "
        "a dated intent event. Submit ordinary JSON matching the declared company schema."
    )
