from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import inspect
from io import StringIO
import json
import unittest
from unittest.mock import patch

import harness
from pydantic import ValidationError

from experiments.harness_bakeoff.models import (
    CompanyResult,
    company_list_json_schema,
    normalize_icp,
    validate_companies,
)
from experiments.harness_bakeoff.prompt import build_prompt
from experiments.harness_bakeoff import worker
from experiments.harness_bakeoff.worker import MODULES


class HarnessContractTests(unittest.TestCase):
    def test_selected_harness_exposes_run_icp(self) -> None:
        for arm, module_name in MODULES.items():
            with self.subTest(arm=arm):
                module = importlib.import_module(module_name)
                self.assertEqual(
                    list(inspect.signature(module.run_icp).parameters), ["icp"]
                )

    def test_top_level_entrypoint_exports_selected_harness(self) -> None:
        selected = importlib.import_module(MODULES["pydantic_ai"])

        self.assertEqual(MODULES["pydantic_ai"], "harness")
        self.assertIs(harness.run_icp, selected.run_icp)
        self.assertEqual(list(inspect.signature(harness.run_icp).parameters), ["icp"])
        self.assertIsInstance(harness.get_last_usage(), dict)

    def test_worker_calls_the_public_run_icp_boundary(self) -> None:
        icp = {"icp_id": "icp_today"}
        output = StringIO()

        with patch.object(harness, "run_icp", return_value=[]) as run_icp:
            with patch("sys.stdin", StringIO(json.dumps(icp))):
                with redirect_stdout(output):
                    returncode = worker.main(["pydantic_ai"])

        self.assertEqual(returncode, 0)
        run_icp.assert_called_once_with(icp)
        payload = json.loads(output.getvalue().removeprefix(worker.SENTINEL))
        self.assertEqual(payload["companies"], [])

    def test_company_output_round_trip(self) -> None:
        company = CompanyResult.model_validate(
            {
                "company_name": "Example",
                "company_website": "https://example.com",
                "company_linkedin": "",
                "industry": "Software",
                "employee_count": "51-200",
                "company_stage": "Series A",
                "country": "United States",
                "state": "California",
                "fit_summary": "Matches the example ICP.",
                "fit_evidence_urls": ["https://example.com/about"],
                "intent_signals": [
                    {
                        "matched_icp_signal": 0,
                        "description": "A current product event.",
                        "date": "2026-08-20",
                        "why_now": "The event gives a timely contact reason.",
                        "url": "https://example.com/news/event",
                        "snippet": "Example launched the product.",
                    }
                ],
                "required_attribute": {
                    "text": "Uses a product-led sales motion",
                    "passed": True,
                    "evidence_url": "https://example.com/about",
                    "evidence_quote": "Customers can start with a self-service plan.",
                    "explanation": "The self-service plan supports the required motion.",
                },
            }
        )
        dumped = company.model_dump(mode="json")

        self.assertEqual(
            CompanyResult.model_validate(dumped).model_dump(mode="json"), dumped
        )
        self.assertEqual(validate_companies([dumped]), [dumped])

    def test_arena_date_anchors_the_prompt(self) -> None:
        with patch.dict(
            "os.environ",
            {"LAB_ARENA_EVALUATION_DATE": "2026-09-04"},
            clear=True,
        ):
            prompt = build_prompt({"icp_id": "today"}, max_companies=1)

        self.assertTrue(prompt.startswith("Evaluation date: 2026-09-04\n"))

    def test_current_multi_intent_shape_preserves_primary_and_bonus_metadata(self) -> None:
        raw = {
            "icp_id": "today",
            "employee_count": ["51-200", "201-500"],
            "intent_signal": "Raised a growth round",
            "intent_category": "funding",
            "intent_max_age_days": 365,
            "intent_signals": [
                "Raised a growth round",
                "Announced a strategic partnership",
            ],
            "bonus_intents": [
                {
                    "intent_signal": "Announced a strategic partnership",
                    "intent_category": "partnership",
                    "intent_max_age_days": 180,
                }
            ],
        }

        normalized = normalize_icp(raw)

        self.assertEqual(
            normalized["intent_contract"],
            [
                {
                    "index": 0,
                    "signal": "Raised a growth round",
                    "category": "FUNDING",
                    "max_age_days": 365,
                    "required": True,
                },
                {
                    "index": 1,
                    "signal": "Announced a strategic partnership",
                    "category": "PARTNERSHIP",
                    "max_age_days": 180,
                    "required": False,
                },
            ],
        )
        self.assertEqual(
            normalized["required_intents"],
            [
                {
                    "signal": "Raised a growth round",
                    "category": "FUNDING",
                    "max_age_days": 365,
                }
            ],
        )
        self.assertEqual(normalized["intent_category"], "FUNDING")
        self.assertEqual(normalized["intent_max_age_days"], 365)
        self.assertEqual(normalize_icp(normalized), normalized)

    def test_explicit_required_intents_do_not_promote_bonus(self) -> None:
        normalized = normalize_icp(
            {
                "icp_id": "today",
                "employee_count": "51-200|201-500",
                "required_intents": [
                    {
                        "signal": "Raised capital",
                        "category": "funding",
                        "max_age_days": 90,
                    }
                ],
                "bonus_intents": [
                    {
                        "signal": "Opened an office",
                        "category": "market_expansion",
                        "max_age_days": 180,
                    }
                ],
            }
        )

        self.assertEqual(
            normalized["intent_signals"], ["Raised capital", "Opened an office"]
        )
        self.assertTrue(normalized["intent_contract"][0]["required"])
        self.assertFalse(normalized["intent_contract"][1]["required"])
        self.assertEqual(normalized["intent_contract"][1]["index"], 1)

    def test_prompt_prioritizes_primary_and_requires_event_grounding(self) -> None:
        prompt = build_prompt(
            {
                "icp_id": "today",
                "industry": "Software",
                "employee_count": ["51-200"],
                "company_stage": "Series A",
                "country": "United States",
                "product_service": "A workflow platform",
                "intent_signal": "Raised funding",
                "intent_category": "FUNDING",
                "intent_max_age_days": 365,
                "intent_signals": ["Raised funding", "Opened an office"],
                "bonus_intents": [
                    {"signal": "Opened an office", "category": "MARKET_EXPANSION"}
                ],
            }
        )

        self.assertIn("Index 0 is the host's required primary intent", prompt)
        self.assertIn("bonus evidence must never replace it", prompt)
        self.assertIn("Keep fit discovery separate from event verification", prompt)
        self.assertIn(
            "begin with one focused search_web news or jobs query",
            prompt,
        )
        self.assertIn("at most two search_companies calls", prompt)
        self.assertIn("loosen exactly one discovery filter", prompt)
        self.assertIn("never loosen final fit", prompt)
        self.assertIn("three total candidate-finding", prompt)
        self.assertIn("before verifying an available plausible candidate", prompt)
        self.assertIn("profile its domain and fetch_page", prompt)
        self.assertIn("Series C, Series D, or a later venture round", prompt)
        self.assertIn("Quote the fetched page's main article", prompt)
        self.assertIn("not search snippets, navigation, or related-article cards", prompt)
        self.assertIn("never attach the surrounding page's date to a linked event", prompt)
        self.assertIn("never substitute a crawl, page-update, or search index date", prompt)
        self.assertIn("state the verified event", prompt)
        self.assertIn("commercial implication clearly as a possibility", prompt)
        self.assertIn("Separate inference from sourced fact", prompt)

    def test_prompt_preserves_event_status_and_avoids_sales_fabrication(self) -> None:
        prompt = build_prompt(
            {
                "icp_id": "today",
                "industry": "Data and Analytics",
                "company_stage": "Series C+",
                "product_service": ["analytics", "workflow", "security"],
                "intent_signal": "Launched a product or appointed an executive",
                "intent_category": "PRODUCT_LAUNCH",
                "intent_max_age_days": 365,
            }
        )

        self.assertIn('"company_stage": "Series C+"', prompt)
        self.assertIn(
            "beta, preview, pilot, or a future announcement is not general availability",
            prompt,
        )
        self.assertIn(
            "distinguish announcement from effective or start date",
            prompt,
        )
        self.assertIn("never treat a future start as completed", prompt)
        self.assertIn("Never copy unrelated offerings", prompt)
        self.assertIn(
            "invent procurement, budget, demand, vendor evaluation, or purchase plans",
            prompt,
        )
        self.assertIn("Avoid benchmark, ICP match, scoring, or qualification jargon", prompt)
        self.assertIn("product_service describes what the target company sells", prompt)
        self.assertIn("not the seller's offering or what the target wants to buy", prompt)
        self.assertIn("verified event's effect on the target's actual operations or growth", prompt)

    def test_prompt_does_not_trust_stored_linkedin_url(self) -> None:
        prompt = build_prompt({"icp_id": "today"})

        self.assertIn(
            "stored profile's LinkedIn URL as an unverified candidate",
            prompt,
        )
        self.assertIn("Use current page evidence for the canonical company URL", prompt)
        self.assertIn(
            "if it cannot be verified, leave the optional company_linkedin field empty",
            prompt,
        )

    def test_output_schema_guides_stage_without_narrowing_host_contract(self) -> None:
        schema = company_list_json_schema()["items"]

        self.assertNotIn("company_stage", schema["required"])
        self.assertNotIn("enum", schema["properties"]["company_stage"])
        self.assertIn(
            "Series C+",
            schema["properties"]["company_stage"]["description"],
        )

    def test_certification_guidance_requires_a_real_granted_event(self) -> None:
        icp = {
            "icp_id": "test-certification",
            "intent_signal": "Announced a certification milestone",
            "intent_category": "REGULATORY_CLEARANCE",
            "intent_max_age_days": 365,
        }
        prompt = build_prompt(icp)
        self.assertIn("actually granted to this company or product and its date", prompt)
        self.assertIn("do not invent one or require an undisclosed auditor's name", prompt)
        self.assertIn("A marketplace listing, partner badge", prompt)
        self.assertIn("check the latest funding or ownership status", prompt)
        icp["intent_category"] = "FUNDING"
        self.assertNotIn("A marketplace listing, partner badge", build_prompt(icp))

    def test_hiring_guidance_requires_direct_function_match(self) -> None:
        icp = {
            "icp_id": "test-hiring",
            "intent_signal": "Hiring for integration roles",
            "intent_category": "HIRING",
            "intent_max_age_days": 365,
        }
        prompt = build_prompt(icp)

        self.assertIn(
            "job responsibilities directly match the requested function",
            prompt,
        )
        self.assertIn(
            "shared words such as systems or platform are insufficient",
            prompt,
        )
        self.assertIn(
            "Do not treat generic hiring or an adjacent function",
            prompt,
        )
        icp["intent_category"] = "FUNDING"
        self.assertNotIn(
            "job responsibilities directly match the requested function",
            build_prompt(icp),
        )

    def test_output_canonicalizes_true_stage_and_employee_format_synonyms(self) -> None:
        base = {
            "company_name": "Example",
            "company_website": "https://example.com",
            "industry": "Software",
            "employee_count": "1,001–5,000 employees",
            "company_stage": "PE-backed",
            "country": "United States",
            "fit_summary": "Matches the example ICP.",
            "fit_evidence_urls": ["https://example.com/about"],
            "intent_signals": [
                {
                    "matched_icp_signal": 0,
                    "description": "Example raised capital.",
                    "date": "2026-08-20",
                    "why_now": "The capital supports growth.",
                    "url": "https://example.com/news/event",
                    "snippet": "Example announced the transaction.",
                }
            ],
        }

        dumped = CompanyResult.model_validate(base).model_dump(mode="json")

        self.assertEqual(dumped["employee_count"], "1,001-5,000")
        self.assertEqual(dumped["company_stage"], "Private Equity")
        self.assertEqual(
            CompanyResult.model_validate(
                {**base, "company_stage": "bootstrapped"}
            ).company_stage,
            "Bootstrapped",
        )
        self.assertEqual(
            CompanyResult.model_validate_json(json.dumps(dumped)).model_dump(mode="json"),
            dumped,
        )

    def test_output_does_not_relabel_ambiguous_stage_or_employee_range(self) -> None:
        base = {
            "company_name": "Example",
            "company_website": "https://example.com",
            "industry": "Software",
            "employee_count": "51-500",
            "company_stage": "growth stage",
            "country": "United States",
            "fit_summary": "Matches the example ICP.",
            "fit_evidence_urls": ["https://example.com/about"],
            "intent_signals": [
                {
                    "matched_icp_signal": 0,
                    "description": "Example raised capital.",
                    "date": "2026-08-20",
                    "why_now": "The capital supports growth.",
                    "url": "https://example.com/news/event",
                    "snippet": "Example announced the transaction.",
                }
            ],
        }

        dumped = CompanyResult.model_validate(base).model_dump(mode="json")

        self.assertEqual(dumped["employee_count"], "51-500")
        self.assertEqual(dumped["company_stage"], "growth stage")

    def test_output_rejects_non_public_and_invalid_port_urls(self) -> None:
        base = {
            "company_name": "Example",
            "company_website": "https://example.com",
            "industry": "Software",
            "employee_count": "51-200",
            "company_stage": "Series A",
            "country": "United States",
            "fit_summary": "Matches the example ICP.",
            "fit_evidence_urls": ["https://example.com/about"],
            "intent_signals": [
                {
                    "matched_icp_signal": 0,
                    "description": "A current product event.",
                    "date": "2026-08-20",
                    "why_now": "The event gives a timely contact reason.",
                    "url": "https://example.com/news/event",
                    "snippet": "Example launched the product.",
                }
            ],
        }
        for bad_url in ("https://internal.test/event", "https://example.com:bad/event"):
            value = dict(base)
            value["company_website"] = bad_url
            with self.subTest(url=bad_url):
                with self.assertRaises(ValidationError):
                    CompanyResult.model_validate(value)

    def test_output_rejects_non_company_linkedin_url_and_hidden_scorer_overflow(
        self,
    ) -> None:
        base = {
            "company_name": "Example",
            "company_website": "https://example.com",
            "industry": "Software",
            "employee_count": "51-200",
            "company_stage": "Series A",
            "country": "United States",
            "fit_summary": "Matches the example ICP.",
            "fit_evidence_urls": ["https://example.com/about"],
            "intent_signals": [
                {
                    "matched_icp_signal": 0,
                    "description": "A current product event.",
                    "date": "2026-08-20",
                    "why_now": "The event gives a timely contact reason.",
                    "url": "https://example.com/news/event",
                    "snippet": "Example launched the product.",
                }
            ],
        }
        for field, value in (
            ("company_linkedin", "https://example.com/company/example"),
            ("company_name", "x" * 201),
            ("fit_summary", "x" * 501),
        ):
            candidate = dict(base)
            candidate[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    CompanyResult.model_validate(candidate)


if __name__ == "__main__":
    unittest.main()
