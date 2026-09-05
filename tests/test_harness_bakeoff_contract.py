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

from experiments.harness_bakeoff.models import CompanyResult, validate_companies
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

    def test_prompt_qualifies_stage_and_geography_before_intent(self) -> None:
        prompt = build_prompt(
            {
                "icp_id": "today",
                "geography": "United States",
                "company_stage": "Series A",
                "intent_signal": "recent funding",
            },
            max_companies=2,
        )

        self.assertIn("- Geography: United States", prompt)
        self.assertIn("- Company stage: Series A", prompt)
        qualification = (
            "Before intent research, reject any candidate whose required geography "
            "or company stage is not verified."
        )
        intent_call = "Do not call get_company_events or run an intent search"
        self.assertIn(qualification, prompt)
        self.assertLess(prompt.index(qualification), prompt.index(intent_call))
        self.assertIn(
            "Every submitted evidence URL must be an exact public URL returned by a tool",
            prompt,
        )
        self.assertIn("Write each why_now as one plain, non-technical sentence", prompt)
        self.assertIn("If its live fetch fails or returns no usable text", prompt)
        self.assertIn("Never convert a provider failure into verified evidence", prompt)

    def test_output_rejects_non_public_and_invalid_port_urls(self) -> None:
        base = {
            "company_name": "Example",
            "company_website": "https://example.com",
            "industry": "Software",
            "employee_count": "51-200",
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
