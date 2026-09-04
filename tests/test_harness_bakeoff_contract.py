from __future__ import annotations

import importlib
import inspect
import unittest
from unittest import mock

from pydantic import ValidationError

import harness
from experiments.harness_bakeoff.models import CompanyResult, validate_companies
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
        self.assertEqual(list(inspect.signature(harness.run_icp).parameters), ["icp"])
        self.assertEqual(harness.get_last_usage(), {})
        icp = {"icp_id": "contract-test"}
        with mock.patch.object(harness, "_run_icp", return_value=[]) as selected:
            self.assertEqual(harness.run_icp(icp), [])
        selected.assert_called_once_with(icp)

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

        with self.assertRaises(ValidationError):
            CompanyResult.model_validate({**dumped, "undocumented": True})
        with self.assertRaises(ValidationError):
            CompanyResult.model_validate(
                {**dumped, "company_website": "http://127.0.0.1"}
            )


if __name__ == "__main__":
    unittest.main()
