from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import arena_entrypoint


def _company() -> dict:
    return {
        "company_name": "Example",
        "company_website": "https://example.com/",
        "company_linkedin": "https://www.linkedin.com/company/example/",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "California",
        "fit_summary": "The company matches the ICP.",
        "fit_evidence_urls": ["https://example.com/about"],
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "A recent product launch.",
                "date": "2026-09-01",
                "why_now": "The launch creates a timely sales reason.",
                "url": "https://example.com/news/launch",
                "snippet": "Example launched a new product.",
            }
        ],
        "required_attribute": {
            "text": "Uses a product-led sales motion",
            "passed": True,
            "evidence_url": "https://example.com/about",
            "evidence_quote": "Customers can start with a self-service plan.",
            "explanation": "The self-service plan proves the required motion.",
        },
    }


class ArenaEntrypointTests(unittest.TestCase):
    def test_run_once_calls_the_stable_function_and_writes_the_contract(self) -> None:
        seen: list[dict] = []

        def runner(icp: dict) -> list[dict]:
            self.assertEqual(os.environ["BAKEOFF_EVALUATION_DATE"], "2026-09-04")
            self.assertEqual(os.environ["BAKEOFF_MAX_COMPANIES"], "5")
            seen.append(icp)
            return [_company()]

        document = {
            "schema_version": "leadpoet.lab_arena.icp_input.v1",
            "icp": {"icp_id": "daily-1", "employee_count": ["51-200"]},
            "evaluation_date": "2026-09-04",
            "company_limit": 5,
            "provider_operations": ["openrouter.chat", "exa.search"],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "companies.json"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            arena_entrypoint.run_once(
                input_path=str(input_path),
                output_path=str(output_path),
                runner=runner,
            )

            output = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(seen, [document["icp"]])
        self.assertEqual(output["schema_version"], "leadpoet.lab_arena.output.v1")
        self.assertEqual(output["companies"], [_company()])
        self.assertNotIn("BAKEOFF_EVALUATION_DATE", os.environ)
        self.assertNotIn("BAKEOFF_MAX_COMPANIES", os.environ)

    def test_invalid_input_does_not_call_the_runner_or_write_output(self) -> None:
        called = False

        def runner(_: dict) -> list[dict]:
            nonlocal called
            called = True
            return []

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "companies.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema_version": "old",
                        "icp": {"icp_id": "daily-1"},
                        "evaluation_date": "2026-09-04",
                        "company_limit": 5,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported"):
                arena_entrypoint.run_once(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    runner=runner,
                )

            self.assertFalse(output_path.exists())

        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
