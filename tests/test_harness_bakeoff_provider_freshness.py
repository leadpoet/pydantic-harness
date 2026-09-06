from __future__ import annotations

from datetime import date
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.harness_bakeoff import providers


class ProviderFreshnessTests(unittest.TestCase):
    def _tools(self, **kwargs) -> providers.LiveProviderTools:
        return providers.LiveProviderTools(
            deepline_api_key="deepline-test",
            scrapingdog_api_key="scrapingdog-test",
            **kwargs,
        )

    def test_environment_date_anchors_provider_recency(self) -> None:
        with patch.dict(
            os.environ,
            {"BAKEOFF_EVALUATION_DATE": "2026-09-02"},
            clear=False,
        ):
            tools = self._tools()

        with patch.object(
            tools, "_scrapingdog_search", return_value=[]
        ) as scrapingdog_search:
            with patch.object(tools, "_exa_search", return_value=[]) as exa_search:
                tools.search_web(
                    {
                        "query": "product launch",
                        "mode": "news",
                        "limit": 5,
                        "recency_days": 30,
                    }
                )

        scrapingdog_search.assert_called_once_with(
            "product launch after:2026-08-03", 5, "news"
        )
        exa_search.assert_called_once_with("product launch", 5, "news", "2026-08-03")

    def test_explicit_run_date_overrides_environment(self) -> None:
        with patch.dict(
            os.environ,
            {"BAKEOFF_EVALUATION_DATE": "2025-01-01"},
            clear=False,
        ):
            tools = self._tools(evaluation_date="2026-09-02")

        self.assertEqual(tools.evaluation_date, date(2026, 9, 2))

    def test_news_default_window_uses_the_evaluation_date(self) -> None:
        tools = self._tools(evaluation_date="2026-09-02")
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"news_results": []},
        )

        with patch.object(providers.httpx, "get", return_value=response) as request:
            tools._scrapingdog_search("product launch", 1, "news")

        self.assertEqual(
            request.call_args.kwargs["params"]["query"],
            "product launch after:2025-09-02",
        )

    def test_invalid_environment_date_falls_back_to_utc(self) -> None:
        fixed_utc_day = date(2027, 4, 9)

        class FixedDateTime:
            @staticmethod
            def now(_timezone):
                class Current:
                    @staticmethod
                    def date():
                        return fixed_utc_day

                return Current()

        with patch.dict(
            os.environ,
            {"BAKEOFF_EVALUATION_DATE": "not-a-date"},
            clear=False,
        ):
            with patch.object(providers, "datetime", FixedDateTime):
                self.assertEqual(providers._evaluation_day(), fixed_utc_day)

    def test_standalone_discovery_uses_hunter_headcount_bands(self) -> None:
        tools = self._tools()

        with patch.object(
            tools,
            "_deepline",
            return_value={"data": {"data": []}},
        ) as deepline:
            tools.search_companies(
                {
                    "query": "software",
                    "employee_count": [
                        "2-10",
                        "501–1,000",
                        "not-a-band",
                    ],
                }
            )

        self.assertEqual(
            deepline.call_args.args[1]["headcount"],
            ["1-10", "501-1000"],
        )

    def test_standalone_profile_includes_latest_financing_context(self) -> None:
        tools = self._tools()

        def execute(tool, _payload, **_kwargs):
            if tool == "free_simple_company_search":
                return {
                    "data": {
                        "rows": [
                            {
                                "domain": "example.com",
                                "company_name": "Example",
                                "employee_count": "11-50",
                            }
                        ]
                    }
                }
            self.assertEqual(tool, "predictleads_company_financing_events")
            return {
                "data": {
                    "data": [
                        {
                            "type": "financing_event",
                            "attributes": {
                                "financing_type": "Series B",
                                "found_at": "2026-06-23T11:00:00+02:00",
                            },
                        },
                        {
                            "type": "financing_event",
                            "attributes": {
                                "financing_type": "Series A",
                                "found_at": "2024-10-24T08:36:02+02:00",
                            },
                        },
                    ]
                }
            }

        with patch.object(tools, "_deepline", side_effect=execute) as deepline:
            profile = tools.get_company_profile({"domain": "example.com"})

        self.assertEqual(
            [
                item["attributes"]["financing_type"]
                for item in profile["latest_financing_events"][0]["data"]["items"]
            ],
            ["Series B", "Series A"],
        )
        self.assertEqual(profile["company"]["employee_count"], "11-50")
        self.assertEqual(profile["errors"], [])
        self.assertEqual(deepline.call_count, 2)
        self.assertEqual(
            deepline.call_args_list[1].args,
            (
                "predictleads_company_financing_events",
                {
                    "company_id_or_domain": "example.com",
                    "page": 1,
                    "limit": 3,
                },
            ),
        )
        self.assertEqual(
            deepline.call_args_list[1].kwargs["fallback_cost"],
            0.004,
        )

    def test_standalone_profile_surfaces_financing_failure(self) -> None:
        tools = self._tools()

        def execute(tool, _payload, **_kwargs):
            if tool == "free_simple_company_search":
                return {
                    "data": {
                        "rows": [
                            {"domain": "example.com", "company_name": "Example"}
                        ]
                    }
                }
            raise RuntimeError("provider unavailable")

        with patch.object(tools, "_deepline", side_effect=execute):
            profile = tools.get_company_profile({"domain": "example.com"})

        self.assertEqual(profile["company"]["company_name"], "Example")
        self.assertEqual(profile["latest_financing_events"], [])
        self.assertEqual(
            profile["errors"],
            [
                {
                    "source": "predictleads_company_financing_events",
                    "error": "RuntimeError: provider unavailable",
                }
            ],
        )

    def test_standalone_profile_accounts_for_both_provider_calls(self) -> None:
        tools = self._tools()
        responses = [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"data": {"rows": [{"domain": "example.com"}]}}
                ),
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"data": {"data": []}}),
                stderr="",
            ),
        ]

        with patch.object(providers.subprocess, "run", side_effect=responses):
            tools.get_company_profile({"domain": "example.com"})

        self.assertEqual(
            [call["tool"] for call in tools.stats.calls],
            [
                "free_simple_company_search",
                "predictleads_company_financing_events",
            ],
        )
        self.assertEqual(tools.stats.estimated_cost_usd, 0.004)


if __name__ == "__main__":
    unittest.main()
