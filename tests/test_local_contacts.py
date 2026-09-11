from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from unittest.mock import Mock

import pytest

from experiments.harness_bakeoff import runner
from experiments.harness_bakeoff.contacts import _profiles, _unwrap, enrich_contacts
from experiments.harness_bakeoff.providers import (
    LiveProviderTools,
    _bounded_tool_result,
    _json_safe,
    verify_live_smoke_company,
)
from experiments.harness_bakeoff.tool_client import ToolClient
from experiments.harness_bakeoff.tool_server import ToolServer
from experiments.harness_bakeoff.worker import SENTINEL


def _company() -> dict:
    return {
        "company_name": "Acme",
        "company_website": "https://acme.example/",
        "company_linkedin": "https://www.linkedin.com/company/acme/",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "California",
        "fit_summary": "Acme matches the requested company profile.",
        "fit_evidence_urls": ["https://acme.example/about"],
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "Acme announced a new product.",
                "date": "2026-09-01",
                "why_now": "The launch creates a current sales opportunity.",
                "url": "https://acme.example/news/launch",
                "snippet": "Acme announced the product launch.",
            }
        ],
    }


def _contact() -> dict:
    return {
        "full_name": "Ada Lovelace",
        "role": "VP Sales",
        "linkedin_url": "https://www.linkedin.com/in/ada-lovelace/",
        "location": {
            "country": "US",
            "region": "California",
            "city": "San Francisco",
        },
        "email": "ada@acme.example",
        "email_source": {
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "record_id": "profile-1",
        },
    }


def _icp(*, contact_policy: str | None = "contacts_v1") -> dict:
    value = {
        "icp_id": "icp-local-contact",
        "target_roles": ["Vice President of Sales"],
        "target_seniority": "VP+",
        "contact_geography": {
            "countries": ["United States"],
            "regions": ["California"],
            "cities": ["San Francisco"],
        },
    }
    if contact_policy is not None:
        value["contact_policy"] = contact_policy
    return value


def _tools(*, allow_contacts: bool, max_cost: float = 4.0) -> LiveProviderTools:
    return LiveProviderTools(
        deepline_api_key="test-deepline-key",
        scrapingdog_api_key="test-scrapingdog-key",
        deepline_bin="test-deepline",
        max_provider_cost_usd=max_cost,
        allow_contacts=allow_contacts,
    )


def _completed(payload: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["test-deepline"],
        returncode=0,
        stdout=json.dumps(payload),
        stderr="",
    )


def test_contact_mode_dispatches_live_tools_and_retains_terminal_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = {
        "billing": {"credits_charged": 0.7, "cost_usd": 0.07},
        "result": {
            "data": {
                "elements": [
                    {
                        "id": "profile-1",
                        "linkedinUrl": "https://www.linkedin.com/in/ada-lovelace/",
                        "currentPositions": [
                            {
                                "title": "VP Sales",
                                "companyName": "Acme",
                                "companyDomain": "acme.example",
                                "companyLinkedinUrl": (
                                    "https://www.linkedin.com/company/acme/"
                                ),
                                "isCurrent": True,
                            }
                        ],
                    }
                ]
            }
        },
    }
    profile = {
        "billing": {"credits_charged": 0.14, "cost_usd": 0.014},
        "result": {
            "data": {
                "element": {
                    "about": "Public profile detail. " * 5_000,
                    "id": "profile-1",
                    "linkedinUrl": "https://www.linkedin.com/in/ada-lovelace/",
                    "firstName": "Ada",
                    "lastName": "Lovelace",
                    "location": {
                        "countryCode": "US",
                        "parsed": {
                            "countryFull": "United States",
                            "state": "California",
                            "city": "San Francisco",
                        },
                    },
                    "workEmail": "ada@acme.example",
                    "currentPosition": [
                        {
                            "title": "VP Sales",
                            "companyName": "Acme",
                            "companyDomain": "acme.example",
                            "companyLinkedinUrl": (
                                "https://www.linkedin.com/company/acme/"
                            ),
                            "isCurrent": True,
                        }
                    ],
                }
            }
        },
    }
    verbose_row = {
        f"public_field_{index}": (f"Public presentation field {index}. " * 40)
        for index in range(20)
    }
    profile["presentation"] = {"rows": [deepcopy(verbose_row) for _ in range(5)]}
    profile["output"] = {"rows": [deepcopy(verbose_row) for _ in range(5)]}
    run = Mock(side_effect=[_completed(search), _completed(profile)])
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers.subprocess.run", run
    )
    tools = _tools(allow_contacts=True)

    assert len(json.dumps(profile).encode("utf-8")) < 1_000_000
    generic = _bounded_tool_result(_json_safe(profile))
    assert generic.get("response_truncated") is True
    assert "json_preview" in generic
    assert _profiles(_unwrap(generic)) == []

    with ToolServer(tools) as server:
        client = ToolClient(server.url, server.token)
        companies = enrich_contacts(_icp(), [_company()], client.call)
        submitted = client.submit_companies(companies)

    assert submitted["companies"][0]["contact"] == _contact()
    assert [call.args[0][3] for call in run.call_args_list] == [
        "harvestapi_search_leads",
        "harvestapi_get_profile",
    ]
    assert [row["estimated_cost_usd"] for row in tools.stats.calls] == [0.07, 0.014]
    assert tools.stats.estimated_cost_usd == pytest.approx(0.084)


def test_opt_out_rejects_contact_tools_and_contact_terminal_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock()
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers.subprocess.run", run
    )
    tools = _tools(allow_contacts=False)
    company_with_contact = {**_company(), "contact": _contact()}

    with ToolServer(tools) as server:
        client = ToolClient(server.url, server.token)
        with pytest.raises(RuntimeError, match="unknown tool"):
            client.call(
                "harvestapi_search_leads",
                {
                    "currentJobTitles": "Vice President of Sales",
                    "page": 1,
                    "currentCompanies": (
                        "https://www.linkedin.com/company/acme/"
                    ),
                },
            )
        with pytest.raises(RuntimeError, match="contact"):
            client.submit_companies([company_with_contact])
        submitted = client.submit_companies([_company()])

    assert submitted["companies"][0]["company_name"] == "Acme"
    assert "contact" not in submitted["companies"][0]
    run.assert_not_called()


def test_contact_provider_failure_preserves_company_without_invented_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=["test-deepline"], returncode=1, stdout="", stderr="provider failed"
        )
    )
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers.subprocess.run", run
    )
    tools = _tools(allow_contacts=True)

    with ToolServer(tools) as server:
        companies = enrich_contacts(
            _icp(), [_company()], ToolClient(server.url, server.token).call
        )

    assert companies == [_company()]
    assert tools.stats.calls[0]["tool"] == "harvestapi_search_leads"
    assert tools.stats.calls[0]["status"] == "error"
    assert tools.stats.calls[0]["estimated_cost_usd"] == 0.07


def test_contact_fallback_cost_exhausts_budget_before_another_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(return_value=_completed({"result": {"data": {"elements": []}}}))
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers.subprocess.run", run
    )
    tools = _tools(allow_contacts=True, max_cost=0.07)

    with ToolServer(tools) as server:
        client = ToolClient(server.url, server.token)
        client.call(
            "harvestapi_search_leads",
            {
                "currentJobTitles": "Vice President of Sales",
                "page": 1,
                "currentCompanies": "https://www.linkedin.com/company/acme/",
            },
        )
        with pytest.raises(RuntimeError, match="provider cost limit exhausted"):
            client.call(
                "harvestapi_get_profile",
                {
                    "url": "https://www.linkedin.com/in/ada-lovelace/",
                    "findEmail": "true",
                },
            )

    assert run.call_count == 1
    assert tools.stats.estimated_cost_usd == pytest.approx(0.07)


def test_dynamic_profile_without_billing_charges_full_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(return_value=_completed({"result": {"data": {"element": {}}}}))
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers.subprocess.run", run
    )
    tools = _tools(allow_contacts=True, max_cost=0.5)

    with ToolServer(tools) as server:
        ToolClient(server.url, server.token).call(
            "harvestapi_get_profile",
            {
                "url": "https://www.linkedin.com/in/ada-lovelace/",
                "findEmail": "true",
            },
        )

    assert tools.stats.calls[0]["estimated_cost_usd"] == 0.5
    assert tools.stats.estimated_cost_usd == pytest.approx(0.5)


def test_oversized_contact_response_fails_after_recording_measured_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "billing": {"credits_charged": 0.14, "cost_usd": 0.014},
        "result": {
            "data": {
                "elements": [
                    {
                        "linkedinUrl": (
                            f"https://www.linkedin.com/in/profile-{index}/"
                        ),
                        "about": "x" * 20_000,
                    }
                    for index in range(60)
                ]
            }
        },
    }
    run = Mock(return_value=_completed(response))
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers.subprocess.run", run
    )
    tools = _tools(allow_contacts=True)

    with ToolServer(tools) as server:
        with pytest.raises(RuntimeError, match="response exceeds size limit"):
            ToolClient(server.url, server.token).call(
                "harvestapi_get_profile",
                {
                    "url": "https://www.linkedin.com/in/ada-lovelace/",
                    "findEmail": "true",
                },
            )

    assert tools.stats.calls == [
        {
            "tool": "harvestapi_get_profile",
            "provider": "deepline",
            "status": "ok",
            "latency_ms": tools.stats.calls[0]["latency_ms"],
            "estimated_cost_usd": 0.014,
        }
    ]
    assert tools.stats.estimated_cost_usd == pytest.approx(0.014)


@pytest.mark.parametrize(
    "url",
    (
        "https://user@www.linkedin.com/in/ada-lovelace/",
        "https://www.linkedin.com:444/in/ada-lovelace/",
    ),
)
def test_profile_lookup_rejects_credentials_and_nonstandard_ports(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    run = Mock()
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers.subprocess.run", run
    )
    tools = _tools(allow_contacts=True)

    with ToolServer(tools) as server:
        with pytest.raises(RuntimeError, match="url is invalid"):
            ToolClient(server.url, server.token).call(
                "harvestapi_get_profile", {"url": url, "findEmail": "true"}
            )

    run.assert_not_called()


def test_run_attempt_threads_contact_policy_and_retains_worker_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool] = []
    original = runner.LiveProviderTools

    def recording_providers(**kwargs: object) -> LiveProviderTools:
        observed.append(bool(kwargs.get("allow_contacts")))
        return original(**kwargs)  # type: ignore[arg-type]

    company = {**_company(), "contact": _contact()}
    worker = {
        "ok": True,
        "companies": [company],
        "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
    }
    monkeypatch.setattr(runner, "LiveProviderTools", recording_providers)
    monkeypatch.setattr(
        runner,
        "_run_worker_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["worker"],
            returncode=0,
            stdout=SENTINEL + json.dumps(worker),
            stderr="",
        ),
    )

    result = runner.run_attempt(
        arm="pydantic_ai",
        icp=_icp(),
        provider_secrets={
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "DEEPLINE_API_KEY": "test-deepline-key",
            "SCRAPINGDOG_API_KEY": "test-scrapingdog-key",
        },
        model="test/model",
        model_pricing={},
        max_companies=5,
        python="python",
        deepline_bin="test-deepline",
        evaluation_date="2026-09-11",
    )

    assert observed == [True]
    assert result["ok"] is True
    assert result["companies"][0]["contact"] == _contact()


def test_smoke_checker_accepts_valid_contact_and_still_checks_company_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = Mock(
        side_effect=lambda url, **kwargs: {
            "url": url,
            "status_code": 200,
            "body_present": kwargs["require_body"],
        }
    )
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers._probe_live_public_url", probe
    )

    result = verify_live_smoke_company({**_company(), "contact": _contact()})

    assert result["company_website"]["url"] == "https://acme.example/"
    assert result["intent_evidence"]["url"].endswith("/news/launch")
    assert probe.call_count == 2


def test_smoke_checker_rejects_malformed_contact_before_url_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = Mock()
    monkeypatch.setattr(
        "experiments.harness_bakeoff.providers._probe_live_public_url", probe
    )
    malformed = deepcopy(_contact())
    malformed["email"] = "not-an-email"

    with pytest.raises(ValueError):
        verify_live_smoke_company({**_company(), "contact": malformed})

    probe.assert_not_called()
