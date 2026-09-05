from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

import arena_transport
from arena_transport import (
    ArenaOpenRouterTransport,
    ArenaToolClient,
    strip_arena_request_headers,
)
from harness import run_icp


def test_arena_transport_uses_credential_free_approved_routes() -> None:
    requests: list[httpx.Request] = []
    evidence_text = ("Verified event evidence. " * 15).strip()

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/hunter_discover/execute"):
            return httpx.Response(
                200,
                request=request,
                json={
                    "result": {
                        "data": {
                            "data": [
                                {
                                    "organization": "Example",
                                    "domain": "example.com",
                                }
                            ]
                        }
                    }
                },
            )
        if request.url.path.endswith("/free_simple_company_search/execute"):
            return httpx.Response(
                200,
                request=request,
                json={"result": {"data": {"rows": [{"domain": "example.com"}]}}},
            )
        if request.url.path.startswith("/api/v2/integrations/predictleads_"):
            return httpx.Response(
                200, request=request, json={"result": {"data": {"data": []}}}
            )
        if request.url.path.endswith("/exa_search/execute"):
            return httpx.Response(
                200,
                request=request,
                json={"result": {"data": {"results": []}}},
            )
        if request.url.path.endswith("/exa_contents/execute"):
            return httpx.Response(
                200,
                request=request,
                json={
                    "result": {
                        "data": {
                            "results": [
                                {
                                    "url": "https://example.com/news/event",
                                    "title": "Example launch",
                                    "text": evidence_text,
                                }
                            ]
                        }
                    }
                },
            )
        raise AssertionError(f"unexpected route: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    tools = ArenaToolClient(client=client)

    discovered = tools.search_companies({"query": "vertical SaaS", "limit": 1})
    profile = tools.get_company_profile({"domain": "example.com"})
    tools.get_company_events(
        {
            "domain": "example.com",
            "categories": ["HIRING", "FUNDING", "PRODUCT_LAUNCH"],
        }
    )
    for mode in ("search", "news", "jobs"):
        tools.search_web({"query": "Example intent", "mode": mode, "limit": 1})
    page = tools.fetch_page({"url": "https://example.com/news/event"})

    assert discovered["companies"][0]["domain"] == "example.com"
    assert profile["company"]["domain"] == "example.com"
    assert page["title"] == "Example launch"
    assert page["text"] == evidence_text
    assert [request.url.path for request in requests] == [
        "/api/v2/integrations/hunter_discover/execute",
        "/api/v2/integrations/free_simple_company_search/execute",
        "/api/v2/integrations/predictleads_company_job_openings/execute",
        "/api/v2/integrations/predictleads_company_financing_events/execute",
        "/api/v2/integrations/predictleads_company_news_events/execute",
        "/api/v2/integrations/exa_search/execute",
        "/api/v2/integrations/exa_search/execute",
        "/api/v2/integrations/exa_search/execute",
        "/api/v2/integrations/exa_contents/execute",
    ]
    assert not any("authorization" in request.headers for request in requests)
    assert not any("api_key" in request.url.params for request in requests)


def test_fetch_page_requests_fresh_content_and_preserves_successful_url() -> None:
    requests: list[httpx.Request] = []
    evidence_text = "x" * 300

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "result": {
                    "data": {
                        "results": [
                            {
                                "url": "https://news.example.com/final?id=7#details",
                                "title": "Verified launch",
                                "text": evidence_text,
                            }
                        ]
                    }
                }
            },
        )

    tools = ArenaToolClient(
        client=httpx.Client(transport=httpx.MockTransport(handle))
    )

    page = tools.fetch_page({"url": "https://example.com/original", "max_chars": 1000})

    assert page == {
        "url": "https://news.example.com/final?id=7",
        "status_code": 200,
        "title": "Verified launch",
        "text": evidence_text,
        "source": "Exa",
    }
    assert json.loads(requests[0].content)["payload"] == {
        "urls": ["https://example.com/original"],
        "text": {"maxCharacters": 1000},
        "maxAgeHours": 0,
    }


@pytest.mark.parametrize("text", ["", "x" * 299, None, {"not_text": "x" * 400}])
def test_fetch_page_rejects_empty_or_thin_text(text) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "result": {
                    "data": {
                        "results": [
                            {
                                "url": "https://example.com/news/event",
                                "text": text,
                            }
                        ]
                    }
                }
            },
        )

    tools = ArenaToolClient(
        client=httpx.Client(transport=httpx.MockTransport(handle))
    )

    with pytest.raises(RuntimeError, match="fewer than 300 text characters"):
        tools.fetch_page({"url": "https://example.com/news/event"})


def test_fetch_page_surfaces_nested_exa_status_error() -> None:
    target = "https://example.com/news/missing"

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "result": {
                    "data": {
                        "results": [],
                        "statuses": [
                            {
                                "id": target,
                                "status": "error",
                                "error": {
                                    "tag": "CRAWL_NOT_FOUND",
                                    "httpStatusCode": 404,
                                },
                            }
                        ],
                    }
                }
            },
        )

    tools = ArenaToolClient(
        client=httpx.Client(transport=httpx.MockTransport(handle))
    )

    with pytest.raises(RuntimeError, match="reported an error"):
        tools.fetch_page({"url": target})


@pytest.mark.parametrize(
    "result, message",
    [
        (None, "no result"),
        ({"text": "x" * 300}, "no valid evidence URL"),
        ({"url": "/relative", "text": "x" * 300}, "no valid evidence URL"),
        ({"url": "https://example.com/event", "text": "x" * 300,
          "error": {"httpStatusCode": 404}}, "reported an error"),
    ],
)
def test_fetch_page_rejects_unusable_result(result, message) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, request=request,
            json={"results": [] if result is None else [result]},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        tools = ArenaToolClient(client=client)
        with pytest.raises(RuntimeError, match=message):
            tools.fetch_page({"url": "https://example.com/event"})


def test_exa_projection_keeps_evidence_fields_and_caps_query(
    monkeypatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/exa_search/execute"):
            request_body = json.loads(request.content)
            query = request_body["payload"]["query"]
            if "jobs OR careers OR hiring" in query:
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "results": [
                            {
                                "title": "Cloud engineer",
                                "url": "https://jobs.example.com/apply?id=7#form",
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                request=request,
                json={
                    "results": [
                        {"title": "No evidence URL"},
                        {"title": "Relative URL", "url": "/news/item"},
                        {
                            "title": "Verified launch",
                            "url": "https://example.com/news/launch#details",
                            "publishedDate": "2026-09-01T00:00:00.000Z",
                            "highlights": ["2 days ago", "Example newsroom"],
                        },
                    ]
                },
            )
        raise AssertionError(f"unexpected route: {request.url}")

    monkeypatch.setenv("LAB_ARENA_EVALUATION_DATE", "2026-09-04")
    tools = ArenaToolClient(
        client=httpx.Client(transport=httpx.MockTransport(handle))
    )

    news = tools.search_web(
        {
            "query": "x" * 900,
            "mode": "news",
            "recency_days": 30,
            "limit": 5,
        }
    )
    jobs = tools.search_web({"query": "Example", "mode": "jobs", "limit": 5})

    assert news == {
        "results": [
            {
                "title": "Verified launch",
                "date": "2026-09-01T00:00:00.000Z",
                "snippet": "2 days ago Example newsroom",
                "source": "Exa",
                "url": "https://example.com/news/launch",
            }
        ],
        "count": 1,
        "mode": "news",
    }
    assert jobs == {
        "results": [
            {
                "title": "Cloud engineer",
                "source": "Exa",
                "url": "https://jobs.example.com/apply?id=7",
            }
        ],
        "count": 1,
        "mode": "jobs",
    }
    news_query = json.loads(requests[0].content)["payload"]["query"]
    assert len(news_query) == 500
    assert news_query.endswith(" after:2026-08-05")
    assert json.loads(requests[1].content)["payload"]["query"].endswith(
        " (jobs OR careers OR hiring)"
    )


def test_openrouter_header_filter_removes_sdk_credentials() -> None:
    request = httpx.Request(
        "POST",
        "http://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer must-not-cross",
            "X-Stainless-Runtime": "python",
            "Content-Type": "application/json",
        },
    )

    asyncio.run(strip_arena_request_headers(request))

    assert "authorization" not in request.headers
    assert "x-stainless-runtime" not in request.headers
    assert request.headers["content-type"] == "application/json"


def test_openrouter_transport_removes_sdk_only_body_fields() -> None:
    seen: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, request=request, json={})

    async def send() -> None:
        async with httpx.AsyncClient(
            transport=ArenaOpenRouterTransport(inner=httpx.MockTransport(handle))
        ) as client:
            await client.post(
                "http://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer local-placeholder"},
                json={
                    "model": "openai/gpt-5.5",
                    "stream": False,
                    "usage": {"include": True},
                    "messages": [
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning": "response-only",
                            "reasoning_details": [],
                            "tool_calls": [],
                        }
                    ],
                },
            )

    asyncio.run(send())

    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert "authorization" not in seen[0].headers
    assert set(body) == {"model", "messages"}
    assert body["messages"] == [{"role": "assistant", "tool_calls": []}]


def test_public_harness_uses_pydantic_ai_without_a_provider_key(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    async def model_response(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "generation-1",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-5.5",
                "provider": "OpenAI",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_companies",
                                        "arguments": '{"companies":[]}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    class FakeArenaTools:
        def __init__(self, timeout: float = 90.0) -> None:
            self.timeout = timeout

        def call(self, name, arguments):
            assert name == "submit_companies"
            assert arguments == {"companies": []}
            return arguments

        def close(self) -> None:
            return None

    def model_client(timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=ArenaOpenRouterTransport(
                inner=httpx.MockTransport(model_response)
            ),
        )

    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/tmp/unused-worker.sock")
    monkeypatch.setenv("BAKEOFF_OPENROUTER_MODEL", "openai/gpt-5.5")
    monkeypatch.setenv("BAKEOFF_RUN_TIMEOUT_SECONDS", "10")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch.object(arena_transport, "ArenaToolClient", FakeArenaTools):
        with patch.object(
            arena_transport, "arena_openrouter_http_client", model_client
        ):
            assert run_icp({"icp_id": "today", "intent_signal": "funding"}) == []

    assert len(seen) == 1
    assert seen[0].url == "http://openrouter.ai/api/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    body = json.loads(seen[0].content)
    assert body["max_tokens"] == 4_096
    assert body["reasoning"] == {"effort": "medium", "exclude": True}
    assert "stream" not in body
    assert "usage" not in body


def test_arena_company_limit_is_forwarded_to_the_prompt(monkeypatch) -> None:
    prompts: list[str] = []

    async def model_response(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompts.append(body["messages"][-1]["content"])
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "generation-1",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-5.5",
                "provider": "OpenAI",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_companies",
                                        "arguments": '{"companies":[]}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    class FakeArenaTools:
        def __init__(self, timeout: float = 90.0) -> None:
            self.timeout = timeout

        def call(self, name, arguments):
            return arguments

        def close(self) -> None:
            return None

    def model_client(timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=ArenaOpenRouterTransport(
                inner=httpx.MockTransport(model_response)
            ),
        )

    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/tmp/unused-worker.sock")
    monkeypatch.setenv("LAB_ARENA_COMPANY_LIMIT", "2")
    monkeypatch.setenv("BAKEOFF_OPENROUTER_MODEL", "openai/gpt-5.5")
    monkeypatch.setenv("BAKEOFF_RUN_TIMEOUT_SECONDS", "10")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch.object(arena_transport, "ArenaToolClient", FakeArenaTools):
        with patch.object(
            arena_transport, "arena_openrouter_http_client", model_client
        ):
            assert run_icp({"icp_id": "today", "intent_signal": "funding"}) == []

    assert prompts and "Return up to 2 companies." in prompts[0]


def test_arena_client_closes_when_model_transport_setup_fails(monkeypatch) -> None:
    closed: list[bool] = []

    class FakeArenaTools:
        def __init__(self, timeout: float = 90.0) -> None:
            self.timeout = timeout

        def close(self) -> None:
            closed.append(True)

    def fail_model_client(timeout: float) -> httpx.AsyncClient:
        raise RuntimeError(f"transport setup failed after {timeout:g} seconds")

    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/tmp/unused-worker.sock")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch.object(arena_transport, "ArenaToolClient", FakeArenaTools):
        with patch.object(
            arena_transport,
            "arena_openrouter_http_client",
            fail_model_client,
        ):
            with pytest.raises(RuntimeError, match="transport setup failed"):
                run_icp({"icp_id": "today", "intent_signal": "funding"})

    assert closed == [True]
