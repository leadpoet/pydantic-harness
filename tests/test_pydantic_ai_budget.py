from __future__ import annotations

import json
from types import SimpleNamespace

from pydantic_ai import messages

from experiments.harness_bakeoff.adapters import pydantic_ai


def _context(*, input_tokens: int = 0, requests: int = 0, tool_calls: int = 0):
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            requests=requests,
            tool_calls=tool_calls,
        )
    )


def _large_result(company: str, suffix: str) -> dict:
    tracking = "tracking-segment/" * 12
    return {
        "results": [
            {
                "company_name": f"{company} {index}",
                "domain": f"{company.lower()}{index}.example",
                "date": f"2026-08-{index + 10:02d}",
                "url": (
                    f"https://news.example/{company.lower()}/{index}/{tracking}{suffix}"
                ),
                "quote": f"{company} launched a verified product. " + (suffix * 300),
                "irrelevant_blob": suffix * 900,
            }
            for index in range(5)
        ]
    }


def _history() -> list[messages.ModelMessage]:
    return [
        messages.ModelRequest.user_text_prompt("Find matching companies"),
        messages.ModelResponse(
            parts=[
                messages.ToolCallPart(
                    "search_web", {"query": "Acme launch"}, tool_call_id="call-1"
                )
            ]
        ),
        messages.ModelRequest(
            parts=[
                messages.ToolReturnPart(
                    "search_web",
                    _large_result("Acme", "a"),
                    tool_call_id="call-1",
                )
            ]
        ),
        messages.ModelResponse(
            parts=[
                messages.ToolCallPart(
                    "search_web", {"query": "Beta launch"}, tool_call_id="call-2"
                )
            ]
        ),
        messages.ModelRequest(
            parts=[
                messages.ToolReturnPart(
                    "search_web",
                    _large_result("Beta", "b"),
                    tool_call_id="call-2",
                )
            ]
        ),
    ]


def test_prior_tool_payload_is_bounded_but_latest_remains_full() -> None:
    original = _history()
    processed = pydantic_ai._process_history(_context(), original)

    old_return = processed[2].parts[0]
    latest_return = processed[4].parts[0]
    assert isinstance(old_return, messages.ToolReturnPart)
    assert isinstance(latest_return, messages.ToolReturnPart)
    assert len(pydantic_ai._json_bytes(old_return.content)) <= 1_200
    compact_json = json.dumps(old_return.content)
    assert "2026-08-10" in compact_json
    expected_url = original[2].parts[0].content["results"][0]["url"]
    assert expected_url in compact_json
    assert "Acme launched a verified product" in compact_json
    assert latest_return.content == original[4].parts[0].content


def test_prior_fetch_page_keeps_full_quote_and_url() -> None:
    exact_url = "https://example.com/news/verified-launch?source=company"
    exact_quote = "The company launched its verified platform on August 20, 2026."
    page = {
        "url": exact_url,
        "text": ("Relevant page context. " * 180) + exact_quote,
    }
    history = [
        messages.ModelRequest.user_text_prompt("Verify the launch"),
        messages.ModelResponse(
            parts=[
                messages.ToolCallPart(
                    "fetch_page", {"url": exact_url}, tool_call_id="fetch-1"
                )
            ]
        ),
        messages.ModelRequest(
            parts=[
                messages.ToolReturnPart(
                    "fetch_page", page, tool_call_id="fetch-1"
                )
            ]
        ),
        messages.ModelResponse(
            parts=[
                messages.ToolCallPart(
                    "search_web", {"query": "another company"}, tool_call_id="call-2"
                )
            ]
        ),
        messages.ModelRequest(
            parts=[
                messages.ToolReturnPart(
                    "search_web",
                    _large_result("Beta", "b"),
                    tool_call_id="call-2",
                )
            ]
        ),
    ]

    processed = pydantic_ai._process_history(_context(), history)
    fetch_return = processed[2].parts[0]

    assert isinstance(fetch_return, messages.ToolReturnPart)
    assert fetch_return.content == page
    assert fetch_return.content["url"] == exact_url
    assert exact_quote in fetch_return.content["text"]


def test_budget_reserve_warns_once_and_hides_only_research_tools() -> None:
    context = _context(input_tokens=82_000, requests=12, tool_calls=12)
    processed = pydantic_ai._process_history(context, _history())
    processed_again = pydantic_ai._process_history(context, processed)

    warnings = [
        part.content
        for message in processed_again
        if isinstance(message, messages.ModelRequest)
        for part in message.parts
        if isinstance(part, messages.UserPromptPart)
        and isinstance(part.content, str)
        and pydantic_ai._FINALIZE_MARKER in part.content
    ]
    assert len(warnings) == 1
    assert "do not invent" in warnings[0].lower()
    tool_definitions = [SimpleNamespace(name="search_web")]
    assert pydantic_ai._prepare_research_tools(context, tool_definitions) == []

    below_reserve = _context(input_tokens=81_999, requests=21, tool_calls=23)
    assert (
        pydantic_ai._prepare_research_tools(below_reserve, tool_definitions)
        == tool_definitions
    )
