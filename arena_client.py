"""Credential-free client for the Leadpoet Arena provider socket."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx2


FRAME_SCHEMA_VERSION = "leadpoet.lab_arena.operation_frame.v1"
WORKER_SOCKET_ENV = "LAB_ARENA_WORKER_SOCKET"
MAX_FRAME_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4 * 1_048_576
ARENA_OPENROUTER_KEY = "arena-host-supplied"
_OPENROUTER_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "reasoning_effort",
        "temperature",
        "max_tokens",
        "top_p",
        "stop",
        "seed",
        "response_format",
        "include_reasoning",
    }
)
_OPERATION_TIMEOUT_SECONDS = {
    "deepline.execute": 60,
    "exa.contents": 60,
    "exa.search": 60,
    "openrouter.chat": 120,
}


class ArenaClientError(RuntimeError):
    """The Arena socket refused a call or returned an invalid response."""


@dataclass(frozen=True)
class ArenaResponse:
    status: int
    headers: dict[str, str]
    body: bytes


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(min(65_536, size - len(output)))
        if not chunk:
            raise ArenaClientError("Arena worker closed the connection")
        output.extend(chunk)
    return bytes(output)


def _socket_path(value: str | None = None) -> str:
    path = str(value or os.environ.get(WORKER_SOCKET_ENV) or "").strip()
    if not path.startswith("/") or not Path(path).name:
        raise ArenaClientError(f"{WORKER_SOCKET_ENV} is unavailable")
    return path


def dispatch(
    operation_id: str,
    parameters: Mapping[str, Any],
    *,
    timeout_seconds: float = 60.0,
    socket_path: str | None = None,
) -> ArenaResponse:
    """Send one plain operation frame. No credential crosses the socket."""

    operation_timeout = _OPERATION_TIMEOUT_SECONDS.get(str(operation_id), 60)
    timeout_ms = max(
        1,
        min(int(float(timeout_seconds) * 1_000), operation_timeout * 1_000),
    )
    document = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "operation_id": str(operation_id),
        "parameters": dict(parameters),
        "timeout_ms": timeout_ms,
    }
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArenaClientError("Arena operation parameters are not JSON") from exc
    if len(encoded) > MAX_FRAME_BYTES:
        raise ArenaClientError("Arena operation frame is too large")

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(float(timeout_seconds) + 15.0)
        connection.connect(_socket_path(socket_path))
        connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
        size = int.from_bytes(_recv_exact(connection, 4), "big")
        if size < 2 or size > MAX_RESPONSE_BYTES:
            raise ArenaClientError("Arena worker returned an invalid frame")
        raw = _recv_exact(connection, size)
    except OSError as exc:
        raise ArenaClientError("Arena worker is unavailable") from exc
    finally:
        connection.close()

    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArenaClientError("Arena worker returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise ArenaClientError("Arena worker returned an invalid response")
    if set(response) == {"error"}:
        code = str(response["error"] or "provider_error")[:64]
        raise ArenaClientError(f"Arena provider call failed: {code}")
    if set(response) != {"status", "headers", "body_b64"}:
        raise ArenaClientError("Arena worker returned an invalid response")
    status = response["status"]
    headers = response["headers"]
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or not 100 <= status <= 599
    ):
        raise ArenaClientError("Arena worker returned an invalid status")
    if not isinstance(headers, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in headers.items()
    ):
        raise ArenaClientError("Arena worker returned invalid headers")
    try:
        body = base64.b64decode(response["body_b64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ArenaClientError("Arena worker returned an invalid body") from exc
    return ArenaResponse(status=status, headers=dict(headers), body=body)


def _json_call(
    operation_id: str,
    parameters: Mapping[str, Any],
    *,
    timeout_seconds: float = 60.0,
    socket_path: str | None = None,
) -> Any:
    response = dispatch(
        operation_id,
        parameters,
        timeout_seconds=timeout_seconds,
        socket_path=socket_path,
    )
    if not 200 <= response.status < 300:
        raise ArenaClientError(f"Arena provider call returned HTTP {response.status}")
    try:
        return json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArenaClientError("Arena provider returned invalid JSON") from exc


def _public_domain(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("a public company domain is required") from exc
    host = (parsed.hostname or "").rstrip(".").removeprefix("www.")
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or "." not in host
        or parsed.username
        or parsed.password
        or (port is not None and port not in {80, 443})
        or len(host) > 253
        or host == "localhost"
        or host.endswith((".internal", ".local", ".localhost", ".onion"))
    ):
        raise ValueError("a public company domain is required")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("a public company domain is required")
    return host


def _evaluation_day() -> date:
    raw = str(
        os.environ.get("BAKEOFF_EVALUATION_DATE")
        or os.environ.get("LAB_ARENA_EVALUATION_DATE")
        or ""
    ).strip()
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.now(timezone.utc).date()


def _exa_rows(payload: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("results")
    if not isinstance(rows, list):
        rows = (
            (payload.get("data") or {}).get("results")
            if isinstance(payload.get("data"), dict)
            else []
        )
    return [dict(row) for row in rows[:limit] if isinstance(row, dict)]


def _search_projection(payload: Any, limit: int) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in _exa_rows(payload, limit):
        url = str(row.get("url") or row.get("id") or "").split("#", 1)[0]
        if not url:
            continue
        highlights = row.get("highlights")
        snippet = " ".join(
            str(item).strip()
            for item in (highlights if isinstance(highlights, list) else [])
            if str(item).strip()
        )
        projected.append(
            {
                "title": str(row.get("title") or "")[:500],
                "url": url,
                "date": str(
                    row.get("publishedDate") or row.get("published_date") or ""
                )[:100],
                "snippet": (snippet or str(row.get("text") or ""))[:1_000],
                "source": "Exa",
            }
        )
    return projected


class ArenaToolClient:
    """Implement the harness tools with organizer-approved Arena operations."""

    def __init__(self, *, socket_path: str | None = None, timeout: float = 60.0):
        self.socket_path = _socket_path(socket_path)
        self.timeout = max(1.0, min(float(timeout), 120.0))

    def _call(self, operation_id: str, parameters: Mapping[str, Any]) -> Any:
        return _json_call(
            operation_id,
            parameters,
            timeout_seconds=self.timeout,
            socket_path=self.socket_path,
        )

    def search_companies(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(int(arguments.get("limit") or 5), 6))
        context = ". ".join(
            part
            for part in (
                query,
                (
                    f"Industry: {str(arguments.get('industry') or '').strip()}"
                    if arguments.get("industry")
                    else ""
                ),
                (
                    f"Headquarters: {str(arguments.get('geography') or '').strip()}"
                    if arguments.get("geography")
                    else ""
                ),
                (
                    "Employee count: "
                    + ", ".join(
                        str(item) for item in arguments.get("employee_count") or []
                    )
                    if arguments.get("employee_count")
                    else ""
                ),
            )
            if part
        )
        raw = self._call(
            "exa.search",
            {
                "query": context[:2_000],
                "type": "auto",
                "category": "company",
                "numResults": limit,
                "contents": {"highlights": True},
            },
        )
        companies: list[dict[str, Any]] = []
        for row in _exa_rows(raw, limit):
            url = str(row.get("url") or row.get("id") or "").split("#", 1)[0]
            try:
                domain = _public_domain(url)
            except ValueError:
                continue
            companies.append(
                {
                    "company_name": str(row.get("title") or domain)[:300],
                    "domain": domain,
                    "company_website": f"https://{domain}/",
                    "summary": str(row.get("text") or "")[:1_000],
                }
            )
        return {"companies": companies, "count": len(companies)}

    def get_company_profile(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        domain = _public_domain(arguments.get("domain"))
        quoted = "'" + domain.replace("'", "''") + "'"
        columns = (
            "normalized_domain, domain, company_name, industry, location, "
            "linkedin_url, employee_count, year_founded, updated_at"
        )
        raw = self._call(
            "deepline.execute",
            {
                "tool": "free_simple_company_search",
                "payload": {
                    "sql": f"SELECT {columns} FROM companies WHERE normalized_domain = {quoted} LIMIT 3"
                },
            },
        )
        return {"domain": domain, "company": raw}

    def get_company_events(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        domain = _public_domain(arguments.get("domain"))
        categories = arguments.get("categories") or ["NEWS", "HIRING", "FUNDING"]
        if not isinstance(categories, list):
            categories = [categories]
        limit = max(1, min(int(arguments.get("limit") or 5), 5))
        terms = " OR ".join(
            str(item).strip() for item in categories if str(item).strip()
        )
        extra = str(arguments.get("query") or "").strip()
        query = f"site:{domain} ({terms}) {extra}".strip()
        start = (_evaluation_day() - timedelta(days=365)).isoformat() + "T00:00:00Z"
        raw = self._call(
            "exa.search",
            {
                "query": query[:2_000],
                "type": "auto",
                "category": "news",
                "numResults": limit,
                "startPublishedDate": start,
                "contents": {"highlights": True},
            },
        )
        return {
            "domain": domain,
            "events": _search_projection(raw, limit),
            "errors": [],
        }

    def search_web(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        mode = str(arguments.get("mode") or "search").strip().lower()
        if mode not in {"search", "news", "jobs"}:
            raise ValueError("mode must be search, news, or jobs")
        limit = max(1, min(int(arguments.get("limit") or 5), 5))
        parameters: dict[str, Any] = {
            "query": (
                f"{query} (jobs OR careers OR hiring)" if mode == "jobs" else query
            )[:2_000],
            "type": "auto",
            "numResults": limit,
            "contents": {"highlights": True},
        }
        if mode == "news":
            parameters["category"] = "news"
        recency = arguments.get("recency_days")
        if recency not in (None, ""):
            days = max(1, min(int(recency), 3_650))
            parameters["startPublishedDate"] = (
                _evaluation_day() - timedelta(days=days)
            ).isoformat() + "T00:00:00Z"
        raw = self._call("exa.search", parameters)
        rows = _search_projection(raw, limit)
        return {"results": rows, "count": len(rows), "mode": mode}

    def fetch_page(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        url = str(arguments.get("url") or "").strip()
        _public_domain(url)
        max_chars = max(1_000, min(int(arguments.get("max_chars") or 2_500), 4_000))
        raw = self._call(
            "exa.contents",
            {
                "urls": [url],
                "text": {"maxCharacters": max_chars},
                "livecrawl": "preferred",
            },
        )
        rows = _exa_rows(raw, 1)
        row = rows[0] if rows else {}
        return {
            "url": str(row.get("url") or url),
            "status_code": 200,
            "title": str(row.get("title") or "")[:500],
            "text": str(row.get("text") or "")[:max_chars],
            "source": "Exa",
        }

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name == "submit_companies":
            return {"companies": list(arguments.get("companies") or [])}
        method = getattr(self, name, None)
        if (
            name.startswith("_")
            or method is None
            or name
            not in {
                "search_companies",
                "get_company_profile",
                "get_company_events",
                "search_web",
                "fetch_page",
            }
        ):
            raise ValueError(f"unknown tool: {name}")
        return method(arguments)


def _openrouter_parameters(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArenaClientError("OpenRouter request is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ArenaClientError("OpenRouter request must be a JSON object")
    stream = payload.pop("stream", False)
    payload.pop("stream_options", None)
    payload.pop("usage", None)
    if stream not in (False, None):
        raise ArenaClientError("streaming is not supported by the Arena operation")
    unknown = sorted(set(payload) - _OPENROUTER_FIELDS)
    if unknown:
        raise ArenaClientError(
            "OpenRouter request uses unsupported fields: " + ", ".join(unknown)
        )
    return payload


class ArenaOpenRouterTransport(httpx2.AsyncBaseTransport):
    """Route PydanticAI's OpenRouter request through one Arena operation."""

    def __init__(self, *, socket_path: str | None = None) -> None:
        self.socket_path = _socket_path(socket_path)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if (
            request.method != "POST"
            or request.url.host != "openrouter.ai"
            or request.url.path != "/api/v1/chat/completions"
        ):
            raise ArenaClientError("only OpenRouter chat completions are supported")
        body = await request.aread()
        response = await asyncio.to_thread(
            dispatch,
            "openrouter.chat",
            _openrouter_parameters(body),
            timeout_seconds=120.0,
            socket_path=self.socket_path,
        )
        return httpx2.Response(
            response.status,
            headers=response.headers,
            content=response.body,
            request=request,
        )


def openrouter_http_client(*, socket_path: str | None = None) -> httpx2.AsyncClient:
    """Return the HTTP client accepted by PydanticAI's OpenRouter provider."""

    return httpx2.AsyncClient(
        transport=ArenaOpenRouterTransport(socket_path=socket_path),
        timeout=125.0,
    )


__all__ = [
    "ARENA_OPENROUTER_KEY",
    "ArenaClientError",
    "ArenaOpenRouterTransport",
    "ArenaResponse",
    "ArenaToolClient",
    "dispatch",
    "openrouter_http_client",
]
