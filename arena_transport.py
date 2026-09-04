"""Credential-free provider transport for the Leadpoet agent Arena.

The Arena exposes one HTTP bridge over a Unix socket.  Requests keep the
provider host and path, but carry no provider credential.  The Arena host adds
credentials and enforces its own call, cost, token, and time limits.
"""

from __future__ import annotations

from datetime import date, timedelta
from html.parser import HTMLParser
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from experiments.harness_bakeoff.models import _public_http_url, validate_companies


_ALLOWED_ARENA_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "accept-language",
        "cache-control",
        "connection",
        "content-length",
        "content-type",
        "date",
        "expect",
        "host",
        "http-referer",
        "keep-alive",
        "pragma",
        "te",
        "user-agent",
        "x-title",
    }
)
_ALLOWED_OPENROUTER_FIELDS = frozenset(
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
_ALLOWED_MESSAGE_FIELDS = frozenset(
    {"role", "content", "name", "tool_call_id", "tool_calls"}
)
_EVENT_TOOLS = {
    "HIRING": "predictleads_company_job_openings",
    "JOBS": "predictleads_company_job_openings",
    "FUNDING": "predictleads_company_financing_events",
    "FINANCING": "predictleads_company_financing_events",
    "PRODUCT_LAUNCH": "predictleads_company_news_events",
    "ACQUISITION": "predictleads_company_news_events",
    "PARTNERSHIP": "predictleads_company_news_events",
    "MARKET_EXPANSION": "predictleads_company_news_events",
    "LEADERSHIP_CHANGE": "predictleads_company_news_events",
    "FACILITY_OPENING": "predictleads_company_news_events",
    "NEWS": "predictleads_company_news_events",
}
_NEWS_CATEGORIES = {
    "PRODUCT_LAUNCH": ["launches"],
    "ACQUISITION": ["acquires", "merges_with", "sells_assets_to"],
    "PARTNERSHIP": ["partners_with"],
    "MARKET_EXPANSION": ["expands_offices_in", "expands_offices_to"],
    "LEADERSHIP_CHANGE": ["hires", "promotes"],
    "FACILITY_OPENING": [
        "expands_facilities",
        "expands_offices_in",
        "expands_offices_to",
        "opens_new_location",
    ],
}
_US_REGIONS = {
    "west coast": ("CA", "OR", "WA"),
    "northeast": ("CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"),
    "midwest": ("IA", "IL", "IN", "KS", "MI", "MN", "MO", "ND", "NE", "OH", "SD", "WI"),
}


def arena_socket_path() -> str:
    """Return the absolute worker socket supplied by the Arena."""

    value = str(os.environ.get("LAB_ARENA_WORKER_SOCKET") or "").strip()
    if not value.startswith("/"):
        raise RuntimeError("LAB_ARENA_WORKER_SOCKET is required")
    return value


async def strip_arena_request_headers(request: httpx.Request) -> None:
    """Remove SDK and credential headers before a request reaches the broker."""

    for name in list(request.headers):
        if name.lower() not in _ALLOWED_ARENA_HEADERS:
            del request.headers[name]


class ArenaOpenRouterTransport(httpx.AsyncBaseTransport):
    """Send OpenAI SDK requests over the Arena socket in its closed schema."""

    def __init__(
        self,
        socket_path: str | None = None,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport(
            uds=socket_path or arena_socket_path()
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("OpenRouter request body is invalid") from exc
        if not isinstance(body, dict):
            raise RuntimeError("OpenRouter request body must be an object")
        # Keep only the Arena's published OpenRouter request fields. The Arena
        # pins streaming off and owns provider usage accounting.
        body = {
            name: value
            for name, value in body.items()
            if name in _ALLOWED_OPENROUTER_FIELDS
        }
        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                for name in list(message):
                    if name not in _ALLOWED_MESSAGE_FIELDS:
                        del message[name]
                for name in ("content", "name", "tool_call_id", "tool_calls"):
                    if message.get(name) is None:
                        message.pop(name, None)
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() in _ALLOWED_ARENA_HEADERS
            and name.lower() not in {"content-length", "transfer-encoding"}
        }
        forwarded = httpx.Request(
            request.method,
            request.url,
            headers=headers,
            content=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        )
        return await self._inner.handle_async_request(forwarded)

    async def aclose(self) -> None:
        await self._inner.aclose()


def arena_openrouter_http_client(timeout: float) -> httpx.AsyncClient:
    """Build the HTTP client used by PydanticAI inside the Arena sandbox."""

    return httpx.AsyncClient(
        transport=ArenaOpenRouterTransport(),
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,
        trust_env=False,
    )


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._title_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in {"script", "style"}:
            self._ignored_depth += 1
        elif normalized == "title" and self._ignored_depth == 0:
            self._title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif normalized == "title" and self._title_depth:
            self._title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = " ".join(str(data).split())
        if text and self._title_depth:
            self.title_parts.append(text)
        elif text:
            self.parts.append(text)


def _page_content(value: str, limit: int) -> tuple[str, str]:
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        without_hidden = re.sub(
            r"<(script|style)\b[^>]*>.*?</\1\s*>",
            " ",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        title_match = re.search(
            r"<title\b[^>]*>(.*?)</title\s*>",
            without_hidden,
            flags=re.IGNORECASE | re.DOTALL,
        )
        title = (
            " ".join(re.sub(r"<[^>]+>", " ", title_match.group(1)).split())[:500]
            if title_match
            else ""
        )
        without_title = re.sub(
            r"<title\b[^>]*>.*?</title\s*>",
            " ",
            without_hidden,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = " ".join(re.sub(r"<[^>]+>", " ", without_title).split())[:limit]
        return title, text
    return " ".join(parser.title_parts)[:500], " ".join(parser.parts)[:limit]


def _evidence_url(value: Any) -> str:
    try:
        return _public_http_url(str(value or ""))
    except ValueError:
        return ""


def _domain(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if "://" not in raw:
        raw = "https://" + raw
    host = (urlsplit(raw).hostname or "").rstrip(".").removeprefix("www.")
    if not host or "." not in host or len(host) > 253:
        return ""
    return host


def _sql_literal(value: str) -> str:
    clean = re.sub(r"[\x00-\x1f\x7f]", " ", value)[:253]
    return "'" + clean.replace("'", "''") + "'"


def _result_data(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    tool_response = payload.get("toolResponse")
    if isinstance(tool_response, dict):
        for key in ("rawV2", "raw", "data"):
            value = tool_response.get(key)
            if isinstance(value, dict):
                return value
    result = payload.get("result")
    if isinstance(result, dict):
        data = result.get("data")
        return data if isinstance(data, dict) else result
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:20_000]
    if isinstance(value, list):
        return [_json_safe(item, depth=depth + 1) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key)[:200]: _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:150]
            if str(key).lower() not in {"api_key", "apikey", "authorization", "token"}
        }
    return str(value)[:2_000]


def _hunter_locations(value: str) -> list[dict[str, str]]:
    normalized = " ".join(str(value or "").lower().replace(",", " ").split())
    for label, states in _US_REGIONS.items():
        if label in normalized:
            return [{"country": "US", "state": state} for state in states]
    if "london" in normalized:
        return [{"country": "GB", "city": "London"}]
    if "united kingdom" in normalized or normalized in {"uk", "great britain"}:
        return [{"country": "GB"}]
    if "united states" in normalized or normalized in {"us", "usa"}:
        return [{"country": "US"}]
    return []


def _project_event_data(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    included: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in payload.get("included") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or "")
        identifier = str(raw.get("id") or "")
        attrs = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
        allowed = (
            ("company_name", "domain", "ticker")
            if kind == "company"
            else ("title", "url", "published_at", "author")
        )
        included[(kind, identifier)] = {
            key: _json_safe(attrs.get(key))
            for key in allowed
            if attrs.get(key) not in (None, "", [])
        }

    attribute_names = {
        "amount",
        "amount_normalized",
        "article_sentence",
        "categories",
        "category",
        "confidence",
        "contract_types",
        "effective_date",
        "event",
        "financing_type",
        "financing_type_normalized",
        "first_seen_at",
        "found_at",
        "headcount",
        "job_title",
        "last_seen_at",
        "location",
        "normalized_title",
        "planning",
        "posted_at",
        "product",
        "recognition",
        "salary",
        "seniority",
        "status",
        "summary",
        "title",
        "url",
        "vulnerability",
    }
    items: list[dict[str, Any]] = []
    raw_items = payload.get("data")
    if not isinstance(raw_items, list):
        raw_items = []
    for raw in raw_items[:limit]:
        if not isinstance(raw, dict):
            continue
        attrs = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
        item: dict[str, Any] = {
            "type": str(raw.get("type") or "event"),
            "attributes": {
                key: _json_safe(value)
                for key, value in attrs.items()
                if key in attribute_names and value not in (None, "", [], {})
            },
        }
        relations = raw.get("relationships") if isinstance(raw.get("relationships"), dict) else {}
        related: dict[str, Any] = {}
        for name, relation in relations.items():
            data = relation.get("data") if isinstance(relation, dict) else None
            if not isinstance(data, dict):
                continue
            key = (str(data.get("type") or ""), str(data.get("id") or ""))
            if key in included:
                related[str(name)] = included[key]
        if related:
            item["related"] = related
        items.append(item)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return {
        "items": items,
        "returned_count": len(items),
        "available_count": meta.get("count"),
    }


class ArenaToolClient:
    """Implement the public semantic tools through approved Arena operations."""

    def __init__(self, timeout: float = 90.0, client: httpx.Client | None = None):
        self.timeout = max(1.0, min(float(timeout), 120.0))
        self._owns_client = client is None
        self._client = client or httpx.Client(
            transport=httpx.HTTPTransport(uds=arena_socket_path()),
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _json_request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(
            method,
            url,
            params=params,
            json=body,
            timeout=self.timeout,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Arena provider returned HTTP {response.status_code} with invalid JSON"
            ) from exc
        if not response.is_success:
            code = (
                (payload.get("error") or {}).get("code")
                if isinstance(payload.get("error"), dict)
                else ""
            )
            raise RuntimeError(str(code or f"Arena provider returned HTTP {response.status_code}"))
        if not isinstance(payload, dict):
            raise RuntimeError("Arena provider returned a non-object")
        return payload

    def _deepline(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"http://code.deepline.com/api/v2/integrations/{tool}/execute",
            body={"payload": payload},
        )

    def search_companies(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        limit = max(1, min(int(arguments.get("limit") or 5), 6))
        industry = str(arguments.get("industry") or "").strip()
        geography = str(arguments.get("geography") or "").strip()
        bands = arguments.get("employee_count") or []
        if not isinstance(bands, list):
            bands = [bands]
        headcount = [
            str(value).replace(",", "").strip()
            for value in bands
            if str(value).strip()
        ]
        context = ". ".join(
            part
            for part in (
                query,
                f"Industry: {industry}" if industry else "",
                f"Headquarters: {geography}" if geography else "",
            )
            if part
        )
        request: dict[str, Any] = {"query": context[:1_000], "limit": limit}
        if headcount:
            request["headcount"] = headcount[:8]
        if industry:
            request["industry"] = {
                "include": [
                    part.strip()
                    for part in re.split(r"[/|]", industry)
                    if part.strip()
                ][:6]
            }
        if locations := _hunter_locations(geography):
            request["headquarters_location"] = {"include": locations}
        data = _result_data(self._deepline("hunter_discover", request))
        rows = data.get("data") or data.get("rows") or []
        if not isinstance(rows, list):
            rows = []
        companies: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            domain = _domain(row.get("domain") or row.get("website"))
            identity = domain or str(
                row.get("organization")
                or row.get("company_name")
                or row.get("name")
                or ""
            ).strip().casefold()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            company: dict[str, Any] = {
                "company_name": str(
                    row.get("organization")
                    or row.get("company_name")
                    or row.get("name")
                    or ""
                )[:300],
                "domain": domain,
            }
            for source, target in (
                ("linkedin_url", "company_linkedin"),
                ("industry", "industry"),
                ("location", "location"),
                ("employee_count", "employee_count"),
                ("headcount", "employee_count"),
            ):
                if row.get(source) not in (None, "", [], {}):
                    company[target] = _json_safe(row[source])
            if domain:
                company["company_website"] = f"https://{domain}/"
            companies.append(company)
            if len(companies) >= limit:
                break
        return {"companies": companies, "count": len(companies)}

    def get_company_profile(self, arguments: dict[str, Any]) -> dict[str, Any]:
        domain = _domain(arguments.get("domain"))
        if not domain:
            raise ValueError("domain is required")
        columns = (
            "normalized_domain, domain, company_name, industry, location, "
            "linkedin_url, employee_count, year_founded, updated_at"
        )
        sql = (
            f"SELECT {columns} FROM companies WHERE normalized_domain = "
            f"{_sql_literal(domain)} LIMIT 3"
        )
        payload = self._deepline("free_simple_company_search", {"sql": sql})
        data = _result_data(payload)
        rows = data.get("rows") or (data.get("data") or {}).get("rows") or []
        if not isinstance(rows, list):
            rows = []
        exact = next(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and _domain(row.get("primary_domain") or row.get("domain")) == domain
            ),
            rows[0] if rows else {},
        )
        return {"domain": domain, "company": exact if isinstance(exact, dict) else {}}

    def get_company_events(self, arguments: dict[str, Any]) -> dict[str, Any]:
        domain = _domain(arguments.get("domain"))
        if not domain:
            raise ValueError("domain is required")
        categories = arguments.get("categories") or ["NEWS", "HIRING", "FUNDING"]
        if not isinstance(categories, list):
            categories = [categories]
        tools: list[str] = []
        unsupported: list[str] = []
        for category in categories:
            normalized = str(category).upper()
            tool = _EVENT_TOOLS.get(normalized)
            if tool is None:
                unsupported.append(normalized)
            elif tool not in tools:
                tools.append(tool)
        limit = max(1, min(int(arguments.get("limit") or 5), 5))
        events: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = [
            {"source": category, "error": "use targeted web search"}
            for category in unsupported
        ]
        for tool in tools[:3]:
            request: dict[str, Any] = {
                "company_id_or_domain": domain,
                "page": 1,
                "limit": limit,
            }
            if tool == "predictleads_company_job_openings":
                request.update({"active_only": True, "not_closed": True})
            elif tool == "predictleads_company_news_events":
                news_categories: list[str] = []
                for category in categories:
                    for value in _NEWS_CATEGORIES.get(str(category).upper(), []):
                        if value not in news_categories:
                            news_categories.append(value)
                if news_categories:
                    request["categories"] = news_categories
            try:
                data = _result_data(self._deepline(tool, request))
                events.append(
                    {"source": tool, "data": _project_event_data(data, limit)}
                )
            except Exception as exc:
                errors.append({"source": tool, "error": type(exc).__name__})
        return {"domain": domain, "events": events, "errors": errors}

    def search_web(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        mode = str(arguments.get("mode") or "search").strip().lower()
        if mode not in {"search", "news", "jobs"}:
            raise ValueError("mode must be search, news, or jobs")
        recency = arguments.get("recency_days")
        evaluation = date.fromisoformat(
            os.environ.get("BAKEOFF_EVALUATION_DATE")
            or os.environ.get("LAB_ARENA_EVALUATION_DATE")
            or date.today().isoformat()
        )
        suffix = ""
        if recency not in (None, ""):
            suffix += " after:" + (
                evaluation - timedelta(days=max(1, int(recency)))
            ).isoformat()
        elif mode == "news":
            suffix += " after:" + (evaluation - timedelta(days=365)).isoformat()
        if mode == "jobs":
            suffix += " (jobs OR careers OR hiring)"
        query = query[: max(0, 500 - len(suffix))].rstrip() + suffix
        limit = max(1, min(int(arguments.get("limit") or 5), 5))
        endpoint = {
            "search": "google",
            "news": "google_news",
            "jobs": "google_jobs",
        }[mode]
        payload = self._json_request(
            "GET",
            f"http://api.scrapingdog.com/{endpoint}",
            params={"query": query},
        )
        result_key = {
            "search": "organic_results",
            "news": "news_results",
            "jobs": "jobs_results",
        }[mode]
        raw_rows = payload.get(result_key, [])
        rows: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for raw in raw_rows if isinstance(raw_rows, list) else []:
            if not isinstance(raw, dict):
                continue
            row: dict[str, Any] = {}
            for key in (
                "title",
                "snippet",
                "source",
                "company_name",
                "location",
                "via",
            ):
                if raw.get(key) not in (None, "", [], {}):
                    row[key] = _json_safe(raw[key])
            observed = raw.get("date") or raw.get("lastUpdated")
            if observed:
                row["date"] = _json_safe(observed)
            extensions = raw.get("extensions")
            if isinstance(extensions, list):
                row["details"] = _json_safe(extensions[:6])
            apply_url = ""
            apply_links = raw.get("apply_links")
            if isinstance(apply_links, list):
                for candidate in apply_links:
                    if not isinstance(candidate, dict):
                        continue
                    apply_url = _evidence_url(
                        candidate.get("link") or candidate.get("url")
                    )
                    if apply_url:
                        row["apply_url"] = apply_url
                        break
            url = _evidence_url(raw.get("url") or raw.get("link")) or apply_url
            if not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            row["url"] = url
            rows.append(row)
            if len(rows) >= limit:
                break
        return {"results": rows, "count": len(rows), "mode": mode}

    def fetch_page(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = str(arguments.get("url") or "").strip()
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("an absolute HTTPS URL is required")
        max_chars = max(1_000, min(int(arguments.get("max_chars") or 2_500), 4_000))
        response = self._client.get(
            "http://api.scrapingdog.com/scrape",
            params={"url": url, "dynamic": "false"},
            timeout=self.timeout,
        )
        if not response.is_success:
            raise RuntimeError(f"Arena page fetch returned HTTP {response.status_code}")
        title, text = _page_content(response.text, max_chars)
        return {
            "url": url,
            "status_code": response.status_code,
            "title": title,
            "text": text,
            "source": "ScrapingDog",
        }

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "submit_companies":
            companies = validate_companies(arguments.get("companies"), max_companies=5)
            return {"companies": companies}
        if name not in {
            "search_companies",
            "get_company_profile",
            "get_company_events",
            "search_web",
            "fetch_page",
        }:
            raise ValueError(f"unknown tool: {name}")
        return getattr(self, name)(arguments)


__all__ = [
    "ArenaOpenRouterTransport",
    "ArenaToolClient",
    "arena_openrouter_http_client",
    "arena_socket_path",
    "strip_arena_request_headers",
]
