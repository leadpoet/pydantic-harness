"""Thin client for the host-owned provider boundary."""

from __future__ import annotations

import os
from typing import Any

import httpx

from arena_client import ArenaToolClient, WORKER_SOCKET_ENV


class ToolClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 90.0,
    ):
        socket_path = os.environ.get(WORKER_SOCKET_ENV, "").strip()
        self.arena = (
            ArenaToolClient(socket_path=socket_path, timeout=timeout)
            if socket_path
            else None
        )
        self.base_url = (base_url or os.environ.get("BAKEOFF_TOOL_URL", "")).rstrip("/")
        self.token = token or os.environ.get("BAKEOFF_TOOL_TOKEN", "")
        if self.arena is None and (not self.base_url or not self.token):
            raise RuntimeError("BAKEOFF_TOOL_URL and BAKEOFF_TOOL_TOKEN are required")
        self.timeout = timeout

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.arena is not None:
            return self.arena.call(name, arguments)
        response = httpx.post(
            f"{self.base_url}/tool",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"name": name, "arguments": arguments},
            timeout=self.timeout,
        )
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(
                f"{name} returned HTTP {response.status_code} with invalid JSON"
            ) from None
        if not response.is_success or not payload.get("ok"):
            raise RuntimeError(
                str(
                    payload.get("error")
                    or f"{name} failed with HTTP {response.status_code}"
                )
            )
        return payload.get("result")

    def search_companies(self, **arguments: Any) -> Any:
        return self.call("search_companies", arguments)

    def get_company_profile(self, **arguments: Any) -> Any:
        return self.call("get_company_profile", arguments)

    def get_company_events(self, **arguments: Any) -> Any:
        return self.call("get_company_events", arguments)

    def search_web(self, **arguments: Any) -> Any:
        return self.call("search_web", arguments)

    def fetch_page(self, **arguments: Any) -> Any:
        return self.call("fetch_page", arguments)

    def submit_companies(self, companies: list[dict[str, Any]]) -> Any:
        return self.call("submit_companies", {"companies": companies})
