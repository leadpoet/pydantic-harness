from __future__ import annotations

import base64
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx2

import arena_client


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(size - len(output))
        if not chunk:
            raise RuntimeError("test client closed early")
        output.extend(chunk)
    return bytes(output)


class _Worker(AbstractContextManager["_Worker"]):
    def __init__(self, response: dict) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="arena-worker-")
        self.path = str(Path(self._directory.name) / "worker.sock")
        self.frame: dict | None = None
        self._response = response
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        connection, _ = self._server.accept()
        with connection:
            size = int.from_bytes(_recv_exact(connection, 4), "big")
            self.frame = json.loads(_recv_exact(connection, size))
            encoded = json.dumps(self._response, separators=(",", ":")).encode()
            connection.sendall(len(encoded).to_bytes(4, "big") + encoded)

    def __enter__(self) -> "_Worker":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._thread.join(timeout=2)
        self._server.close()
        self._directory.cleanup()


def _success(body: object, status: int = 200) -> dict:
    encoded = json.dumps(body, separators=(",", ":")).encode()
    return {
        "status": status,
        "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(encoded).decode(),
    }


class ArenaClientTests(unittest.TestCase):
    def test_dispatch_sends_only_the_plain_operation_frame(self) -> None:
        with _Worker(_success({"results": []})) as worker:
            response = arena_client.dispatch(
                "exa.search",
                {"query": "product launch", "numResults": 5},
                timeout_seconds=12,
                socket_path=worker.path,
            )

        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body), {"results": []})
        self.assertEqual(
            worker.frame,
            {
                "schema_version": "leadpoet.lab_arena.operation_frame.v1",
                "operation_id": "exa.search",
                "parameters": {"query": "product launch", "numResults": 5},
                "timeout_ms": 12_000,
            },
        )
        self.assertNotIn("credential", json.dumps(worker.frame).lower())
        self.assertNotIn("authorization", json.dumps(worker.frame).lower())

    def test_dispatch_reports_a_generic_worker_error(self) -> None:
        with _Worker({"error": "budget_exhausted"}) as worker:
            with self.assertRaisesRegex(
                arena_client.ArenaClientError, "budget_exhausted"
            ):
                arena_client.dispatch(
                    "exa.search",
                    {"query": "test"},
                    socket_path=worker.path,
                )

    def test_dispatch_caps_each_operation_at_the_arena_timeout(self) -> None:
        with _Worker(_success({"results": []})) as worker:
            arena_client.dispatch(
                "exa.search",
                {"query": "test"},
                timeout_seconds=90,
                socket_path=worker.path,
            )

        self.assertEqual(worker.frame["timeout_ms"], 60_000)

    def test_tool_client_uses_only_approved_arena_operations(self) -> None:
        calls: list[tuple[str, dict]] = []

        def answer(operation_id: str, parameters: dict, **_: object) -> dict:
            calls.append((operation_id, parameters))
            return {"results": []}

        with patch("arena_client._json_call", side_effect=answer):
            client = arena_client.ArenaToolClient(socket_path="/tmp/worker.sock")
            client.search_companies({"query": "B2B software", "limit": 3})
            client.search_web({"query": "funding", "mode": "news", "limit": 2})
            client.fetch_page({"url": "https://example.com/news", "max_chars": 2_000})

        self.assertEqual(
            [operation_id for operation_id, _ in calls],
            ["exa.search", "exa.search", "exa.contents"],
        )

    def test_run_icp_uses_arena_socket_without_provider_credentials(self) -> None:
        import harness
        from experiments.harness_bakeoff.adapters import pydantic_ai

        self.addCleanup(pydantic_ai.LAST_USAGE.clear)

        completion = {
            "id": "chatcmpl-test",
            "created": 1,
            "model": "openai/gpt-5.6-sol",
            "object": "chat.completion",
            "provider": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "submit_companies",
                                    "arguments": json.dumps({"companies": []}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }
        with _Worker(_success(completion)) as worker:
            with patch.dict(
                os.environ,
                {
                    "LAB_ARENA_WORKER_SOCKET": worker.path,
                    "BAKEOFF_RUN_TIMEOUT_SECONDS": "5",
                },
                clear=True,
            ):
                self.assertNotIn("OPENROUTER_API_KEY", os.environ)
                companies = harness.run_icp({"icp_id": "daily-1", "employee_count": []})

        self.assertEqual(companies, [])
        self.assertEqual(worker.frame["operation_id"], "openrouter.chat")
        self.assertEqual(worker.frame["parameters"]["max_tokens"], 4_096)
        self.assertNotIn(
            arena_client.ARENA_OPENROUTER_KEY,
            json.dumps(worker.frame),
        )


class ArenaOpenRouterTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_removes_sdk_headers_and_fixed_fields(self) -> None:
        completion = {
            "id": "chatcmpl-test",
            "created": 1,
            "model": "openai/test-model",
            "object": "chat.completion",
            "provider": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        with _Worker(_success(completion)) as worker:
            transport = arena_client.ArenaOpenRouterTransport(socket_path=worker.path)
            request = httpx2.Request(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer must-not-cross"},
                json={
                    "model": "openai/test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 20,
                    "stream": False,
                    "usage": {"include": True},
                },
            )
            response = await transport.handle_async_request(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(worker.frame["operation_id"], "openrouter.chat")
        self.assertEqual(
            worker.frame["parameters"],
            {
                "model": "openai/test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 20,
            },
        )
        self.assertNotIn("must-not-cross", json.dumps(worker.frame))


if __name__ == "__main__":
    unittest.main()
