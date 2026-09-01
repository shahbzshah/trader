"""httpx MockTransport helpers for brain.py tests."""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx


def make_transport(responder: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Wrap a responder in httpx.MockTransport."""
    return httpx.MockTransport(responder)


def json_response(payload: dict, request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=payload, request=request)


def make_chat_response(content: str, model: str = "test-model") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def recording_transport(responses: list[dict]) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Transport that returns responses in order and captures every request."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if not responses:
            return httpx.Response(500, json={"error": "no more canned responses"},
                                    request=request)
        return httpx.Response(200, json=responses.pop(0), request=request)

    return httpx.MockTransport(handler), captured


def make_raising_transport(exc: Exception) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)