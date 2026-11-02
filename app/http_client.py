"""Shared httpx client construction.

Why this exists: the sandbox's egress-proxy environment carries a
malformed ``no_proxy`` entry (``[::1]``), and a bare ``httpx.Client()``
reads it at construction time and raises ``InvalidURL`` ("Invalid port:
':1]'"). Passing the proxy explicitly with ``trust_env=False`` avoids
parsing ``no_proxy`` entirely, while still routing through the proxy
when one is configured (this sandbox) and going direct when none is
(Render, where no proxy env vars exist).
"""

from __future__ import annotations

import os

import httpx


def make_http_client(
    *,
    timeout: float = 20.0,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Build an httpx client that survives the sandbox proxy env."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    kwargs: dict = {"timeout": timeout, "trust_env": False}
    if proxy:
        kwargs["proxy"] = proxy
    if transport is not None:
        kwargs["transport"] = transport
    return httpx.Client(**kwargs)


__all__ = ["make_http_client"]
