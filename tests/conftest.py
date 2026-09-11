from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import WSMsgType, web

from pipecat_oruk.realtime import EMOTION, TRANSCRIPT


@dataclass
class Gateway:
    mode: str = "normal"
    fail_upgrades: list[int] = field(default_factory=list)
    requests: int = 0
    audio: list[bytearray] = field(default_factory=list)
    configs: list[dict[str, Any]] = field(default_factory=list)
    auth_headers: list[str | None] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    sockets: list[web.WebSocketResponse] = field(default_factory=list)
    first_audio: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    endpoint: str = ""

    async def handler(self, request: web.Request) -> web.StreamResponse:
        self.requests += 1
        self.auth_headers.append(request.headers.get("Authorization"))
        self.queries.append(request.query_string)
        if self.fail_upgrades:
            return web.Response(status=self.fail_upgrades.pop(0))
        if self.mode == "redirect":
            raise web.HTTPFound("http://127.0.0.1:1/should-not-follow")
        socket = web.WebSocketResponse(protocols=["oruk-realtime"])
        await socket.prepare(request)
        self.sockets.append(socket)
        index = len(self.audio)
        pcm = bytearray()
        self.audio.append(pcm)
        request_id = request.headers.get("X-Request-ID", "req_local")

        async def send(event: dict[str, Any]) -> None:
            await socket.send_json({"request_id": request_id, **event})

        try:
            await send({"type": "session.created", "session": {"model": "oruk-realtime"}})
            async for message in socket:
                if message.type == WSMsgType.BINARY:
                    assert len(message.data) <= 10_240 and len(message.data) % 2 == 0
                    first = not pcm
                    pcm.extend(message.data)
                    self.first_audio.set()
                    if self.mode == "disconnect":
                        await socket.close(code=1011)
                        break
                    if self.mode == "invalid_json":
                        await socket.send_str("not-json")
                    elif first:
                        await send({"type": TRANSCRIPT + "delta", "delta": "Provisional "})
                        await send({"type": TRANSCRIPT + "delta", "delta": "words"})
                elif message.type == WSMsgType.TEXT:
                    event = json.loads(message.data)
                    if event["type"] == "session.update":
                        assert not pcm
                        self.configs.append(event["session"])
                        await send({"type": "session.updated", "session": event["session"]})
                    elif event["type"] == "input_audio_buffer.commit":
                        if self.mode == "hang":
                            continue
                        final = {
                            "type": TRANSCRIPT + "completed",
                            "transcript": f"Final turn {index}.",
                        }
                        if self.mode != "usage_only":
                            await send(final)
                        if self.mode == "conflicting_final":
                            await send({**final, "transcript": "A conflicting final."})
                        if self.mode == "duplicate":
                            await send(final)
                        phrase = {
                            "type": EMOTION + "completed",
                            "phrase_id": "phrase_1",
                            "start": 0.0,
                            "end": len(pcm) / 32_000,
                            "speaker": "speaker_0",
                            "text": "A provisional phrase.",
                            "emotions": [{"label": "happy", "score": 0.7}],
                            "top_emotion": {"label": "happy", "score": 0.7},
                        }
                        if self.mode == "bad_score":
                            phrase["emotions"] = [{"label": "happy", "score": float("nan")}]
                        if self.mode == "phrase_failure":
                            phrase = {
                                **phrase,
                                "type": EMOTION + "failed",
                                "error": {"code": "phrase_emotion_unavailable"},
                            }
                        await send(phrase)
                        if self.mode == "duplicate":
                            await send(phrase)
                        if self.mode != "final_only":
                            await send(
                                {
                                    "type": "session.usage",
                                    "usage": {
                                        "audio_seconds": len(pcm) / 32_000,
                                        "billable_seconds": max(1, len(pcm) / 32_000),
                                    },
                                }
                            )
                        if self.mode == "error_after_usage":
                            await send(
                                {
                                    "type": "error",
                                    "error": {"code": "realtime_upstream_unavailable"},
                                }
                            )
                        await socket.close(code=1000)
                        break
        finally:
            self.closed.set()
        return socket


@pytest.fixture
def gateway():
    @asynccontextmanager
    async def open_gateway(**kwargs):
        server = Gateway(**kwargs)
        app = web.Application()
        app.router.add_get("/v1/realtime", server.handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        server.endpoint = f"ws://127.0.0.1:{port}/v1/realtime?model=oruk-realtime"
        try:
            yield server
        finally:
            for socket in server.sockets:
                await socket.close()
            await runner.cleanup()

    return open_gateway
