"""Bounded, caller-only Vapi PCM bridge. Importing this module opens no listener.

The deployment owner supplies authentication, TLS and a permitted Oruk key.
Vapi owns the LLM/voice loop; this example only returns transcripts.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import uuid
from collections import deque

from aiohttp import WSMsgType, web
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState

from pipecat_oruk.realtime import (
    ENDPOINT, TRANSCRIPT, RealtimeOptions, new_http_session, run_turn, validate_endpoint,
)

STATE = web.AppKey("vapi_bridge", dict)
START = {"type": "start", "encoding": "linear16", "container": "raw",
         "sampleRate": 16000, "channels": 2}
MAX_WIRE_BYTES = 64_000
MAX_CALL_BYTES = 600 * 64_000  # Ten minutes of stereo PCM16 at 16 kHz.
MAX_TEXT = 16_000


class ProtocolError(ValueError):
    """Fixed codes only; never put client data in exception messages."""


def parse_start(text: str) -> None:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError("invalid_start")
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        raise ProtocolError("invalid_start") from None
    if (not isinstance(value, dict) or value != START
            or type(value.get("sampleRate")) is not int
            or type(value.get("channels")) is not int):
        raise ProtocolError("unsupported_audio_profile")


class CallerPCM:
    """Retain split stereo samples; never trim or mix the assistant channel."""
    def __init__(self):
        self.tail = b""
        self.received = 0

    def feed(self, data: bytes) -> bytes:
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_WIRE_BYTES:
            raise ProtocolError("invalid_audio_frame")
        self.received += len(data)
        if self.received > MAX_CALL_BYTES:
            raise ProtocolError("call_audio_limit")
        joined = self.tail + data
        end = len(joined) - len(joined) % 4
        self.tail = joined[end:]
        out = bytearray(end // 2)
        out[0::2], out[1::2] = joined[:end:4], joined[1:end:4]
        return bytes(out)


def response(text: str, kind: str) -> dict:
    if not isinstance(text, str) or len(text) > MAX_TEXT or kind not in ("partial", "final"):
        raise ProtocolError("invalid_transcript")
    return {"type": "transcriber-response", "transcription": text,
            "channel": "customer", "transcriptType": kind}


async def _session(socket, http, api_key, endpoint, analyzer):
    """Every provider task belongs to this connection's TaskGroup."""
    analyzer.set_sample_rate(16_000)
    frame_bytes = analyzer.num_frames_required() * 2
    if type(frame_bytes) is not int or not 0 < frame_bytes <= 2_048:
        raise ProtocolError("invalid_vad_frame_size")
    outgoing = asyncio.Queue(maxsize=64)
    owned = set()
    current = None
    preroll = deque(maxlen=20)
    pending = bytearray()
    demux = CallerPCM()

    async def send():
        while True:
            event = await outgoing.get()
            async with asyncio.timeout(2):
                await socket.send_json(event)

    async def transcribe(audio):
        partial = ""
        async def chunks():
            while (chunk := await audio.get()) is not None:
                yield chunk
        def on_event(event):
            nonlocal partial
            if event.get("type") == TRANSCRIPT + "delta":
                partial += event["delta"]
                # Vapi partials replace their predecessor; Oruk events are deltas.
                outgoing.put_nowait(response(partial, "partial"))
        result = await run_turn(
            session=http, api_key=api_key, endpoint=endpoint, audio=chunks(),
            request_id="vapi_" + uuid.uuid4().hex,
            options=RealtimeOptions(phrase_emotions=False, max_turn_seconds=30, finish_timeout=10),
            on_event=on_event, max_connect_retries=0,
        )
        # A provider's early final is insufficient: require usage and clean close.
        if not isinstance(result.transcript, str) or not result.transcript.strip():
            raise ProtocolError("empty_final_transcript")
        outgoing.put_nowait(response(result.transcript, "final"))

    async def pump(group):
        nonlocal current
        async with asyncio.timeout(5):
            hello = await socket.receive()
        if hello.type != WSMsgType.TEXT:
            raise ProtocolError("start_required")
        parse_start(hello.data)
        while True:
            async with asyncio.timeout(15):
                message = await socket.receive()
            if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                return  # Disconnect cancels unfinished work; it is not a commit.
            if message.type != WSMsgType.BINARY:
                raise ProtocolError("binary_audio_required")
            pending.extend(demux.feed(message.data))
            while len(pending) >= frame_bytes:
                frame = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
                state = await analyzer.analyze_audio(frame)
                if current is None:
                    preroll.append(frame)
                    if state not in (VADState.STARTING, VADState.SPEAKING):
                        continue
                    queue = asyncio.Queue(maxsize=64)
                    for chunk in preroll:
                        queue.put_nowait(chunk)
                    preroll.clear()
                    task = group.create_task(transcribe(queue))
                    owned.add(task)
                    task.add_done_callback(owned.discard)
                    current = queue, task
                else:
                    current[0].put_nowait(frame)
                if state == VADState.QUIET:
                    current[0].put_nowait(None)
                    await current[1]
                    current = None
                    # Serialize completed utterances. Transport backpressure and
                    # finite queues bound waiting; real call latency needs testing.

    async with asyncio.TaskGroup() as group:
        owned.add(group.create_task(send()))
        reader = group.create_task(pump(group))
        owned.add(reader)
        try:
            await reader
        finally:
            for task in list(owned):
                task.cancel()


def create_app(*, inbound_token: str, oruk_api_key: str, enabled: bool = False,
               endpoint: str = ENDPOINT, max_sessions: int = 2, analyzer_factory=None):
    """Return an unbound app. No provider/client is acquired until authentication."""
    if type(enabled) is not bool or type(max_sessions) is not int or not 1 <= max_sessions <= 32:
        raise ValueError("invalid_bridge_limits")
    for key in (inbound_token, oruk_api_key):
        if (not isinstance(key, str) or not 16 <= len(key) <= 512
                or not key.isascii() or any(c.isspace() for c in key)):
            raise ValueError("explicit_credentials_required")
    if hmac.compare_digest(inbound_token, oruk_api_key):
        raise ValueError("separate_inbound_credential_required")
    validate_endpoint(endpoint)
    factory = analyzer_factory or (lambda: SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.6)))
    app = web.Application(client_max_size=MAX_WIRE_BYTES)
    state = app[STATE] = {"active": set(), "accepting": enabled}

    async def handler(request):
        if not state["accepting"]:
            raise web.HTTPServiceUnavailable(text="bridge_disabled")
        auth = request.headers.getall("Authorization", [])
        expected = "Bearer " + inbound_token
        if (request.query_string or len(auth) != 1
                or not hmac.compare_digest(auth[0].encode(), expected.encode())):
            raise web.HTTPUnauthorized(text="authentication_required")
        if len(state["active"]) >= max_sessions:
            raise web.HTTPServiceUnavailable(text="bridge_at_capacity")
        owner = asyncio.current_task()
        state["active"].add(owner)
        socket = web.WebSocketResponse(max_msg_size=MAX_WIRE_BYTES, heartbeat=10,
                                       receive_timeout=None, compress=False)
        try:
            await socket.prepare(request)
            async with asyncio.timeout(600):
                async with new_http_session() as http:
                    await _session(socket, http, oruk_api_key, endpoint, factory())
        except asyncio.CancelledError:
            raise
        except web.HTTPException:
            raise
        except Exception:
            if socket.prepared:
                await socket.close(code=1011, message=b"transcriber_session_failed")
        finally:
            try:
                if socket.prepared:
                    await socket.close()
            finally:
                state["active"].discard(owner)
        return socket

    async def shutdown(_app):
        state["accepting"] = False
        active = list(state["active"])
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)

    app.router.add_get("/api/custom-transcriber", handler)
    app.on_shutdown.append(shutdown)
    return app
