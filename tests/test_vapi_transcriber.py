"""Vapi wire-shaped loopback cases, not calls to Vapi or a speech model."""
import asyncio
import json
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import pytest
from aiohttp import ClientSession, ClientTimeout, WSServerHandshakeError, WSMsgType, web
from pipecat.audio.vad.vad_analyzer import VADState

from examples.vapi.transcriber import (
    CallerPCM, MAX_CALL_BYTES, ProtocolError, START, STATE, create_app, parse_start, response,
)
from hume_gateway_helpers import ALLOWED_PORTS
from hume_gateway_helpers import exact_loopback_only  # noqa: F401 - autouse fixture

TOKEN = "synthetic-vapi-upgrade-key"
ORUK = "synthetic-oruk-provider-key"


class MarkerVAD:
    """First nonzero sample is synthetic speech; zeros are synthetic silence."""
    def set_sample_rate(self, rate):
        assert rate == 16000
    def num_frames_required(self):
        return 512
    async def analyze_audio(self, audio):
        return VADState.SPEAKING if any(audio) else VADState.QUIET


@asynccontextmanager
async def bridge(endpoint, **kwargs):
    app = create_app(inbound_token=TOKEN, oruk_api_key=ORUK, endpoint=endpoint,
                     analyzer_factory=MarkerVAD, **kwargs)
    runner = web.AppRunner(app, access_log=None, handle_signals=False, shutdown_timeout=3)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    upstream = urlsplit(endpoint).port
    ALLOWED_PORTS.update((port, upstream))
    try:
        async with ClientSession(timeout=ClientTimeout(total=5), trust_env=False) as client:
            yield client, f"http://127.0.0.1:{port}/api/custom-transcriber", app
    finally:
        await runner.cleanup()
        ALLOWED_PORTS.difference_update((port, upstream))
        assert not app[STATE]["active"]


async def connect(client, url):
    return await client.ws_connect(url, headers={"Authorization": "Bearer " + TOKEN})


def stereo(caller, assistant=27_000):
    return (caller.to_bytes(2, "little", signed=True) + assistant.to_bytes(2, "little", signed=True)) * 512


async def event(ws):
    message = await asyncio.wait_for(ws.receive(), 3)
    assert message.type == WSMsgType.TEXT
    return json.loads(message.data)


@pytest.mark.parametrize("change", [
    {"channels": 1}, {"channels": True}, {"channels": 2.0}, {"sampleRate": 8000},
    {"sampleRate": 16000.0}, {"encoding": "mulaw"}, {"container": "wav"}, {"type": "stop"},
    {"extra": "unrecognized"},
])
def test_start_profile_is_explicit(change):
    parse_start(json.dumps(START))
    with pytest.raises(ProtocolError):
        parse_start(json.dumps({**START, **change}))


@pytest.mark.parametrize("raw", ["not-json", "[]", "null", "{}", '{"type":"stop","type":"start"}'])
def test_invalid_start_never_echoes_raw(raw):
    with pytest.raises(ProtocolError) as err:
        parse_start(raw)
    assert str(err.value) in {"invalid_start", "unsupported_audio_profile"}


@pytest.mark.parametrize("split", [1, 2, 3, 5, 7, 1023])
def test_stereo_samples_cross_packets_without_assistant_leakage(split):
    pcm = CallerPCM()
    wire = stereo(-1234)
    result = pcm.feed(wire[:split]) + pcm.feed(wire[split:])
    assert result == (-1234).to_bytes(2, "little", signed=True) * 512
    assert pcm.tail == b""


def test_audio_and_text_bounds():
    for data in (b"", b"a" * 64_001):
        with pytest.raises(ProtocolError):
            CallerPCM().feed(data)
    pcm = CallerPCM()
    pcm.received = MAX_CALL_BYTES
    with pytest.raises(ProtocolError, match="call_audio_limit"):
        pcm.feed(b"1234")
    with pytest.raises(ProtocolError):
        response("x" * 16_001, "partial")


async def test_disabled_and_unauthenticated_never_dispatch(gateway):
    async with gateway() as native:
        async with bridge(native.endpoint) as (client, url, app):
            with pytest.raises(WSServerHandshakeError) as err:
                await connect(client, url)
            assert err.value.status == 503
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            for headers, suffix in (({}, ""), ({"Authorization": "Bearer wrong"}, ""),
                                     ({"Authorization": "Bearer " + TOKEN}, "?key=synthetic")):
                with pytest.raises(WSServerHandshakeError) as err:
                    await client.ws_connect(url + suffix, headers=headers)
                assert err.value.status == 401
            assert not app[STATE]["active"]
        assert native.requests == 0


async def test_partials_before_commit_exact_caller_audio_final_after_clean_close(gateway):
    async with gateway() as native:
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            ws = await connect(client, url)
            await ws.send_json(START)
            wire = stereo(-1234)
            for piece in (wire[:1], wire[1:9], wire[9:]):
                await ws.send_bytes(piece)
            first, second = await event(ws), await event(ws)
            assert first == response("Provisional ", "partial")
            assert second == response("Provisional words", "partial")
            assert not native.closed.is_set()  # Partials arrive while turn remains open.
            await ws.send_bytes(stereo(0))
            assert await event(ws) == response("Final turn 0.", "final")
            assert native.closed.is_set()
            assert native.requests == 1
            assert bytes(native.audio[0]) == (-1234).to_bytes(2, "little", signed=True) * 512 + b"\x00" * 1024
            assert native.auth_headers == ["Bearer " + ORUK] and ORUK not in native.queries[0]
            assert native.configs[0]["phrase_emotions"] is False
            await ws.close()


@pytest.mark.parametrize("mode", ["final_only", "usage_only", "error_after_usage", "conflicting_final"])
async def test_incomplete_or_failed_native_turn_never_emits_final_or_retries(gateway, mode):
    async with gateway(mode=mode) as native:
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            ws = await connect(client, url)
            await ws.send_json(START)
            await ws.send_bytes(stereo(1234) + stereo(0))
            seen = []
            while (message := await asyncio.wait_for(ws.receive(), 3)).type == WSMsgType.TEXT:
                seen.append(json.loads(message.data))
            assert not any(m.get("transcriptType") == "final" for m in seen)
            assert native.requests == 1
            assert ws.close_code == 1011


async def test_concurrent_calls_are_isolated_and_capacity_is_enforced(gateway):
    async with gateway() as native:
        async with bridge(native.endpoint, enabled=True, max_sessions=2) as (client, url, app):
            sockets = [await connect(client, url), await connect(client, url)]
            with pytest.raises(WSServerHandshakeError) as err:
                await connect(client, url)
            assert err.value.status == 503
            for value, ws in zip((1111, 2222), sockets):
                await ws.send_json(START)
                await ws.send_bytes(stereo(value))
                assert (await event(ws))["transcriptType"] == "partial"
                assert (await event(ws))["transcriptType"] == "partial"
            assert len(set(native.client_request_ids)) == 2
            for value, pcm in zip((1111, 2222), native.audio):
                assert bytes(pcm) == value.to_bytes(2, "little") * 512
            for ws in sockets:
                await ws.close()
        assert native.requests == 2 and all(s.closed for s in native.sockets)


async def test_shutdown_joins_open_call_and_native_socket(gateway):
    async with gateway() as native:
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            ws = await connect(client, url)
            await ws.send_json(START)
            await ws.send_bytes(stereo(1234))
            await event(ws)
            await app.shutdown()
            assert not app[STATE]["active"]
            await asyncio.wait_for(native.closed.wait(), 2)
            assert native.sockets[0].closed
            await ws.close()


@pytest.mark.parametrize("first", ["binary", "invalid-start", "repeat-start"])
async def test_bad_control_flow_closes_without_provider_work(gateway, first):
    async with gateway() as native:
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            ws = await connect(client, url)
            if first == "binary":
                await ws.send_bytes(stereo(1234))
            elif first == "invalid-start":
                await ws.send_json({**START, "channels": 1})
            else:
                await ws.send_json(START)
                await ws.send_json(START)
            assert (await asyncio.wait_for(ws.receive(), 3)).type == WSMsgType.CLOSE
            assert native.requests == 0


async def test_two_utterances_keep_order_on_one_connection(gateway):
    async with gateway() as native:
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            ws = await connect(client, url)
            await ws.send_json(START)
            for index, value in enumerate((1234, 5678)):
                await ws.send_bytes(stereo(value) + stereo(0))
                assert await event(ws) == response("Provisional ", "partial")
                assert await event(ws) == response("Provisional words", "partial")
                assert await event(ws) == response(f"Final turn {index}.", "final")
                assert bytes(native.audio[index]) == value.to_bytes(2, "little") * 512 + b"\x00" * 1024
            assert len(set(native.client_request_ids)) == native.requests == 2
            await ws.close()


async def test_stalled_provider_hits_bounded_queue_and_is_joined(gateway, monkeypatch):
    from examples.vapi import transcriber as module
    started, closed = asyncio.Event(), asyncio.Event()
    calls = []
    async def stalled(**kwargs):
        calls.append(kwargs)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    monkeypatch.setattr(module, "run_turn", stalled)
    async with gateway() as native:
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            ws = await connect(client, url)
            await ws.send_json(START)
            await ws.send_bytes(stereo(1234))
            await asyncio.wait_for(started.wait(), 2)
            for _ in range(3):
                await ws.send_bytes(stereo(1234) * 30)
            assert (await asyncio.wait_for(ws.receive(), 3)).type == WSMsgType.CLOSE
            assert ws.close_code == 1011
            assert closed.is_set() and len(calls) == 1
            assert calls[0]["max_connect_retries"] == 0 and native.requests == 0


async def test_empty_final_cannot_leave_provisional_text_open(gateway, monkeypatch):
    from examples.vapi import transcriber as module
    original = module.run_turn
    async def empty_final(**kwargs):
        result = await original(**kwargs)
        result.transcript = " "
        return result
    monkeypatch.setattr(module, "run_turn", empty_final)
    async with gateway() as native:
        async with bridge(native.endpoint, enabled=True) as (client, url, app):
            ws = await connect(client, url)
            await ws.send_json(START)
            await ws.send_bytes(stereo(1234) + stereo(0))
            seen = []
            while (message := await asyncio.wait_for(ws.receive(), 3)).type == WSMsgType.TEXT:
                seen.append(json.loads(message.data))
            assert not any(m.get("transcriptType") == "final" for m in seen)
            assert ws.close_code == 1011 and native.requests == 1
