"""Actual local aiohttp WebSockets + Pipecat; named synthetic VAD/STT/LLM/TTS.

No model, provider, browser, external DNS, TLS or production qualification.
Root/application owns a finite outer process deadline in addition to these joins.
"""
import asyncio
import base64
import io
import json
from types import SimpleNamespace
import wave

import pytest
from aiohttp import WSMsgType, web
from pipecat.frames.frames import EndFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

from examples.hume_evi.gateway import GatewayLimits, ProviderBridge, Session, TURN, create_gateway
from hume_gateway_helpers import KEY, SETTINGS, NamedTTS, local_gateway
from hume_gateway_helpers import exact_loopback_only  # noqa: F401 - autouse fixture
from hume_helpers import eventually


async def receive_until(ws, kind, *, timeout=4):
    result = []
    async with asyncio.timeout(timeout):
        while True:
            packet = await ws.receive()
            assert packet.type == WSMsgType.TEXT, ("unexpected_socket_terminal", packet.type)
            value = json.loads(packet.data)
            result.append(value)
            if value["type"] == kind:
                return result
            assert value["type"] != "error", value.get("code")


async def connect(client, url):
    ws = await client.ws_connect(url, params={"api_key": KEY}, compress=0)
    await ws.send_json(SETTINGS)
    metadata = (await receive_until(ws, "chat_metadata"))[-1]
    return ws, metadata


async def audio(ws, marker=1):
    pcm = marker.to_bytes(2, "little") * 320
    await ws.send_json({"type": "audio_input", "data": base64.b64encode(pcm).decode()})
    return pcm


async def assert_closed(gateway, made):
    await eventually(lambda: not gateway.active)
    assert all(report["all_owned_tasks_done"] and report["provider_cleanup_joined"] for report in gateway.reports)
    assert all(owner.closed.is_set() for owner in made)


def test_factory_is_disabled_unbound_and_requires_explicit_callbacks():
    app = create_gateway()
    assert len(list(app.router.routes())) == 2  # aiohttp HEAD registration + GET.
    with pytest.raises(ValueError, match="explicit_auth_and_providers_required"):
        create_gateway(enabled=True)
    with pytest.raises(ValueError, match="invalid_gateway_limits"):
        GatewayLimits(sessions=True)


async def test_disabled_auth_and_query_refusals_have_no_provider():
    async with local_gateway(enabled=False) as (client, url, gateway, made, _):
        async with client.get(url) as response:
            assert response.status == 404
        assert made == [] and not gateway.active
    async with local_gateway() as (client, url, gateway, made, _):
        for params, status in (({}, 401), ({"api_key": "wrong-synthetic-credential"}, 401), ({"api_key": KEY, "config_id": "no"}, 400)):
            async with client.get(url, params=params) as response:
                assert response.status == status
                assert KEY not in await response.text()
        assert made == [] and not gateway.active


@pytest.mark.parametrize("change", [
    {"audio": {"encoding": "webm", "sample_rate": 16000, "channels": 1}},
    {"audio": {"encoding": "linear16", "sample_rate": 48000, "channels": 1}},
    {"voice_id": "not-supported"}, {"tools": []}, {"custom_session_id": "no-resume"},
])
async def test_settings_refuse_before_provider_creation(change):
    async with local_gateway() as (client, url, gateway, made, _):
        ws = await client.ws_connect(url, params={"api_key": KEY})
        await ws.send_json({**SETTINGS, **change})
        error = (await receive_until(ws, "error"))[-1]
        assert error["code"] == "invalid_message"
        await ws.close()
        await assert_closed(gateway, made)
        assert made == []


async def test_two_pcm_turns_preserve_primary_words_exact_ms_and_real_48k_wav():
    async with local_gateway() as (client, url, gateway, made, _):
        ws, metadata = await connect(client, url)
        all_pcm, ids = b"", []
        for n in (1, 2):
            all_pcm += await audio(ws, n)
            messages = await receive_until(ws, "assistant_end")
            primary = [m for m in messages if m["type"] == "user_message"]
            assert primary == [{"type": "user_message", "from_text": False, "interim": False,
                                "message": {"role": "user", "content": f"Primary {n}."}, "models": {},
                                "time": {"begin": (n - 1) * 20, "end": n * 20}}]
            assistant = next(m for m in messages if m["type"] == "assistant_message")
            assert assistant["models"] == {} and assistant["message"]["content"] == f"Synthetic reply {n}."
            output = next(m for m in messages if m["type"] == "audio_output")
            ids.append(output["id"])
            assert output["id"] == assistant["id"] and output["index"] == 0
            with wave.open(io.BytesIO(base64.b64decode(output["data"], validate=True)), "rb") as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (48000, 1, 2, 960)
                assert wav.readframes(960) == b"\x01\x00" * 960
        assert len(set(ids)) == 2 and metadata["chat_id"] != metadata["chat_group_id"]
        assert bytes(made[0].stt.pcm) == all_pcm
        assert any(m.get("content") == "Primary 1." for m in made[0].llm.inputs[1])
        await ws.close()
        await assert_closed(gateway, made)
        assert gateway.reports[0]["input_bytes"] == 1280 and gateway.reports[0]["turns"] == 2


async def test_text_input_has_no_fabricated_emotions_or_audio_inference():
    async with local_gateway() as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.send_json({"type": "user_input", "text": "Typed primary text."})
        result = await receive_until(ws, "assistant_end")
        primary = next(m for m in result if m["type"] == "user_message")
        assert primary["from_text"] is True and primary["models"] == {} and primary["time"] == {"begin": 0, "end": 0}
        assert bytes(made[0].stt.pcm) == b""
        await ws.close()
        await assert_closed(gateway, made)


@pytest.mark.parametrize("mode,code", [("unscoped_stt", "unscoped_primary"), ("untagged_llm", "untagged_provider_output"), ("untagged_tts", "untagged_provider_output"), ("wrong_rate", "unsupported_tts_audio")])
async def test_ambiguous_provenance_and_wrong_tts_rate_fail_closed(mode, code):
    async with local_gateway(mode=mode) as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await audio(ws)
        result = await receive_until(ws, "error")
        assert result[-1]["code"] == code
        assert not any(m["type"] == "audio_output" for m in result)
        await ws.close()
        await assert_closed(gateway, made)


async def test_late_primary_transcript_cannot_be_assigned_to_new_pcm_scope():
    async with local_gateway(mode="late_stt") as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await audio(ws, 1)
        await asyncio.wait_for(made[0].held.wait(), 2)
        await audio(ws, 2)
        result = await receive_until(ws, "assistant_end")
        assert [m["message"]["content"] for m in result if m["type"] == "user_message"] == ["Primary 2."]
        assert next(m for m in result if m["type"] == "user_message")["time"] == {"begin": 20, "end": 40}
        made[0].release.set()
        await eventually(lambda: next(iter(gateway.sessions.values())).stale_frames >= 1)
        assert not any(m.get("content") == "Primary 1." for messages in made[0].llm.inputs for m in messages)
        await ws.close()
        await assert_closed(gateway, made)


async def test_interruption_drops_late_tagged_llm_output_and_reconnect_is_fresh():
    async with local_gateway(mode="late_llm") as (client, url, gateway, made, _):
        ws, first = await connect(client, url)
        await ws.send_json({"type": "user_input", "text": "First."})
        await asyncio.wait_for(made[0].held.wait(), 2)
        await ws.send_json({"type": "user_input", "text": "Second."})
        result = await receive_until(ws, "assistant_end")
        assert sum(m["type"] == "user_interruption" for m in result) == 1
        assert [m["message"]["content"] for m in result if m["type"] == "assistant_message"] == ["Synthetic reply 2."]
        assert len(set(made[0].tokens)) == 2
        made[0].release.set()
        await eventually(lambda: next(iter(gateway.sessions.values())).stale_frames >= 1)
        await ws.close()
        await assert_closed(gateway, made)
        again, second = await connect(client, url)
        assert first["chat_id"] != second["chat_id"] and first["chat_group_id"] != second["chat_group_id"]
        await again.close()
        await assert_closed(gateway, made)


async def test_tool_exact_id_continuation_and_duplicate_reply_refusal():
    async with local_gateway(mode="tool") as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.send_json({"type": "user_input", "text": "Order status."})
        result = await receive_until(ws, "tool_call")
        call = result[-1]
        assert not any(m["type"] == "assistant_end" for m in result)
        assert call["name"] == "lookup_demo_order" and call["tool_type"] == "function" and call["response_required"] is True
        assert json.loads(call["parameters"]) == {"order_id": "DEMO-100"}
        reply = {"type": "tool_response", "tool_call_id": call["tool_call_id"], "content": "Synthetic packed."}
        await ws.send_json(reply)
        result = await receive_until(ws, "assistant_end")
        assert any(m["type"] == "audio_output" for m in result)
        assert len(made[0].tokens) == 2 and len(set(made[0].tokens)) == 1
        assert "Synthetic packed." in json.dumps(made[0].llm.inputs[-1])
        await ws.send_json(reply)
        assert (await receive_until(ws, "error"))[-1]["code"] == "unknown_tool_response"
        await ws.close()
        await assert_closed(gateway, made)


async def test_old_tool_reply_after_interruption_cannot_continue_new_turn():
    async with local_gateway(mode="tool") as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.send_json({"type": "user_input", "text": "Order."})
        call = (await receive_until(ws, "tool_call"))[-1]
        await ws.send_json({"type": "user_input", "text": "Different request."})
        await receive_until(ws, "assistant_end")
        assert len(set(made[0].tokens)) == 2
        await ws.send_json({"type": "tool_response", "tool_call_id": call["tool_call_id"], "content": "late"})
        assert (await receive_until(ws, "error"))[-1]["code"] == "unknown_tool_response"
        assert not any("late" in json.dumps(messages) for messages in made[0].llm.inputs)
        await ws.close()
        await assert_closed(gateway, made)


async def test_tool_deadline_is_explicit_result_and_still_joined():
    async with local_gateway(mode="tool", limits=GatewayLimits(tool_seconds=0.1)) as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.send_json({"type": "user_input", "text": "Order."})
        await receive_until(ws, "tool_call")
        await receive_until(ws, "assistant_end")
        assert "tool_timeout" in json.dumps(made[0].llm.inputs[-1])
        await ws.close()
        await assert_closed(gateway, made)


@pytest.mark.parametrize("wire,code", [
    ('{"type":"user_input","text":"x","text":"y"}', "invalid_message"),
    ('{"type":"audio_input","data":"%%%"}', "invalid_audio"),
    ('{"type":"audio_input","data":"AQ=="}', "invalid_audio"),
    ('{"type":"tool_response","tool_call_id":"unknown","content":"x"}', "unknown_tool_response"),
    ('{"type":"session_settings","audio":{}}', "unsupported_message"),
])
async def test_bad_wire_stops_without_raw_payload_echo(wire, code):
    async with local_gateway() as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.send_str(wire)
        result = (await receive_until(ws, "error"))[-1]
        assert result["code"] == code and wire not in json.dumps(result) and KEY not in json.dumps(result)
        await ws.close()
        await assert_closed(gateway, made)


async def test_binary_webm_is_refused():
    async with local_gateway() as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.send_bytes(b"\x1aE\xdf\xa3synthetic-webm")
        assert (await receive_until(ws, "error"))[-1]["code"] == "text_json_required"
        await ws.close()
        await assert_closed(gateway, made)


async def test_turn_and_audio_total_limits_are_session_bounds():
    async with local_gateway(limits=GatewayLimits(turns=1, input_bytes=640)) as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await audio(ws)
        await receive_until(ws, "assistant_end")
        await audio(ws)
        assert (await receive_until(ws, "error"))[-1]["code"] == "audio_limit"
        await ws.close()
        await assert_closed(gateway, made)
    async with local_gateway(limits=GatewayLimits(turns=1)) as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        for n in (1, 2):
            await ws.send_json({"type": "user_input", "text": f"Turn {n}"})
            result = await receive_until(ws, "assistant_end" if n == 1 else "error")
        assert result[-1]["code"] == "turn_limit"
        await ws.close()
        await assert_closed(gateway, made)


async def test_session_deadline_and_capacity_release():
    async with local_gateway(limits=GatewayLimits(sessions=1, session_seconds=0.15)) as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        async with client.get(url, params={"api_key": KEY}) as response:
            assert response.status == 503
        assert (await receive_until(ws, "error"))[-1]["code"] == "deadline"
        await ws.close()
        await assert_closed(gateway, made)
        assert len(made) == 1


async def test_held_actual_send_hits_queue_bound_without_detached_writer(monkeypatch):
    original = web.WebSocketResponse.send_str
    entered = asyncio.Event()
    async def held_once(ws, data, *args, **kwargs):
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return await original(ws, data, *args, **kwargs)
    monkeypatch.setattr(web.WebSocketResponse, "send_str", held_once)
    async with local_gateway(limits=GatewayLimits(pending_output=1)) as (client, url, gateway, made, _):
        ws = await client.ws_connect(url, params={"api_key": KEY})
        await ws.send_json(SETTINGS)
        await asyncio.wait_for(entered.wait(), 2)
        await ws.send_json({"type": "user_input", "text": "Queue bound."})
        assert (await receive_until(ws, "error"))[-1]["code"] == "output_backpressure"
        await ws.close()
        await assert_closed(gateway, made)


async def test_repeated_handler_cancellation_retains_provider_cleanup_and_slot():
    gate = asyncio.Event()
    async with local_gateway(cleanup_gate=gate, limits=GatewayLimits(sessions=1)) as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        owner = next(iter(gateway.active.values()))
        owner.cancel()
        await asyncio.wait_for(made[0].closing.wait(), 2)
        owner.cancel()
        await asyncio.sleep(0)
        assert not owner.done() and gateway.active and not made[0].closed.is_set()
        async with client.get(url, params={"api_key": KEY}) as response:
            assert response.status == 503
        gate.set()
        await asyncio.gather(owner, return_exceptions=True)
        await ws.close()
        await assert_closed(gateway, made)


async def test_application_shutdown_joins_owned_session_and_late_provider_task():
    gate = asyncio.Event()
    async with local_gateway(mode="late_llm", cleanup_gate=gate) as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.send_json({"type": "user_input", "text": "Held."})
        await asyncio.wait_for(made[0].held.wait(), 2)
        shutdown = asyncio.create_task(gateway.shutdown(None))
        await asyncio.wait_for(made[0].closing.wait(), 2)
        assert not shutdown.done() and gateway.active
        gate.set()
        await asyncio.wait_for(shutdown, 3)
        await ws.close()
        await assert_closed(gateway, made)
        assert all(task.done() for task in made[0].background)


async def test_failed_cleanup_quarantines_new_admission_without_success_claim():
    async with local_gateway(mode="close_failure") as (client, url, gateway, made, _):
        ws, _ = await connect(client, url)
        await ws.close()
        await eventually(lambda: not gateway.active)
        assert gateway.quarantined and gateway.reports[-1]["status"] == "provider_cleanup_failed"
        assert gateway.reports[-1]["all_owned_tasks_done"] is True
        assert gateway.reports[-1]["provider_cleanup_joined"] is False
        async with client.get(url, params={"api_key": KEY}) as response:
            assert response.status == 503 and await response.text() == "provider_cleanup_unresolved"
        assert len(made) == 1


async def test_fresh_invalid_bundle_joins_acquired_finalizer_before_refusal():
    gate = asyncio.Event()
    async with local_gateway(mode="invalid_bundle", cleanup_gate=gate) as (client, url, gateway, made, _):
        ws = await client.ws_connect(url, params={"api_key": KEY})
        await ws.send_json(SETTINGS)
        await eventually(lambda: bool(made))
        await asyncio.wait_for(made[0].closing.wait(), 2)
        result = asyncio.create_task(receive_until(ws, "error"))
        await asyncio.sleep(0)
        assert gateway.active and not result.done() and not made[0].closed.is_set()
        gate.set()
        assert (await result)[-1]["code"] == "invalid_provider_bundle"
        await ws.close()
        await assert_closed(gateway, made)
        assert not gateway.quarantined


async def test_reused_active_bundle_quarantines_without_closing_original_owner():
    async with local_gateway(mode="reuse_bundle") as (client, url, gateway, made, _):
        first, _ = await connect(client, url)
        second = await client.ws_connect(url, params={"api_key": KEY})
        await second.send_json(SETTINGS)
        assert (await receive_until(second, "error"))[-1]["code"] == "provider_reuse"
        await second.close()
        await eventually(lambda: len(gateway.active) == 1)
        assert gateway.quarantined and not made[0].closing.is_set()
        assert gateway.reports[-1]["provider_cleanup_joined"] is False
        # The first connection still owns its pipeline; the illegal second
        # acquisition may not close or steal those existing processors.
        await first.send_json({"type": "user_input", "text": "Still owned."})
        await receive_until(first, "assistant_end")
        await first.close()
        await eventually(lambda: not gateway.active)
        assert made[0].closed.is_set() and gateway.reports[-1]["provider_cleanup_joined"] is True


async def test_throwing_factory_cannot_claim_no_acquisition_or_reopen_capacity():
    async with local_gateway(mode="factory_failure") as (client, url, gateway, made, _):
        ws = await client.ws_connect(url, params={"api_key": KEY})
        await ws.send_json(SETTINGS)
        assert (await receive_until(ws, "error"))[-1]["code"] == "session_failed"
        await ws.close()
        await eventually(lambda: not gateway.active)
        assert gateway.quarantined and gateway.reports[-1]["provider_cleanup_joined"] is False
        assert made == []
        async with client.get(url, params={"api_key": KEY}) as response:
            assert response.status == 503


async def test_terminal_gate_blocks_new_tagged_tts_work_but_forwards_teardown():
    # Focused processor composition, not a second WebSocket/inference claim.
    # The unchanged synthetic TTS implementation would begin synthesis if the
    # bridge hands it this correctly tagged text after failure or closure.
    for terminal in ("failed", "closed"):
        state = Session(None, GatewayLimits())
        state.active = "synthetic-owned-turn"
        if terminal == "failed":
            state.abort("synthetic_failure")
        else:
            state.closed = True
        bridge = ProviderBridge(state)
        tts = NamedTTS(SimpleNamespace(mode="normal"))
        emitted = []
        async def output(frame, direction=FrameDirection.DOWNSTREAM):
            emitted.append(frame)
        bridge.push_frame = tts.process_frame
        tts.push_frame = output
        late = LLMTextFrame("Must not start synthesis.")
        late.metadata[TURN] = state.active
        await bridge.process_frame(late, FrameDirection.DOWNSTREAM)
        assert tts.accepted_texts == [] and emitted == []
        end = EndFrame()
        await bridge.process_frame(end, FrameDirection.DOWNSTREAM)
        assert emitted == [end]
