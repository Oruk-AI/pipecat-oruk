import asyncio

import pytest
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from examples.hume_evi.config import Config
from examples.hume_evi.pipeline import build_pipeline
from examples.hume_evi.policy import SCOPE_METADATA
from hume_helpers import (
    PassThrough,
    SyntheticLLM,
    SyntheticPrimarySTT,
    SyntheticTTS,
    SyntheticTransport,
    eventually,
    loopback_only,  # noqa: F401
)
from test_pipecat import audio, running

pytestmark = pytest.mark.usefixtures("loopback_only")


def build(
    *,
    config=None,
    endpoint=None,
    tool_args=None,
    playback_delay=0,
    tts_chunks=1,
    exact=False,
):
    transport = SyntheticTransport(delay=playback_delay)
    stt, llm, tts = (
        SyntheticPrimarySTT(),
        SyntheticLLM(tool_args),
        SyntheticTTS(tts_chunks),
    )
    session = build_pipeline(
        transport,
        stt,
        llm,
        tts,
        config=config or Config(greeting=""),
        vad=PassThrough(),
        endpoint=endpoint,
        api_key="fixture-key",
        scope_resolver=(lambda f: [f.metadata[SCOPE_METADATA]]) if exact else None,
    )
    assert session.stt is stt
    return session, transport, stt, llm, tts


async def turn(worker, stt, n=1):
    # Wait for prefix to reach the injected STT before announcing synthetic VAD.
    size = len(stt.pcm)
    await worker.queue_frame(audio(bytes([n, 0]) * 320))
    await eventually(lambda: len(stt.pcm) >= size + 640)
    await worker.queue_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await worker.queue_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))


async def test_greeting_two_turns_history_and_primary_provider_preserved(gateway):
    async with gateway() as server:
        session, transport, stt, llm, tts = build(
            config=Config(greeting="Synthetic greeting.", signals_enabled=True),
            endpoint=server.endpoint,
        )
        async with running([session.pipeline]) as (worker, _, _):
            await session.connected(worker)
            await session.connected(worker)
            await eventually(
                lambda: any(
                    m.get("content") == "Synthetic greeting."
                    for m in session.context.get_messages()
                )
            )
            await turn(worker, stt)
            await eventually(
                lambda: any(
                    m.get("content") == "Synthetic reply 1."
                    for m in session.context.get_messages()
                )
            )
            await turn(worker, stt, 2)
            await eventually(
                lambda: len(llm.inputs) == 2 and len(transport.outgoing.written) >= 3
            )
            assert tts.texts.count("Synthetic greeting.") == 1
            assert [
                m["content"] for m in llm.inputs[-1] if m.get("role") == "user"
            ] == ["Primary words one.", "Primary words two."]
            assert (
                sum(m.get("content") == "Synthetic reply 1." for m in llm.inputs[-1])
                == 1
            )
            assert all(
                "Final turn" not in str(messages) and "Estimates:" not in str(messages)
                for messages in llm.inputs
            )
            assert len(server.audio) == 2 and len(stt.finals) == 2
            assert session.stt is stt
        assert session.store.closed and session.tap.active_task.done()


@pytest.mark.parametrize(
    "mode", ["hang", "disconnect", "bad_score", "error_after_usage", "phrase_failure"]
)
async def test_optional_failure_keeps_primary_conversation_and_usage_uncertainty(
    gateway, mode
):
    async with gateway(mode=mode) as server:
        session, transport, stt, llm, _ = build(
            config=Config(
                greeting="",
                signals_enabled=True,
                signal_policy="exact_scope",
                signal_timeout=0.08,
            ),
            endpoint=server.endpoint,
            exact=True,
        )
        async with running([session.pipeline]) as (worker, _, _):
            await turn(worker, stt)
            await eventually(lambda: transport.outgoing.written)
            assert llm.inputs[0][-1]["content"] == "Primary words one."
            assert "Estimates:" not in str(llm.inputs)
            await eventually(
                lambda: session.tap.active_task is not None
                and session.tap.active_task.done()
            )
            assert server.requests == 1
            if mode == "error_after_usage":
                assert any(
                    t["usage"].get("billable_seconds") == 1
                    and t["status"] == "signal_failed_outcome_unknown"
                    for t in session.store.trace
                    if "usage" in t
                )
            elif mode != "phrase_failure":
                assert any("unknown" in t["status"] for t in session.store.trace)


async def test_ready_exact_scope_adds_note_and_next_typed_inference_expires_it(gateway):
    async with gateway() as server:
        session, transport, stt, llm, _ = build(
            config=Config(
                greeting="", signals_enabled=True, signal_policy="exact_scope"
            ),
            endpoint=server.endpoint,
            exact=True,
        )
        async with running([session.pipeline]) as (worker, _, _):
            await turn(worker, stt)
            await eventually(lambda: transport.outgoing.written)
            assert "Estimates:" in str(llm.inputs[0])
            assert "Final turn" not in str(llm.inputs[0])
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Typed question."}],
                    run_llm=True,
                )
            )
            await eventually(lambda: len(llm.inputs) == 2)
            assert "Estimates:" not in str(llm.inputs[1])
            assert llm.inputs[1][-1]["content"] == "Typed question."


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"order_id": "DEMO-100"}, "packed"),
        ({"order_id": "missing"}, "not_found"),
        ({"order_id": ["hostile"]}, "invalid_arguments"),
    ],
)
async def test_real_tool_dispatch_settles_once_and_continues(args, expected):
    session, transport, stt, llm, _ = build(tool_args=args)
    async with running([session.pipeline]) as (worker, down, _):
        await turn(worker, stt)
        await eventually(lambda: len(llm.inputs) == 2 and transport.outgoing.written)
        tool_messages = [
            m for m in session.context.get_messages() if m.get("role") == "tool"
        ]
        assert len(tool_messages) == 1 and expected in tool_messages[0]["content"]
        assert tool_messages[0]["tool_call_id"] == "synthetic-call-1"
        assert (
            sum(
                m.get("role") == "assistant" and bool(m.get("tool_calls"))
                for m in session.context.get_messages()
            )
            == 1
        )
        assert len(llm.inputs) == 2


async def test_real_tool_timeout_settles_and_rejects_late_result():
    session, transport, stt, llm, _ = build(tool_args={"order_id": "DEMO-100"})
    cancelled = asyncio.Event()
    callback = []

    async def slow(params):
        callback.append(params.result_callback)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    llm.register_function(
        "lookup_demo_order", slow, cancel_on_interruption=True, timeout_secs=0.03
    )
    async with running([session.pipeline]) as (worker, _, _):
        await turn(worker, stt)
        await asyncio.wait_for(cancelled.wait(), 3)
        await eventually(lambda: len(llm.inputs) == 2 and transport.outgoing.written)
        before = str(session.context.get_messages())
        await callback[0]({"late": "must_not_appear"})
        await asyncio.sleep(0.03)
        assert str(session.context.get_messages()) == before
        tools = [m for m in session.context.get_messages() if m.get("role") == "tool"]
        assert len(tools) == 1 and "cancel" in tools[0]["content"].lower()


async def test_tool_continuation_cannot_inherit_signal_note(gateway):
    async with gateway() as server:
        session, transport, stt, llm, _ = build(
            config=Config(
                greeting="", signals_enabled=True, signal_policy="exact_scope"
            ),
            endpoint=server.endpoint,
            exact=True,
            tool_args={"order_id": "DEMO-100"},
        )
        async with running([session.pipeline]) as (worker, _, _):
            await turn(worker, stt)
            await eventually(
                lambda: len(llm.inputs) == 2 and transport.outgoing.written
            )
            assert "Estimates:" in str(llm.inputs[0])
            assert "Estimates:" not in str(llm.inputs[1])
            assert "Estimates:" not in str(session.context.get_messages())


async def test_barge_in_flushes_real_playback_queue_and_next_turn_works():
    session, transport, stt, llm, _ = build(playback_delay=0.015, tts_chunks=40)
    async with running([session.pipeline]) as (worker, _, _):
        await turn(worker, stt)
        await eventually(lambda: len(transport.outgoing.written) >= 2)
        assert len(transport.outgoing.written) < 40
        await turn(worker, stt, 2)
        await eventually(
            lambda: any(pcm[:2] == b"\2\0" for pcm in transport.outgoing.written)
        )
        first_new = next(
            i for i, pcm in enumerate(transport.outgoing.written) if pcm[:2] == b"\2\0"
        )
        await eventually(lambda: len(transport.outgoing.written) >= first_new + 4)
        assert 0 < sum(pcm[:2] == b"\1\0" for pcm in transport.outgoing.written) < 40
        assert all(pcm[:2] != b"\1\0" for pcm in transport.outgoing.written[first_new:])
        assert len(llm.inputs) == 2 and len(stt.finals) == 2
        assert transport.outgoing.cancelled_writes >= 1


async def test_barge_in_cancels_tool_then_primary_conversation_continues():
    session, transport, stt, llm, _ = build(tool_args={"order_id": "DEMO-100"})
    started, cancelled = asyncio.Event(), asyncio.Event()
    callback = []

    async def pending(params):
        callback.append(params.result_callback)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    llm.register_function(
        "lookup_demo_order", pending, cancel_on_interruption=True, timeout_secs=3
    )
    async with running([session.pipeline]) as (worker, _, _):
        await turn(worker, stt)
        await asyncio.wait_for(started.wait(), 3)
        await turn(worker, stt, 2)
        await asyncio.wait_for(cancelled.wait(), 3)
        await eventually(lambda: len(llm.inputs) == 2 and transport.outgoing.written)
        before = str(session.context.get_messages())
        await callback[0]({"late_interrupted_tool": True})
        await asyncio.sleep(0.02)
        assert str(session.context.get_messages()) == before
        assert [m["content"] for m in llm.inputs[-1] if m.get("role") == "user"] == [
            "Primary words one.",
            "Primary words two.",
        ]
        assert (
            len([m for m in session.context.get_messages() if m.get("role") == "tool"])
            == 1
        )


async def test_cancel_inflight_tool_and_audio_then_reconnect_fresh_generation(gateway):
    async with gateway(mode="hang") as server:
        session, transport, stt, llm, _ = build(
            config=Config(greeting="", signals_enabled=True),
            endpoint=server.endpoint,
            tool_args={"order_id": "DEMO-100"},
        )
        started, cancelled = asyncio.Event(), asyncio.Event()
        callback = []

        async def pending(params):
            callback.append(params.result_callback)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        llm.register_function(
            "lookup_demo_order", pending, cancel_on_interruption=True, timeout_secs=3
        )
        async with running([session.pipeline]) as (worker, down, _):
            await turn(worker, stt)
            await asyncio.wait_for(server.first_audio.wait(), 3)
            await asyncio.wait_for(started.wait(), 3)
            old_ids = set(session.store.records)
            await session.disconnected(worker)
            await asyncio.wait_for(cancelled.wait(), 3)
            await asyncio.wait_for(server.closed.wait(), 3)
            await eventually(lambda: session.store.closed)
            context_after_cancel = str(session.context.get_messages())
            await callback[0]({"late_old_generation": True})
            await asyncio.sleep(0.02)
            assert str(session.context.get_messages()) == context_after_cancel
            assert not transport.outgoing.written
        assert session.tap.active_task.done() and session.tap._session is None
        assert any("unknown" in t["status"] for t in session.store.trace)
        server.mode = "normal"
        new, playback, new_stt, new_llm, _ = build(
            config=Config(greeting="", signals_enabled=True), endpoint=server.endpoint
        )
        assert new.store.generation != session.store.generation and new_stt is not stt
        async with running([new.pipeline]) as (worker, _, _):
            await turn(worker, new_stt, 2)
            await eventually(lambda: playback.outgoing.written)
            assert not old_ids.intersection(new.store.records)
            assert len(new_llm.inputs) == 1 and "late_old_generation" not in str(
                new_llm.inputs
            )
        assert server.requests == 2


async def test_disconnect_cancels_already_queued_tts_playback():
    session, transport, stt, llm, _ = build(playback_delay=0.015, tts_chunks=40)
    async with running([session.pipeline]) as (worker, _, _):
        await turn(worker, stt)
        await eventually(lambda: len(transport.outgoing.written) >= 2)
        await session.disconnected(worker)
        await eventually(lambda: session.store.closed)
        await asyncio.sleep(0.02)
        count = len(transport.outgoing.written)
        assert 0 < count < 40
        await asyncio.sleep(0.05)
        assert len(transport.outgoing.written) == count and len(llm.inputs) == 1


async def test_default_bundled_vad_drives_injected_primary_stt_offline():
    import wave
    from test_pipecat import SPEECH_FIXTURE, chunks

    transport = SyntheticTransport()
    stt, llm, tts = SyntheticPrimarySTT(), SyntheticLLM(), SyntheticTTS()
    session = build_pipeline(transport, stt, llm, tts, config=Config(greeting=""))
    with wave.open(str(SPEECH_FIXTURE), "rb") as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1
        pcm = source.readframes(source.getnframes()) + b"\0\0" * 16000
    async with running([session.pipeline]) as (worker, _, _):
        await worker.queue_frames(chunks(pcm))
        await eventually(lambda: transport.outgoing.written)
        assert llm.inputs[0][-1]["content"] == "Primary words one."
        assert bytes(stt.pcm) == pcm
        assert len(stt.finals) == 1 and session.tap.active_task is None
