"""Real Pipecat/OpenAI-SDK adapters behind the gateway, over loopback only.

A local OpenAI-compatible fixture stands in for the providers: it is not a
model and qualifies no provider quality, latency, billing or voice rights. The
pinned Pipecat OpenAILLMService and the OpenAI SDK do the real request/stream
serialization, and the gateway applies its normal provenance checks.
"""

import asyncio
import json

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from examples.hume_evi import providers as adapters
from examples.hume_evi.gateway import AUDIO
from examples.hume_evi.providers import (
    ScopedSynthesisTTS,
    ScopedUtteranceSTT,
    ScopedVAD,
    openai_speech_synthesizer,
    openai_transcriber,
)
from hume_gateway_helpers import exact_loopback_only  # noqa: F401 - autouse fixture
from hume_provider_fakes import (
    CHUNK,
    SPEECH_SAMPLES,
    SYNTHETIC,
    VAD_PARAMS,
    EnergyVAD,
    adapter_gateway,
    chunk,
    closed,
    connect,
    receive_until,
    say,
    wav_frames,
)


async def test_real_adapters_scope_exact_utterances_and_return_bounded_48k_audio():
    async with adapter_gateway() as (client, url, gateway, made, fake):
        ws = await connect(client, url)
        stream = b""
        for marker in (1, 2):
            stream += await say(ws, marker)
            messages = await receive_until(ws, "assistant_end")
            user = [m for m in messages if m["type"] == "user_message"]
            assert [m["message"]["content"] for m in user] == [f"Utterance {marker}."]
            begin, end = user[0]["time"]["begin"], user[0]["time"]["end"]
            # The transcriber received exactly the reported span, nothing inferred.
            assert fake.transcriptions[-1]["pcm"] == stream[int(begin * 32):int(end * 32)]
            # The span may open one chunk early (the analyzer's carried partial frame),
            # but never inside the previous utterance and never after the speech.
            assert chunk(marker, speech=True) * 3 in fake.transcriptions[-1]["pcm"]
            assert len(fake.transcriptions[-1]["pcm"]) <= 8 * CHUNK * 2
            assistant = [m["message"]["content"] for m in messages if m["type"] == "assistant_message"]
            assert "".join(assistant) == f"Real reply {marker}. Second sentence here."
            audio = [m for m in messages if m["type"] == "audio_output"]
            frames = [wav_frames(m) for m in audio]
            assert all(0 < n <= 4_800 for n in frames)
            # Two sentences, each 150 ms at 24 kHz, resampled to 48 kHz.
            assert abs(sum(frames) - 2 * SPEECH_SAMPLES * 2) <= 200
            assert [m["index"] for m in audio] == list(range(len(audio)))
            assert len({m["id"] for m in audio + [m for m in messages if m["type"] == "assistant_message"]}) == 1
            assert messages[-1]["type"] == "assistant_end"
        assert [s["input"] for s in fake.speech] == ["Real reply 1.", "Second sentence here.",
                                                     "Real reply 2.", "Second sentence here."]
        assert {s["response_format"] for s in fake.speech} == {"pcm"}
        assert any(m.get("content") == "Utterance 1." for m in fake.chats[-1]["messages"])
        assert made[0].stt.requests == 2 and made[0].tts.requests == 4
        await ws.close()
        await closed(gateway, made)


async def test_single_provider_tool_call_is_reissued_with_gateway_tool_id():
    async with adapter_gateway() as (client, url, gateway, made, fake):
        ws = await connect(client, url)
        await say(ws, 7)
        messages = await receive_until(ws, "tool_call")
        call = messages[-1]
        assert call["tool_call_id"] != "call_provider_a" and json.loads(call["parameters"]) == {"order_id": "DEMO-100"}
        await ws.send_json({"type": "tool_response", "tool_call_id": call["tool_call_id"], "content": "packed"})
        messages = await receive_until(ws, "assistant_end")
        assert "".join(m["message"]["content"] for m in messages if m["type"] == "assistant_message") == "Your order is packed."
        # The provider saw one consistent ID: the issued one, never its own.
        history = fake.chats[-1]["messages"]
        issued = [c["id"] for m in history for c in m.get("tool_calls") or []]
        results = [m["tool_call_id"] for m in history if m["role"] == "tool"]
        assert issued == results == [call["tool_call_id"]]
        await ws.close()
        await closed(gateway, made)


@pytest.mark.parametrize("marker,fail,expected_requests", [(8, None, "chat"), (9, None, "chat"), (1, "stt", "stt"), (1, "tts", "tts"), (1, "chat", "chat")])
async def test_provider_failures_and_parallel_tool_calls_fail_closed(marker, fail, expected_requests):
    async with adapter_gateway() as (client, url, gateway, made, fake):
        if fail:
            fake.fail.add(fail)
        ws = await connect(client, url)
        await say(ws, marker)
        messages = await receive_until(ws, "error")
        assert messages[-1]["code"] == "provider_error"
        assert not any(m["type"] == "tool_call" for m in messages)
        assert {"chat": fake.chats, "stt": fake.transcriptions, "tts": fake.speech}[expected_requests]
        await ws.close()
        await closed(gateway, made)


async def test_typed_turn_uses_real_llm_and_tts_without_audio_inference():
    async with adapter_gateway() as (client, url, gateway, made, fake):
        ws = await connect(client, url)
        await ws.send_json({"type": "user_input", "text": "Typed primary text."})
        messages = await receive_until(ws, "assistant_end")
        assert any(m["type"] == "audio_output" for m in messages)
        assert fake.transcriptions == [] and made[0].stt.requests == 0
        await ws.close()
        await closed(gateway, made)


# ---- Unit checks of the adapters outside the gateway -----------------------------


async def drive(processor, steps):
    """Run frames through one processor; a callable step is awaited in between."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import WorkerRunner
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameProcessor

    seen = []

    class Sink(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            seen.append(frame)
            await self.push_frame(frame, direction)

    worker = PipelineWorker(Pipeline([processor, Sink()]), params=PipelineParams(audio_in_sample_rate=16_000),
                            enable_rtvi=False, cancel_on_idle_timeout=False)
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    for step in steps:
        if callable(step):
            await step()
        else:
            await worker.queue_frame(step)
    await worker.stop_when_done()
    await asyncio.wait_for(task, 5)
    return seen


def issued(epoch, seq, pcm, generation="g"):
    frame = InputAudioRawFrame(audio=pcm, sample_rate=16_000, num_channels=1)
    frame.metadata[AUDIO] = {"generation": generation, "epoch": epoch, "first": seq, "last": seq}
    return frame


def vad_spans(seen):
    return [("Started" if isinstance(f, VADUserStartedSpeakingFrame) else "Stopped",
             f.metadata[AUDIO]["epoch"], f.metadata[AUDIO]["first"], f.metadata[AUDIO]["last"])
            for f in seen if isinstance(f, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame))]


async def test_scoped_vad_reports_exact_spans_and_resets_on_epoch():
    frames = [issued(0, 1, chunk(1, speech=True)), issued(0, 2, chunk(1, speech=True))]
    frames += [issued(1, n, chunk(2, speech=True)) for n in (3, 4, 5)]  # Typed input retired epoch 0.
    frames += [issued(1, n, chunk(0, speech=False)) for n in range(6, 10)]
    spans = vad_spans(await drive(ScopedVAD(EnergyVAD(params=VAD_PARAMS)), frames))
    assert spans[0] == ("Started", 0, 1, 1)
    # Epoch 0 never receives a stop; epoch 1 starts fresh at its own first chunk.
    assert spans[1] == ("Started", 1, 3, 3)
    assert spans[2][:3] == ("Stopped", 1, 3) and 6 <= spans[2][3] <= 9
    assert len(spans) == 3


async def test_scoped_vad_attributes_onset_carried_across_small_chunks():
    small = lambda marker, speech: chunk(marker, speech=speech)[:640]  # 20 ms; analyzer frames are 32 ms.
    frames = [issued(0, n, small(3, True)) for n in range(1, 6)]
    frames += [issued(0, n, small(0, False)) for n in range(6, 20)]
    spans = vad_spans(await drive(ScopedVAD(EnergyVAD(params=VAD_PARAMS)), frames))
    # The first scored frame begins in chunk 1, so the utterance must include it.
    assert spans[0][:3] == ("Started", 0, 1) and spans[1][:3] == ("Stopped", 0, 1)


async def test_scoped_vad_refuses_overlong_speech_instead_of_splitting():
    from pipecat.frames.frames import ErrorFrame

    frames = [issued(0, n, chunk(4, speech=True)) for n in range(1, 8)]
    seen = await drive(ScopedVAD(EnergyVAD(params=VAD_PARAMS), max_seconds=0.3), frames)
    assert [f.error for f in seen if isinstance(f, ErrorFrame)] == ["vad_utterance_too_long"]
    assert vad_spans(seen) == [("Started", 0, 1, 1)]  # No split, no second start.


async def test_scoped_stt_interruption_cancels_unfinished_transcription():
    from pipecat.frames.frames import InterruptionFrame, TranscriptionFrame

    started, cancelled = asyncio.Event(), asyncio.Event()

    async def transcribe(_pcm, _rate):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    stop = VADUserStoppedSpeakingFrame()
    stop.metadata[AUDIO] = {"generation": "g", "epoch": 0, "first": 1, "last": 2}
    stt = ScopedUtteranceSTT(transcribe)
    seen = await drive(stt, [issued(0, 1, chunk(5, speech=True)), issued(0, 2, chunk(5, speech=True)), stop,
                             lambda: asyncio.wait_for(started.wait(), 2), InterruptionFrame(),
                             lambda: asyncio.wait_for(cancelled.wait(), 2)])
    assert stt.requests == 1 and not any(isinstance(f, TranscriptionFrame) for f in seen)


async def test_scoped_tts_graceful_end_speaks_accepted_text_and_keeps_response_end_order():
    from pipecat.frames.frames import EndFrame, LLMFullResponseEndFrame, LLMTextFrame, TTSAudioRawFrame
    from examples.hume_evi.gateway import TURN

    async def synthesize(text):
        for _ in range(3):  # 3 x 1200 samples at 24 kHz = 150 ms.
            yield (300).to_bytes(2, "little") * 1_200, 24_000

    def tagged(frame):
        frame.metadata[TURN] = "turn-1"
        return frame

    seen = await drive(ScopedSynthesisTTS(synthesize), [tagged(LLMTextFrame("Hello there. And")),
                                                         tagged(LLMTextFrame(" more")), tagged(LLMFullResponseEndFrame())])
    audio = [f for f in seen if isinstance(f, TTSAudioRawFrame)]
    order = [type(f).__name__ for f in seen if isinstance(f, (TTSAudioRawFrame, LLMFullResponseEndFrame, EndFrame))]
    assert order[-2:] == ["LLMFullResponseEndFrame", "EndFrame"] and order.count("LLMFullResponseEndFrame") == 1
    # Two sentences, each 3,600 input samples resampled to exactly 7,200 at 48 kHz.
    assert sum(len(f.audio) for f in audio) == 2 * 7_200 * 2
    assert all(f.sample_rate == 48_000 and len(f.audio) <= 9_600 and f.metadata[TURN] == "turn-1" for f in audio)


async def test_oruk_realtime_transcriber_sends_exact_span_once(monkeypatch):
    from pipecat_oruk import realtime

    calls = []

    async def run_turn(**kwargs):
        calls.append({**kwargs, "audio": [piece async for piece in kwargs["audio"]]})
        return realtime.TurnResult(request_id=kwargs["request_id"], transcript=" exact text ")

    monkeypatch.setattr(realtime, "run_turn", run_turn)
    transcribe = adapters.oruk_realtime_transcriber(object(), api_key=SYNTHETIC, endpoint="wss://api.oruk.ai/v1/realtime")
    pcm = bytes(range(256)) * 30
    assert await transcribe(pcm, 16_000) == " exact text "
    assert b"".join(calls[0]["audio"]) == pcm and all(len(p) <= 3_200 for p in calls[0]["audio"])
    assert calls[0]["max_connect_retries"] == 0 and calls[0]["options"].phrase_emotions is False
    with pytest.raises(ValueError, match="oruk_realtime_requires_16k"):
        await transcribe(pcm, 8_000)
    assert len(calls) == 1


async def test_adapter_bounds_refuse_invalid_configuration():
    with pytest.raises(ValueError):
        ScopedVAD(EnergyVAD(), max_seconds=0)
    with pytest.raises(ValueError):
        ScopedUtteranceSTT(openai_transcriber(None), max_seconds=True)
    with pytest.raises(ValueError):
        ScopedUtteranceSTT(openai_transcriber(None), timeout=0)
    with pytest.raises(ValueError):
        ScopedSynthesisTTS(openai_speech_synthesizer(None), max_chars=True)
