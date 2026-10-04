"""Real Pipecat/OpenAI-SDK adapters behind the gateway, over loopback only.

A local OpenAI-compatible fixture stands in for the providers: it is not a
model and qualifies no provider quality, latency, billing or voice rights. The
pinned Pipecat OpenAILLMService and the OpenAI SDK do the real request/stream
serialization, and the gateway applies its normal provenance checks.
"""

import asyncio
import base64
import collections
import io
import json
import wave
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from openai import AsyncOpenAI
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.openai.llm import OpenAILLMService

from examples.hume_evi import providers as adapters
from examples.hume_evi.gateway import AUDIO, GATEWAY, Providers, create_gateway
from examples.hume_evi.providers import (
    ScopedSynthesisTTS,
    ScopedUtteranceSTT,
    ScopedVAD,
    TurnTaggedLLMMixin,
    openai_speech_synthesizer,
    openai_transcriber,
)
from hume_gateway_helpers import ALLOWED_PORTS, KEY, SETTINGS
from hume_gateway_helpers import exact_loopback_only  # noqa: F401 - autouse fixture
from hume_helpers import eventually

SYNTHETIC = "synthetic-provider-credential"
CHUNK = 1_600  # 100 ms of 16 kHz input samples.
SPEECH_RATE = 24_000
SPEECH_SAMPLES = 3_600  # 150 ms per synthesized sentence.
VAD_PARAMS = VADParams(confidence=0.5, start_secs=0.06, stop_secs=0.2, min_volume=0.0)


class EnergyVAD(VADAnalyzer):
    """Deterministic amplitude detector driving Pipecat's real VAD state machine."""

    def num_frames_required(self) -> int:
        return 512

    def voice_confidence(self, buffer: bytes) -> float:
        samples = memoryview(buffer).cast("h")
        return 1.0 if max((abs(s) for s in samples), default=0) > 1_000 else 0.0


class GatewayLLM(TurnTaggedLLMMixin, OpenAILLMService):
    def create_client(self, api_key=None, base_url=None, **_kwargs):
        # No SDK retry and no environment proxy: one loopback request per inference.
        return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                           http_client=httpx.AsyncClient(trust_env=False))


def chunk(marker: int, *, speech: bool) -> bytes:
    value = 4_000 + marker if speech else 0
    return value.to_bytes(2, "little", signed=True) * CHUNK


def speech_text(marker: int) -> str:
    return {7: "Check order DEMO-100.", 8: "Check two orders.", 9: "Check broken order."}.get(marker, f"Utterance {marker}.")


class FakeProviders:
    """OpenAI-compatible loopback endpoints that record every request."""

    def __init__(self):
        self.transcriptions, self.speech, self.chats = [], [], []
        self.fail = set()
        self.app = web.Application()
        self.app.router.add_post("/v1/audio/transcriptions", self.transcribe)
        self.app.router.add_post("/v1/audio/speech", self.synthesize)
        self.app.router.add_post("/v1/chat/completions", self.chat)

    async def transcribe(self, request):
        form = await request.post()
        upload = form["file"]
        with wave.open(io.BytesIO(upload.file.read()), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16_000, 1, 2)
            pcm = wav.readframes(wav.getnframes())
        self.transcriptions.append({"model": form["model"], "pcm": pcm})
        if "stt" in self.fail:
            return web.json_response({"error": {"message": "synthetic"}}, status=500)
        samples = [s for s in memoryview(pcm).cast("h") if s]
        marker = collections.Counter(samples).most_common(1)[0][0] - 4_000
        return web.json_response({"text": speech_text(marker)})

    async def synthesize(self, request):
        body = await request.json()
        self.speech.append(body)
        if "tts" in self.fail:
            return web.json_response({"error": {"message": "synthetic"}}, status=500)
        pcm = (300).to_bytes(2, "little") * SPEECH_SAMPLES
        response = web.StreamResponse(headers={"Content-Type": "application/octet-stream"})
        await response.prepare(request)
        for start, end in ((0, 1_001), (1_001, 4_334), (4_334, len(pcm))):  # Odd splits.
            await response.write(pcm[start:end])
        await response.write_eof()
        return response

    async def chat(self, request):
        body = await request.json()
        self.chats.append(body)
        if "chat" in self.fail:
            return web.json_response({"error": {"message": "synthetic"}}, status=500)
        messages = body["messages"]
        last = messages[-1]
        if last["role"] == "tool":
            deltas = [{"role": "assistant", "content": "Your order "}, {"content": "is packed."}]
        elif "broken order" in str(last.get("content")):
            broken = self.call(0, "call_provider_a")
            broken["function"]["arguments"] = '{"order_id": "DEMO-'  # Truncated JSON.
            deltas = [{"role": "assistant", "tool_calls": [broken]}]
        elif "two orders" in str(last.get("content")):
            # One call per chunk, as the provider streams them.
            deltas = [{"role": "assistant", "tool_calls": [self.call(0, "call_provider_a")]},
                      {"tool_calls": [self.call(1, "call_provider_b")]}]
        elif "order" in str(last.get("content")):
            deltas = [{"role": "assistant", "tool_calls": [self.call(0, "call_provider_a")]}]
        else:
            n = sum(m["role"] == "user" for m in messages)
            deltas = [{"role": "assistant", "content": f"Real reply {n}. "}, {"content": "Second sentence here."}]
        finish = "tool_calls" if "tool_calls" in deltas[0] else "stop"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index, delta in enumerate(deltas):
            done = finish if index == len(deltas) - 1 else None
            payload = {"id": "chatcmpl-synthetic", "object": "chat.completion.chunk", "created": 0,
                       "model": body["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": done}]}
            await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    @staticmethod
    def call(index, call_id):
        return {"index": index, "id": call_id, "type": "function",
                "function": {"name": "lookup_demo_order", "arguments": json.dumps({"order_id": "DEMO-100"})}}


@asynccontextmanager
async def adapter_gateway():
    fake = FakeProviders()
    fake_runner = web.AppRunner(fake.app, access_log=None, handle_signals=False)
    await fake_runner.setup()
    fake_site = web.TCPSite(fake_runner, "127.0.0.1", 0)
    await fake_site.start()
    fake_port = fake_site._server.sockets[0].getsockname()[1]
    ALLOWED_PORTS.add(fake_port)
    base = f"http://127.0.0.1:{fake_port}/v1"
    made = []

    def factory(_config):
        client = AsyncOpenAI(api_key=SYNTHETIC, base_url=base, max_retries=0,
                             http_client=httpx.AsyncClient(trust_env=False))
        llm = GatewayLLM(api_key=SYNTHETIC, base_url=base, settings=GatewayLLM.Settings(model="synthetic-chat"))
        owner = SimpleNamespace(
            stt=ScopedUtteranceSTT(openai_transcriber(client, model="synthetic-stt")),
            llm=llm,
            tts=ScopedSynthesisTTS(openai_speech_synthesizer(client, model="synthetic-tts", voice="synthetic")),
            vad=ScopedVAD(EnergyVAD(params=VAD_PARAMS)),
            closed=asyncio.Event(),
        )

        async def close():
            await client.close()
            await llm._client.close()
            owner.closed.set()

        made.append(owner)
        return Providers(owner.stt, owner.llm, owner.tts, owner.vad, close)

    async def authenticate(key):
        return key == KEY

    app = create_gateway(enabled=True, authenticate=authenticate, providers=factory)
    runner = web.AppRunner(app, access_log=None, handle_signals=False, shutdown_timeout=5)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    ALLOWED_PORTS.add(port)
    try:
        async with ClientSession(timeout=ClientTimeout(total=10), trust_env=False) as client:
            yield client, f"http://127.0.0.1:{port}/v0/evi/chat", app[GATEWAY], made, fake
    finally:
        await runner.cleanup()
        await fake_runner.cleanup()
        ALLOWED_PORTS.discard(port)
        ALLOWED_PORTS.discard(fake_port)
        assert not app[GATEWAY].active
        assert all(owner.closed.is_set() for owner in made)


async def receive_until(ws, kind, *, timeout=8):
    result = []
    async with asyncio.timeout(timeout):
        while True:
            packet = await ws.receive()
            assert packet.type == WSMsgType.TEXT, ("unexpected_socket_terminal", packet.type)
            value = json.loads(packet.data)
            result.append(value)
            if value["type"] == kind:
                return result
            if kind != "error":
                assert value["type"] != "error", value.get("code")


async def connect(client, url):
    ws = await client.ws_connect(url, params={"api_key": KEY}, compress=0)
    await ws.send_json(SETTINGS)
    await receive_until(ws, "chat_metadata")
    return ws


async def say(ws, marker, *, speech=3, silence=4):
    sent = b""
    for index in range(speech + silence):
        pcm = chunk(marker, speech=index < speech)
        await ws.send_json({"type": "audio_input", "data": base64.b64encode(pcm).decode()})
        sent += pcm
        await asyncio.sleep(0)
    return sent


def wav_frames(message):
    with wave.open(io.BytesIO(base64.b64decode(message["data"], validate=True)), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (48_000, 1, 2)
        return wav.getnframes()


async def closed(gateway, made):
    await eventually(lambda: not gateway.active)
    assert all(r["all_owned_tasks_done"] and r["provider_cleanup_joined"] for r in gateway.reports)
    assert all(owner.closed.is_set() for owner in made)


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
