"""Shared loopback fakes for the real-provider adapter tests; never a model."""

import asyncio
import base64
import collections
import io
import json
import wave
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from openai import AsyncOpenAI
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.services.openai.llm import OpenAILLMService

from examples.hume_evi.gateway import GATEWAY, Providers, create_gateway
from examples.hume_evi.providers import (
    ScopedSynthesisTTS,
    ScopedUtteranceSTT,
    ScopedVAD,
    TurnTaggedLLMMixin,
    openai_speech_synthesizer,
    openai_transcriber,
)
from hume_gateway_helpers import ALLOWED_PORTS, KEY, SETTINGS
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
async def adapter_gateway(build=None):
    """Gateway on loopback with a fake OpenAI-compatible provider.

    `build(base_url, made)` may return a custom provider factory; the default
    wires the adapters to the fake directly.
    """
    fake = FakeProviders()
    fake_runner = web.AppRunner(fake.app, access_log=None, handle_signals=False)
    await fake_runner.setup()
    fake_site = web.TCPSite(fake_runner, "127.0.0.1", 0)
    await fake_site.start()
    fake_port = fake_site._server.sockets[0].getsockname()[1]
    ALLOWED_PORTS.add(fake_port)
    base = f"http://127.0.0.1:{fake_port}/v1"
    made = []

    def default_factory(_config):
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

    factory = default_factory if build is None else build(base, made)
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


