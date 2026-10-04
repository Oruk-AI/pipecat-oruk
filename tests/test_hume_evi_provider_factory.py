"""The explicit Oruk STT + OpenAI LLM/TTS factory, offline and over loopback.

The Oruk realtime and OpenAI-compatible endpoints are local fakes. These
tests check construction, ownership and wire behavior, never a model.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from examples.hume_evi.config import Config
from examples.hume_evi.gateway import Providers
from examples.hume_evi.provider_factory import OrukOpenAISettings, oruk_openai_providers
from examples.hume_evi.providers import ScopedSynthesisTTS, ScopedUtteranceSTT, ScopedVAD
from hume_gateway_helpers import ALLOWED_PORTS
from hume_gateway_helpers import exact_loopback_only  # noqa: F401 - autouse fixture
from hume_provider_fakes import (
    SYNTHETIC,
    VAD_PARAMS,
    EnergyVAD,
    adapter_gateway,
    closed,
    connect,
    receive_until,
    say,
    wav_frames,
)

ORUK_KEY = "synthetic-oruk-credential"
BASE = OrukOpenAISettings(oruk_api_key=ORUK_KEY, openai_api_key=SYNTHETIC, llm_model="synthetic-chat",
                          tts_model="synthetic-tts", tts_voice="synthetic")


@pytest.mark.parametrize("change", [
    {"oruk_api_key": ""}, {"openai_api_key": "has space in it"}, {"llm_model": ""},
    {"tts_voice": "two words"}, {"oruk_endpoint": "ws://speech-api.oruk.ai/v1/realtime"},
    {"oruk_endpoint": "wss://speech-api.oruk.ai/v1/realtime?key=secret"}, {"oruk_language": "!!"},
    {"openai_base_url": "http://example.com/v1"}, {"vad_stop_secs": 0}, {"max_utterance_seconds": True},
    {"stt_timeout": float("nan")},
])
def test_settings_refuse_invalid_values(change):
    with pytest.raises(ValueError):
        replace(BASE, **change)


def test_settings_repr_never_contains_keys():
    assert ORUK_KEY not in repr(BASE) and SYNTHETIC not in repr(BASE)
    with pytest.raises(ValueError, match="explicit"):
        oruk_openai_providers({"oruk_api_key": ORUK_KEY})


async def test_factory_builds_fresh_offline_bundles_with_silero_and_closes_every_client():
    factory = oruk_openai_providers(BASE)
    first, second = factory(Config()), factory(Config())
    for bundle in (first, second):
        assert isinstance(bundle, Providers)
        assert isinstance(bundle.vad, ScopedVAD) and isinstance(bundle.stt, ScopedUtteranceSTT)
        assert isinstance(bundle.tts, ScopedSynthesisTTS)
    processors = [p for b in (first, second) for p in (b.stt, b.llm, b.tts, b.vad)]
    assert len({id(p) for p in processors}) == 8
    clients = []
    for bundle in (first, second):
        realtime = bundle.stt._transcribe.__closure__  # The owned aiohttp session.
        session = next(c.cell_contents for c in realtime if type(c.cell_contents).__name__ == "ClientSession")
        speech = next(c.cell_contents for c in bundle.tts._synthesize.__closure__
                      if type(c.cell_contents).__name__ == "AsyncOpenAI")
        clients.append((session, speech, bundle.llm._client))
        await bundle.close()
    assert all(s.closed and o.is_closed() and llm.is_closed() for s, o, llm in clients)


async def test_oruk_stt_and_openai_llm_tts_stack_end_to_end(gateway):
    async with gateway() as oruk:
        port = urlsplit(oruk.endpoint).port
        ALLOWED_PORTS.add(port)
        try:
            def build(base, made):
                settings = replace(BASE, oruk_endpoint=oruk.endpoint, openai_base_url=base)
                inner = oruk_openai_providers(settings, vad_analyzer=lambda _params: EnergyVAD(params=VAD_PARAMS))

                def factory(config):
                    bundle = inner(config)
                    owner = SimpleNamespace(stt=bundle.stt, tts=bundle.tts, closed=asyncio.Event())
                    made.append(owner)

                    async def close():
                        await bundle.close()
                        owner.closed.set()

                    return Providers(bundle.stt, bundle.llm, bundle.tts, bundle.vad, close)
                return factory

            async with adapter_gateway(build) as (client, url, gateway_state, made, fake):
                ws = await connect(client, url)
                stream = await say(ws, 1)
                messages = await receive_until(ws, "assistant_end")
                user = [m for m in messages if m["type"] == "user_message"]
                assert [m["message"]["content"] for m in user] == ["Final turn 0."]
                begin, end = user[0]["time"]["begin"], user[0]["time"]["end"]
                # Oruk received exactly the scoped utterance once, with header auth only.
                assert oruk.requests == 1 and bytes(oruk.audio[0]) == stream[int(begin * 32):int(end * 32)]
                assert ORUK_KEY in (oruk.auth_headers[0] or "") and ORUK_KEY not in oruk.queries[0]
                assert oruk.configs[0]["phrase_emotions"] is False and oruk.configs[0]["sample_rate"] == 16_000
                text = "".join(m["message"]["content"] for m in messages if m["type"] == "assistant_message")
                assert text == "Real reply 1. Second sentence here."
                assert all(0 < wav_frames(m) <= 4_800 for m in messages if m["type"] == "audio_output")
                assert [s["voice"] for s in fake.speech] == ["synthetic", "synthetic"]
                assert fake.chats[-1]["model"] == "synthetic-chat"
                await ws.close()
                await closed(gateway_state, made)
        finally:
            ALLOWED_PORTS.discard(port)
