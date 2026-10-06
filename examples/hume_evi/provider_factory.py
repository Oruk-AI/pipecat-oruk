"""An explicit provider factory for the PCM EVI gateway: Oruk STT, OpenAI LLM/TTS.

This is one concrete replacement stack for Hume EVI built from `providers.py`:

- VAD: Silero through `ScopedVAD` (or an injected analyzer).
- STT: Oruk realtime, one turn per utterance, no connection retry.
- LLM: Pipecat's `OpenAILLMService` with turn tagging and no SDK retry.
- TTS: OpenAI-compatible speech, resampled to the gateway's 48 kHz output.

Settings are passed explicitly; ambient SDK custom headers are refused and
HTTP environment proxies are disabled. No listener is started.
Each awaited call of the returned factory builds a fresh bundle
whose `close` closes every client it created. Model, voice and endpoint
choices are the deployer's; none of them is qualified here.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_oruk.realtime import ENDPOINT, RealtimeOptions, new_http_session, validate_endpoint

from .config import Config
from .gateway import Providers, joined
from .providers import (
    ScopedSynthesisTTS,
    ScopedUtteranceSTT,
    ScopedVAD,
    TurnTaggedLLMMixin,
    openai_speech_synthesizer,
    oruk_realtime_transcriber,
)


class GatewayOpenAILLM(TurnTaggedLLMMixin, OpenAILLMService):
    """OpenAI LLM for the gateway: issued-turn tagging and no silent SDK retry."""

    def create_client(self, api_key=None, base_url=None, organization=None, project=None,
                      default_headers=None, **_kwargs):
        # The factory acquires the HTTP transport first so constructor failures
        # still have an owner. The pinned superclass calls this synchronously.
        return _explicit_openai_client(api_key, base_url, self._owned_transport)

    def __init__(self, *, owned_transport, **kwargs):
        self._owned_transport = owned_transport
        super().__init__(**kwargs)

    async def close_client(self):
        await self._client.close()


def _secret(value, name):
    if not isinstance(value, str) or not 8 <= len(value) <= 512 or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be a nonempty key without whitespace")


def _name(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be a short identifier")


def _seconds(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}] seconds")


def _validate_base_url(value):
    if value is None:
        return
    if not isinstance(value, str) or any(ord(c) <= 32 or ord(c) == 127 for c in value) or "\\" in value:
        raise ValueError("invalid_openai_base_url")
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        raise ValueError("invalid_openai_base_url") from None
    if (not url.hostname or url.username is not None or url.password is not None
            or url.query or url.fragment or (port is not None and not 1 <= port <= 65535)
            or not (url.scheme == "https" or (url.scheme == "http" and url.hostname == "127.0.0.1"))):
        raise ValueError("invalid_openai_base_url")


def _explicit_openai_client(key, base_url, transport):
    # openai 2.54.0 always reads OPENAI_CUSTOM_HEADERS, even with explicit
    # default_headers. Refuse it instead of mutating the process environment.
    if os.environ.get("OPENAI_CUSTOM_HEADERS"):
        raise ValueError("ambient_openai_custom_headers_refused")
    return AsyncOpenAI(api_key=key, admin_api_key="", organization="", project="",
                       webhook_secret="", base_url=base_url or "https://api.openai.com/v1",
                       http_client=transport, max_retries=0)


@dataclass(frozen=True)
class OrukOpenAISettings:
    """Explicit provider settings. Keys stay in memory; never log this object."""

    oruk_api_key: str
    openai_api_key: str
    llm_model: str
    tts_model: str
    tts_voice: str
    oruk_endpoint: str = ENDPOINT
    oruk_language: str = "auto"
    openai_base_url: str | None = None
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.6
    max_utterance_seconds: float = 30.0
    stt_timeout: float = 20.0
    tts_timeout: float = 20.0

    def __post_init__(self):
        _secret(self.oruk_api_key, "oruk_api_key")
        _secret(self.openai_api_key, "openai_api_key")
        for field in ("llm_model", "tts_model", "tts_voice"):
            _name(getattr(self, field), field)
        validate_endpoint(self.oruk_endpoint)
        RealtimeOptions(language=self.oruk_language)  # Validates the locale.
        _validate_base_url(self.openai_base_url)
        _seconds(self.vad_start_secs, "vad_start_secs", 0.05, 2)
        _seconds(self.vad_stop_secs, "vad_stop_secs", 0.1, 5)
        _seconds(self.max_utterance_seconds, "max_utterance_seconds", 1, 300)
        _seconds(self.stt_timeout, "stt_timeout", 0.1, 120)
        _seconds(self.tts_timeout, "tts_timeout", 0.1, 120)

    def __repr__(self):  # Never print keys.
        return f"OrukOpenAISettings(llm_model={self.llm_model!r}, tts_model={self.tts_model!r})"


def oruk_openai_providers(
    settings: OrukOpenAISettings,
    *,
    vad_analyzer: Callable[[VADParams], VADAnalyzer] | None = None,
) -> Callable[[Config], Awaitable[Providers]]:
    """Return a gateway provider factory that builds a fresh bundle per session.

    `vad_analyzer(params)` replaces the default Silero analyzer, for example in
    tests. Construction opens no connection; clients connect on first use and
    `close` closes all of them, reporting failure if any close fails.
    """
    if not isinstance(settings, OrukOpenAISettings):
        raise ValueError("explicit OrukOpenAISettings required")
    params = VADParams(start_secs=settings.vad_start_secs, stop_secs=settings.vad_stop_secs)

    def make_analyzer() -> VADAnalyzer:
        if vad_analyzer is not None:
            return vad_analyzer(params)
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        return SileroVADAnalyzer(params=params)

    async def factory(_config: Config) -> Providers:
        # Everything that can fail without I/O first; network clients last.
        if os.environ.get("OPENAI_CUSTOM_HEADERS"):
            raise ValueError("ambient_openai_custom_headers_refused")
        vad = ScopedVAD(make_analyzer(), max_seconds=settings.max_utterance_seconds)
        stack = AsyncExitStack()
        try:
            transports = []
            for _ in range(2):
                transport = httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=20.0)
                stack.push_async_callback(transport.aclose)
                transports.append(transport)
            llm = GatewayOpenAILLM(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                                   owned_transport=transports[0],
                                   settings=GatewayOpenAILLM.Settings(model=settings.llm_model))
            stack.push_async_callback(llm.close_client)
            speech = _explicit_openai_client(settings.openai_api_key, settings.openai_base_url, transports[1])
            stack.push_async_callback(speech.close)
            realtime = new_http_session()
            stack.push_async_callback(realtime.close)
            stt = ScopedUtteranceSTT(
                oruk_realtime_transcriber(realtime, api_key=settings.oruk_api_key, endpoint=settings.oruk_endpoint,
                                          options=RealtimeOptions(language=settings.oruk_language, phrase_emotions=False)),
                max_seconds=settings.max_utterance_seconds, timeout=settings.stt_timeout)
            tts = ScopedSynthesisTTS(openai_speech_synthesizer(speech, model=settings.tts_model, voice=settings.tts_voice),
                                     timeout=settings.tts_timeout)
        except BaseException:
            await joined(asyncio.create_task(stack.aclose()))
            raise

        async def close():
            await joined(asyncio.create_task(stack.aclose()))

        return Providers(stt, llm, tts, vad, close)

    return factory
