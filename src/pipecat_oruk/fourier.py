"""English Fourier 2B recognition for Pipecat's segmented speech pipeline.

Supports two explicit modes:
- ``mode="local"`` (default): runs the pinned Q8_0 GGUF file locally on CPU via
  the optional ``pipecat-oruk[fourier]`` extra (CC BY-NC 4.0 weights).
- ``mode="hosted"``: sends mono 16 kHz PCM16 WAV utterances to Oruk's hosted
  ``POST https://speech-api.oruk.ai/v1/audio/transcriptions`` endpoint using
  ``model="oruk-fourier-2b"`` and ``ORUK_API_KEY`` (no extra packages required).
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import threading
import uuid
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import aiohttp
import numpy as np
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from .local import _LocalSTTService

MODEL_ID = "oruk/oruk-fourier-2b"
REVISION = "8bcce21695e4ba13569c48de6cedffcb8a7576f5"
FILENAME = "Fourier-2B-Q8_0.gguf"
SHA256 = "6a7c419bd571f7b0e21da2d5ab1e3a1a3938a83470d622be57427af98f322981"

HOSTED_MODEL_ID = "oruk-fourier-2b"
DEFAULT_HOSTED_BASE_URL = "https://speech-api.oruk.ai"
MAX_HOSTED_AUDIO_BYTES = 30 * 1024 * 1024 - 64 * 1024
_HOSTED_ENGLISH_CODES = {"en", "en-US", "en-GB"}


def _validate_hosted_base_url(base_url: str) -> str:
    """Require HTTPS (or loopback HTTP for local testing) with no credentials."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty URL.")
    cleaned = base_url.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("base_url must not contain credentials, query parameters, or fragments.")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise ValueError("base_url must use https (or http only for localhost/loopback).")
    return cleaned


def _normalize_hosted_language(language: Any) -> str:
    if language is None or language == Language.EN:
        return "en"
    if isinstance(language, str) and language in _HOSTED_ENGLISH_CODES:
        return language
    raise ValueError("Fourier uses a fixed English speech model (en, en-US, or en-GB).")


def _safe_error_code(raw: Any) -> str:
    if isinstance(raw, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", raw):
        return raw
    return "api_error"


def _split_audio(audio: np.ndarray, rate: int):
    """Prefer a quiet 20 ms boundary near the end of a long model window."""
    limit = 30 * rate
    frame = rate // 50
    while len(audio) > limit:
        start = limit - 2 * rate
        window = audio[start:limit].reshape(-1, frame)
        power = (window * window).mean(axis=1)
        boundary = start + (int(np.argmin(power)) + 1) * frame
        yield audio[:boundary]
        audio = audio[boundary:]
    if audio.size:
        yield audio


class _FourierRuntime:
    def __init__(self, cache_dir: str | None, local_files_only: bool):
        try:
            from huggingface_hub import hf_hub_download
            from transcribe_cpp import Model
        except ImportError as exc:
            raise ImportError(
                "Install the fourier extra to use local FourierSTTService."
            ) from exc
        self._download = hf_hub_download
        self._model_class = Model
        self._cache_dir = cache_dir
        self._offline = local_files_only
        self._model = None
        self._lock = threading.Lock()
        self._closed = False

    def _load(self):
        if self._closed:
            raise RuntimeError("Fourier STT is closed")
        if self._model is None:
            path = self._download(
                repo_id=MODEL_ID,
                revision=REVISION,
                filename=FILENAME,
                cache_dir=self._cache_dir,
                local_files_only=self._offline,
            )
            with Path(path).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                    raise ValueError("Fourier model checksum mismatch")
            self._model = self._model_class(str(path), backend="cpu")
        return self._model

    def prewarm(self):
        with self._lock:
            self._load()

    def recognize(self, audio: np.ndarray, sample_rate: int) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("Fourier STT is closed")
            if not audio.size:
                return ""
            if sample_rate != 16000:
                raise ValueError("Fourier requires 16 kHz input")
            model = self._load()
            parts = []
            for chunk in _split_audio(audio, sample_rate):
                with model.session() as session:
                    text = session.run(
                        chunk, language="en", timestamps="none"
                    ).text.strip()
                    if text:
                        parts.append(text)
            return " ".join(parts)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._model is not None:
                self._model.close()
                self._model = None


class _HostedFourierClient:
    """Async HTTP client for POST /v1/audio/transcriptions (model=oruk-fourier-2b)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._url = f"{base_url}/v1/audio/transcriptions"
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._external_session = session
        self._owned_session: aiohttp.ClientSession | None = None
        self._closed = False

    def _get_session(self) -> aiohttp.ClientSession:
        if self._external_session is not None:
            return self._external_session
        if self._owned_session is None or self._owned_session.closed:
            self._owned_session = aiohttp.ClientSession(timeout=self._timeout)
        return self._owned_session

    async def prewarm(self) -> None:
        if self._closed:
            raise RuntimeError("Fourier STT is closed")

    async def transcribe(self, audio: bytes, *, language: str = "en") -> str:
        if self._closed:
            raise RuntimeError("Fourier STT is closed")
        if len(audio) > MAX_HOSTED_AUDIO_BYTES:
            raise ValueError(
                "Hosted FourierSTTService utterance exceeds the 30 MiB request limit."
            )
        with wave.open(io.BytesIO(audio), "rb") as source:
            if (
                source.getframerate() != 16000
                or source.getnchannels() != 1
                or source.getsampwidth() != 2
            ):
                raise ValueError("FourierSTTService requires mono PCM16 at 16 kHz.")
            if source.getnframes() == 0:
                return ""

        request_id = str(uuid.uuid4())
        form = aiohttp.FormData()
        form.add_field("model", HOSTED_MODEL_ID)
        form.add_field("language", language)
        form.add_field("response_format", "json")
        form.add_field(
            "file",
            audio,
            filename="utterance.wav",
            content_type="audio/wav",
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Request-ID": request_id,
        }
        session = self._get_session()
        async with session.post(
            self._url,
            data=form,
            headers=headers,
            timeout=self._timeout,
        ) as response:
            resp_req_id = response.headers.get("X-Request-ID") or request_id
            if response.status != 200:
                code = "api_error"
                try:
                    payload = await response.json(content_type=None)
                except (ValueError, aiohttp.ContentTypeError, aiohttp.ClientError):
                    payload = None
                if isinstance(payload, dict):
                    err = (
                        payload.get("error")
                        if isinstance(payload.get("error"), dict)
                        else payload
                    )
                    code = _safe_error_code(err.get("code") or err.get("type"))
                raise RuntimeError(
                    f"Oruk Fourier API error {response.status}: {code} (request_id={resp_req_id})"
                )
            payload = await response.json(content_type=None)
            text = payload.get("text") if isinstance(payload, dict) else None
            if text is None or not isinstance(text, str):
                raise RuntimeError(
                    f"Oruk Fourier API returned an invalid response (request_id={resp_req_id})"
                )
            return text.strip()

    async def close(self) -> None:
        self._closed = True
        if self._owned_session is not None and not self._owned_session.closed:
            await self._owned_session.close()
        self._owned_session = None


class FourierSTTService(_LocalSTTService):
    """Produce final English text from mono 16 kHz speech segments with Fourier 2B.

    Args:
        mode: ``"local"`` (default) loads the pinned Q8_0 GGUF file on CPU via
            ``pipecat-oruk[fourier]`` (CC BY-NC 4.0 weights). ``"hosted"`` sends
            each utterance to ``https://speech-api.oruk.ai/v1/audio/transcriptions``
            with ``model="oruk-fourier-2b"`` using ``ORUK_API_KEY`` (no extra
            dependencies required).
        cache_dir: Optional Hugging Face cache directory (local mode only).
        local_files_only: Require cached weights offline (local mode only).
        api_key: Oruk API key (hosted mode only; defaults to ``ORUK_API_KEY``).
        base_url: Speech API base URL (hosted mode only; defaults to
            ``https://speech-api.oruk.ai``).
        timeout: Per-utterance HTTP timeout in seconds (hosted mode only).
        aiohttp_session: Optional caller-managed ``aiohttp.ClientSession``
            (hosted mode only).
        settings: Optional ``STTSettings`` pinned to English.
        **kwargs: Additional ``SegmentedSTTService`` options.
    """

    Settings = STTSettings
    transcript_language = Language.EN

    def __init__(
        self,
        *,
        mode: Literal["local", "hosted"] = "local",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        api_key: str | None = None,
        base_url: str = DEFAULT_HOSTED_BASE_URL,
        timeout: float = 60.0,
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: STTSettings | None = None,
        **kwargs: Any,
    ) -> None:
        if mode not in ("local", "hosted"):
            raise ValueError("mode must be 'local' or 'hosted'.")
        if kwargs.pop("sample_rate", 16000) != 16000:
            raise ValueError("FourierSTTService requires a 16 kHz input pipeline.")

        self._mode = mode
        self._hosted: _HostedFourierClient | None = None
        self._hosted_language = "en"

        if mode == "local":
            if api_key is not None or aiohttp_session is not None or base_url != DEFAULT_HOSTED_BASE_URL:
                raise ValueError("api_key, base_url, and aiohttp_session are only valid in hosted mode.")
            self._validate_local_settings(settings or STTSettings())
            super().__init__(
                sample_rate=16000,
                settings=STTSettings(model=MODEL_ID, language=Language.EN),
                **kwargs,
            )
            self._runtime = _FourierRuntime(cache_dir, local_files_only)
        else:
            if cache_dir is not None or local_files_only:
                raise ValueError("cache_dir and local_files_only are only valid in local mode.")
            if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be a positive finite number.")
            resolved_key = (api_key if api_key is not None else os.environ.get("ORUK_API_KEY", "")).strip()
            if not resolved_key:
                raise ValueError("Set ORUK_API_KEY or pass api_key for hosted FourierSTTService.")
            validated_url = _validate_hosted_base_url(base_url)
            self._hosted_language = self._validate_hosted_settings(settings or STTSettings())
            super().__init__(
                sample_rate=16000,
                settings=STTSettings(model=HOSTED_MODEL_ID, language=Language.EN),
                **kwargs,
            )
            self._hosted = _HostedFourierClient(
                api_key=resolved_key,
                base_url=validated_url,
                timeout=float(timeout),
                session=aiohttp_session,
            )

    @staticmethod
    def _validate_local_settings(settings: STTSettings) -> None:
        values = settings.given_fields()
        if (
            values.keys() - {"model", "language"}
            or values.get("model", MODEL_ID) != MODEL_ID
            or values.get("language", Language.EN) not in (Language.EN, "en")
        ):
            raise ValueError("Fourier uses a fixed English speech model.")

    @staticmethod
    def _validate_hosted_settings(settings: STTSettings) -> str:
        values = settings.given_fields()
        if (
            values.keys() - {"model", "language"}
            or values.get("model", HOSTED_MODEL_ID) != HOSTED_MODEL_ID
        ):
            raise ValueError("Hosted Fourier uses the fixed English oruk-fourier-2b model.")
        return _normalize_hosted_language(values.get("language", Language.EN))

    def _validate_settings(self, settings: STTSettings) -> None:  # type: ignore[override]
        if self._mode == "hosted":
            self._hosted_language = self._validate_hosted_settings(settings)
        else:
            self._validate_local_settings(settings)

    async def prewarm(self) -> None:
        """Prepare the local model or verify the hosted client is open."""
        if self._hosted is not None:
            await self._hosted.prewarm()
        else:
            await super().prewarm()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Recognize the WAV segment supplied by SegmentedSTTService."""
        if self._hosted is None:
            async for frame in super().run_stt(audio):
                yield frame
            return
        text = await self._hosted.transcribe(audio, language=self._hosted_language)
        if text:
            yield TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), language=self.transcript_language
            )

    async def cleanup(self) -> None:
        """Cancel pending framework work and release local or HTTP resources."""
        if self._hosted is None:
            await super().cleanup()
        else:
            await SegmentedSTTService.cleanup(self)
            await self._hosted.close()
