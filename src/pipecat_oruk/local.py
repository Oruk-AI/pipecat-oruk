"""Optional local Orukeet recognition for VAD-segmented utterances."""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from pipecat.frames.frames import Frame, AudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601

MODEL = "oruk/orukeet"


class OrukeetSTTService(SegmentedSTTService):
    """Transcribe complete utterances locally on CPU with Orukeet INT8.

    Feed mono PCM16 at 16 kHz after Pipecat's VADProcessor. This service emits
    final text only. It does not provide interim text, a detected language code,
    word timestamps, speaker labels, or the hosted API's phrase estimates.

    Args:
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Require cached weights instead of downloading them.
        settings: The model is fixed to oruk/orukeet, with automatic language
            detection. Other model or language values are rejected.
        **kwargs: Additional SegmentedSTTService options.
    """

    Settings = STTSettings

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        settings: STTSettings | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from ._local_model import Model
        except ImportError as exc:
            raise ImportError(
                "Install pipecat-oruk[local] to use OrukeetSTTService."
            ) from exc

        if kwargs.pop("sample_rate", 16000) != 16000:
            raise ValueError("OrukeetSTTService requires a 16 kHz input pipeline.")
        self._validate_settings(settings or STTSettings())
        super().__init__(
            sample_rate=16000,
            settings=STTSettings(model=MODEL, language=None),
            **kwargs,
        )
        self._runtime = Model(cache_dir=cache_dir, local_files_only=local_files_only)

    async def prewarm(self) -> None:
        """Download, verify and load the model without blocking the event loop."""
        await asyncio.to_thread(self._runtime.prewarm)

    @staticmethod
    def _validate_settings(settings: STTSettings) -> None:
        fields = settings.given_fields()
        if (
            fields.keys() - {"model", "language"}
            or fields.get("model", MODEL) != MODEL
            or fields.get("language") is not None
        ):
            raise ValueError(
                "Orukeet uses a fixed model and automatic language detection."
            )

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        try:
            self._validate_settings(delta)
        except ValueError as exc:
            await self.push_error(str(exc), category=ErrorCategory.APPLICATION)
            return {}
        return await super()._update_settings(delta)

    async def process_audio_frame(
        self, frame: AudioRawFrame, direction: FrameDirection
    ) -> None:
        """Reject mismatched audio before its metadata is replaced by a WAV header."""
        if (
            frame.sample_rate != 16000
            or frame.num_channels != 1
            or len(frame.audio) % 2
        ):
            raise ValueError("OrukeetSTTService requires mono PCM16 at 16 kHz.")
        await super().process_audio_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Recognize the WAV segment supplied by SegmentedSTTService."""
        with wave.open(io.BytesIO(audio), "rb") as source:
            if (
                source.getframerate() != 16000
                or source.getnchannels() != 1
                or source.getsampwidth() != 2
            ):
                raise ValueError("OrukeetSTTService requires mono PCM16 at 16 kHz.")
            pcm = source.readframes(source.getnframes())
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        text = await asyncio.to_thread(self._runtime.recognize, samples, 16000)
        if text:
            yield TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), language=None
            )

    async def cleanup(self) -> None:
        """Cancel pending framework work, then wait for native inference and release it."""
        await super().cleanup()
        await asyncio.to_thread(self._runtime.close)
