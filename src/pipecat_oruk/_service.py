"""Streaming Oruk STT and independently timed phrase emotions for Pipecat 1.8.

Place a Pipecat VADProcessor before this service, or select ``vad=False`` and
queue OrukCommitFrame after each utterance. An Oruk commit closes its WebSocket.
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import aiohttp
import numpy as np
import soxr
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    DataFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    STTMuteFrame,
    SystemFrame,
    TranscriptionFrame,
    UninterruptibleFrame,
    UserAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsage, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory, classify_http_status_code

from .realtime import (
    BYTES_PER_SECOND,
    EMOTION,
    ENDPOINT,
    MODEL,
    SAMPLE_RATE,
    TRANSCRIPT,
    RealtimeError,
    RealtimeOptions,
    TurnResult,
    new_http_session,
    run_turn,
    validate_endpoint,
)


@dataclass
class OrukCommitFrame(SystemFrame):
    """Finish the audio accepted so far; ordered with incoming audio system frames."""


@dataclass
class OrukEventFrame(DataFrame, UninterruptibleFrame):
    """Provider event with identities and its start on the incoming audio timeline.

    Speaker IDs and phrase timestamps are local to this turn. These data frames
    survive assistant interruptions; cancellation still stops the pipeline.
    """

    stream_id: str
    turn_id: str
    user_id: str
    audio_offset: float
    event: dict[str, Any]


@dataclass
class OrukPhraseEmotionFrame(OrukEventFrame):
    """A completed or failed phrase estimate, separate from transcript text."""


@dataclass
class OrukSpeakerBoundaryFrame(OrukEventFrame):
    """An optional provider speaker-start or speaker-end event."""


@dataclass
class OrukTurnCompletedFrame(DataFrame, UninterruptibleFrame):
    """A turn with final text, usage and a successful WebSocket close."""

    stream_id: str
    turn_id: str
    user_id: str
    audio_offset: float
    result: TurnResult


@dataclass
class _Turn:
    turn_id: str
    user_id: str
    source: str | None
    offset: float
    options: RealtimeOptions
    audio: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    input_samples: int = 0
    pcm_bytes: int = 0
    interim: str = ""
    phrases: list[dict[str, Any]] = field(default_factory=list)
    resampler: Any = None
    speech_end: float | None = None


def _finite(value: float, name: str, maximum: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be positive and at most {maximum}")


class OrukSTTService(STTService):
    """One streamed connection per utterance, preserving phrase identity and audio.

    Use one service per input track. Pipeline input must be mono PCM16 at a fixed
    8–96 kHz. Other rates are converted to 16 kHz by a streaming VHQ SoX resampler;
    16 kHz input is sent unchanged. ``vad=True`` requires an upstream VADProcessor.
    The initial pre-roll retains audio already forwarded before VAD announces speech.

    ``wait_for_emotions`` delays the final TranscriptionFrame until the turn closes,
    so its metadata includes late phrase events. It does not delay interim text.
    Failure after audio delivery never replays the utterance: the service becomes
    unusable, letting Pipecat's configured unusable-processor policy take effect.
    """

    Settings = STTSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        options: RealtimeOptions | None = None,
        settings: STTSettings | None = None,
        vad: bool = True,
        pre_roll_seconds: float = 0.5,
        max_buffer_seconds: float = 30,
        max_pending_turns: int = 32,
        max_pending_events: int = 128,
        wait_for_emotions: bool = False,
        shutdown_timeout: float = 30,
        event_handler_timeout: float = 2,
        endpoint: str = ENDPOINT,
        connect_timeout: float = 10,
        max_connect_retries: int = 3,
        retry_interval: float = 2,
        **kwargs: Any,
    ):
        self._api_key = api_key if api_key is not None else os.environ.get("ORUK_API_KEY", "")
        if not self._api_key or any(c.isspace() for c in self._api_key):
            raise ValueError("Provide a nonempty ORUK_API_KEY without whitespace")
        validate_endpoint(endpoint)
        for name, value, maximum in (
            ("pre_roll_seconds", pre_roll_seconds, 5),
            ("max_buffer_seconds", max_buffer_seconds, 120),
            ("shutdown_timeout", shutdown_timeout, 300),
            ("event_handler_timeout", event_handler_timeout, 30),
            ("connect_timeout", connect_timeout, 120),
        ):
            _finite(value, name, maximum)
        for name, value in (
            ("max_pending_turns", max_pending_turns),
            ("max_pending_events", max_pending_events),
        ):
            if type(value) is not int or not 1 <= value <= 1024:
                raise ValueError(f"{name} must be an integer in [1, 1024]")
        if type(max_connect_retries) is not int or not 0 <= max_connect_retries <= 10:
            raise ValueError("max_connect_retries must be an integer in [0, 10]")
        if (
            isinstance(retry_interval, bool)
            or not math.isfinite(retry_interval)
            or retry_interval < 0
        ):
            raise ValueError("retry_interval must be finite and nonnegative")
        if type(vad) is not bool or type(wait_for_emotions) is not bool:
            raise ValueError("vad and wait_for_emotions must be bools")
        # Keepalive audio would create artificial, potentially billable turns.
        if kwargs.get("keepalive_timeout") is not None:
            raise ValueError("Oruk opens connections per turn; audio keepalive is unsupported")
        self._options = options or RealtimeOptions()
        if settings is not None:
            values = settings.given_fields()
            if values.keys() - {"model", "language"} or values.get("model", MODEL) != MODEL:
                raise ValueError("Only the language of oruk-realtime can be configured in Settings")
            language = values.get("language", self._options.language)
            if isinstance(language, Language):
                language = language.value
            self._options = replace(self._options, language=language)
        super().__init__(
            settings=STTSettings(model=MODEL, language=self._options.language), **kwargs
        )
        self._vad = vad
        self._pre_roll_seconds = pre_roll_seconds
        self._buffer_limit = int(max_buffer_seconds * BYTES_PER_SECOND)
        self._max_pending_turns = max_pending_turns
        self._max_pending_events = max_pending_events
        self._wait_for_emotions = wait_for_emotions
        self._shutdown_timeout = shutdown_timeout
        self._event_handler_timeout = event_handler_timeout
        self._endpoint = endpoint
        self._connect_timeout = connect_timeout
        self._max_connect_retries = max_connect_retries
        self._retry_interval = retry_interval
        self.stream_id = uuid.uuid4().hex
        self._http: aiohttp.ClientSession | None = None
        self._worker: asyncio.Task | None = None
        self._turns: asyncio.Queue[_Turn | None] = asyncio.Queue()
        self._current: _Turn | None = None
        self._prefix = bytearray()
        self._prefix_identity: tuple[str, str | None] | None = None
        self._input_samples = 0
        self._last_chunk_seconds = 0.0
        self._queued_bytes = 0
        self._pending_turns = 0
        self._speech_active = False
        self._accepting = False
        self._closed = False
        self._failed = False
        self._ttfb_reported = False
        for name in ("on_phrase_emotion", "on_speaker_boundary", "on_turn_completed"):
            # Await handlers in order. Do not spawn an unbounded task per event.
            self._register_event_handler(name, sync=True)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str:
        return language.value

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        if type(self.sample_rate) is not int or not 8_000 <= self.sample_rate <= 96_000:
            raise ValueError("Oruk requires pipeline input between 8000 and 96000 Hz")

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        if self._closed:
            raise RuntimeError("Create a new OrukSTTService for a new pipeline")
        if self._worker is None:
            self._http = new_http_session()
            self._accepting = True
            self._worker = self.create_task(self._work(), name="oruk_turns")

    async def stop(self, frame: EndFrame) -> None:
        if not self._closed:
            self._accepting = False
            try:
                self._commit()
                self._turns.put_nowait(None)
                if self._worker is not None:
                    async with asyncio.timeout(self._shutdown_timeout):
                        await asyncio.shield(self._worker)
            except TimeoutError:
                await self._fail(RealtimeError("shutdown_timeout"))
            except RealtimeError as exc:
                await self._fail(exc)
            finally:
                await self._shutdown()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        await self._shutdown()
        await super().cancel(frame)

    async def cleanup(self) -> None:
        await self._shutdown()
        await super().cleanup()

    async def _shutdown(self) -> None:
        self._closed = True
        self._accepting = False
        worker, self._worker = self._worker, None
        if worker is not None and worker is not asyncio.current_task():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        session, self._http = self._http, None
        if session is not None:
            await session.close()
        self._current = None
        self._prefix.clear()
        while not self._turns.empty():
            self._turns.get_nowait()
        self._queued_bytes = 0
        self._pending_turns = 0

    async def _fail(self, error: RealtimeError) -> None:
        if self._failed:
            return
        self._failed = True
        await self._shutdown()
        category = (
            classify_http_status_code(error.status)
            if error.status is not None
            else ErrorCategory.UNKNOWN
        )
        await self.push_error(
            f"Oruk turn failed: {error.code}. Audio was not replayed; restart this service.",
            category=category,
            force_treat_as_permanent=True,
        )

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        fields = delta.given_fields()
        try:
            if self._current is not None or self._closed:
                raise ValueError("Change settings between turns on an open service")
            if fields.keys() - {"model", "language"} or fields.get("model", MODEL) != MODEL:
                raise ValueError("Only the language of oruk-realtime can be updated")
            language = fields.get("language", self._options.language)
            if isinstance(language, Language):
                language = language.value
            options = replace(self._options, language=language)
        except (TypeError, ValueError) as exc:
            await self.push_error(str(exc), category=ErrorCategory.APPLICATION)
            return {}
        changed = await super()._update_settings(delta)
        self._options = options
        return changed

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if direction == FrameDirection.UPSTREAM and isinstance(
            frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
        ):
            # Our VAD is upstream. A downstream broadcaster must not commit audio
            # a second time or reopen a turn from an upstream copy of its event.
            await FrameProcessor.process_frame(self, frame, direction)
            await self.push_frame(frame, direction)
            return
        if direction == FrameDirection.DOWNSTREAM and self._accepting:
            try:
                if isinstance(frame, OrukCommitFrame):
                    self._commit()
                elif isinstance(frame, STTMuteFrame) and frame.mute:
                    self._commit()
                    self._prefix.clear()
                    self._speech_active = False
            except RealtimeError as exc:
                await self._fail(exc)
        await super().process_frame(frame, direction)

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection) -> None:
        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, InputAudioRawFrame):
            return
        if not self._accepting or not self.is_usable:
            return
        try:
            if frame.sample_rate != self.sample_rate or frame.num_channels != 1:
                raise RealtimeError("input_format_changed_or_not_mono")
            if not isinstance(frame.audio, bytes) or len(frame.audio) % 2:
                raise RealtimeError("invalid_pcm16")
            if len(frame.audio) > self.sample_rate * 2:
                raise RealtimeError("audio_frame_exceeds_one_second")
            if not frame.audio:
                return
            user_id = frame.user_id if isinstance(frame, UserAudioRawFrame) else ""
            identity = (user_id, frame.transport_source)
            offset = self._input_samples / self.sample_rate
            self._input_samples += len(frame.audio) // 2
            self._last_chunk_seconds = len(frame.audio) / (2 * self.sample_rate)
            if self._muted:
                return
            self._last_audio_time = time.monotonic()
            if self._vad and not self._speech_active:
                if identity != self._prefix_identity:
                    self._prefix.clear()
                    self._prefix_identity = identity
                self._prefix.extend(frame.audio)
                limit = int(self._pre_roll_seconds * self.sample_rate) * 2
                if len(self._prefix) > limit:
                    del self._prefix[: len(self._prefix) - limit]
                return
            if self._current is None:
                self._begin(user_id, frame.transport_source, offset)
            assert self._current is not None
            if identity != (self._current.user_id, self._current.source):
                raise RealtimeError("multiple_input_tracks_in_one_turn")
            self._feed(frame.audio)
        except RealtimeError as exc:
            await self._fail(exc)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # The normal framework path above retains track identity and validates format.
        await self.process_audio_frame(
            InputAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        yield None

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame) -> None:
        await super()._handle_vad_user_started_speaking(frame)
        if not self._vad or not self._accepting or self._muted or self._speech_active:
            return
        try:
            if (
                not math.isfinite(frame.start_secs)
                or frame.start_secs < 0
                or frame.start_secs + self._last_chunk_seconds > self._pre_roll_seconds
            ):
                raise RealtimeError("pre_roll_shorter_than_vad_confirmation")
            self._speech_active = True
            if self._prefix:
                user_id, source = self._prefix_identity or ("", None)
                offset = (self._input_samples - len(self._prefix) // 2) / self.sample_rate
                self._begin(user_id, source, offset)
                self._feed(bytes(self._prefix))
                self._prefix.clear()
        except RealtimeError as exc:
            await self._fail(exc)

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame) -> None:
        # Keep per-turn timing: a new VAD start can arrive before the old final text.
        self._user_speaking = False
        if not self._vad or not self._accepting:
            return
        try:
            if self._current and frame.stop_secs > 0 and math.isfinite(frame.timestamp):
                self._current.speech_end = frame.timestamp - frame.stop_secs
            self._commit()
        except RealtimeError as exc:
            await self._fail(exc)

    def _begin(self, user_id: str, source: str | None, offset: float) -> None:
        if self._pending_turns >= self._max_pending_turns:
            raise RealtimeError("too_many_pending_turns")
        turn = _Turn(uuid.uuid4().hex, user_id, source, offset, self._options)
        if self.sample_rate != SAMPLE_RATE:
            turn.resampler = soxr.ResampleStream(
                self.sample_rate, SAMPLE_RATE, 1, dtype="int16", quality="VHQ"
            )
        self._current = turn
        self._pending_turns += 1
        self._turns.put_nowait(turn)

    def _feed(self, audio: bytes) -> None:
        assert self._current is not None
        turn = self._current
        turn.input_samples += len(audio) // 2
        if turn.input_samples > turn.options.max_turn_seconds * self.sample_rate:
            raise RealtimeError("turn_too_long")
        if turn.resampler is not None:
            audio = turn.resampler.resample_chunk(np.frombuffer(audio, dtype="<i2")).tobytes()
        self._queue_pcm(turn, audio)

    def _queue_pcm(self, turn: _Turn, audio: bytes) -> None:
        if not audio:
            return
        if self._queued_bytes + len(audio) > self._buffer_limit:
            raise RealtimeError("audio_buffer_overflow")
        self._queued_bytes += len(audio)
        turn.pcm_bytes += len(audio)
        turn.audio.put_nowait(audio)

    def _commit(self) -> None:
        turn = self._current
        self._speech_active = False
        if turn is None:
            return
        if turn.resampler is not None:
            tail = turn.resampler.resample_chunk(np.array([], dtype=np.int16), last=True)
            self._queue_pcm(turn, tail.tobytes())
            turn.resampler = None
        turn.audio.put_nowait(None)
        self._current = None

    async def _audio(self, turn: _Turn) -> AsyncGenerator[bytes, None]:
        while True:
            audio = await turn.audio.get()
            if audio is None:
                return
            self._queued_bytes -= len(audio)
            yield audio

    async def _work(self) -> None:
        try:
            while (turn := await self._turns.get()) is not None:
                await self._run(turn)
                self._pending_turns -= 1
        except RealtimeError as exc:
            await self._fail(exc)
        except Exception:
            # Arbitrary errors can contain credentials, audio, transcripts or URLs.
            await self._fail(RealtimeError("adapter_error"))

    async def _run(self, turn: _Turn) -> None:
        events: asyncio.Queue[dict[str, Any] | TurnResult | RealtimeError] = asyncio.Queue(
            maxsize=self._max_pending_events
        )

        def on_event(event: dict[str, Any]) -> None:
            try:
                events.put_nowait(event)
            except asyncio.QueueFull:
                raise RealtimeError("provider_event_buffer_overflow") from None

        async def produce() -> None:
            assert self._http is not None
            try:
                result = await run_turn(
                    session=self._http,
                    api_key=self._api_key,
                    endpoint=self._endpoint,
                    audio=self._audio(turn),
                    request_id=turn.turn_id,
                    options=turn.options,
                    on_event=on_event,
                    connect_timeout=self._connect_timeout,
                    max_connect_retries=self._max_connect_retries,
                    retry_interval=self._retry_interval,
                )
            except RealtimeError as exc:
                await events.put(exc)
            except Exception:
                await events.put(RealtimeError("adapter_transport_error"))
            else:
                await events.put(result)

        producer = asyncio.create_task(produce(), name="oruk_pipecat_transport")
        try:
            while True:
                event = await events.get()
                if isinstance(event, RealtimeError):
                    raise event
                try:
                    async with asyncio.timeout(self._event_handler_timeout):
                        if isinstance(event, TurnResult):
                            await self._complete(turn, event)
                            return
                        await self._event(turn, event)
                except TimeoutError:
                    raise RealtimeError("event_handler_timeout") from None
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    def _metadata(self, turn: _Turn) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "turn_id": turn.turn_id,
            "audio_offset": turn.offset,
            "model": MODEL,
            "phrases": copy.deepcopy(
                sorted(turn.phrases, key=lambda p: (p["start"], p["phrase_id"]))
            ),
        }

    async def _final(self, turn: _Turn, event: dict[str, Any]) -> None:
        try:
            language = Language(turn.options.language)
        except ValueError:
            language = None  # 'auto' is not evidence of a detected language.
        frame = TranscriptionFrame(
            text=event["transcript"],
            user_id=turn.user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            language=language,
            result=copy.deepcopy(event),
            finalized=True,
        )
        frame.metadata["oruk"] = self._metadata(turn)
        if (
            turn.speech_end is not None
            and self.metrics_enabled
            and (not self.report_only_initial_ttfb or not self._ttfb_reported)
        ):
            latency = time.time() - turn.speech_end
            if latency >= 0:
                metric = MetricsFrame(
                    data=[TTFBMetricsData(processor=self.name, model=MODEL, value=latency)]
                )
                metric.metadata["oruk"] = {"turn_id": turn.turn_id}
                await self.push_frame(metric)
                self._ttfb_reported = True
        await self.push_frame(frame)

    async def _event(self, turn: _Turn, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == TRANSCRIPT + "delta":
            turn.interim += event["delta"]
            if len(turn.interim) > 128_000:
                raise RealtimeError("transcript_too_long")
            frame = InterimTranscriptionFrame(
                text=turn.interim,
                user_id=turn.user_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                result=copy.deepcopy(event),
            )
            frame.metadata["oruk"] = self._metadata(turn)
            await self.push_frame(frame)
        elif kind == TRANSCRIPT + "completed":
            if not self._wait_for_emotions:
                await self._final(turn, event)
        elif kind in (EMOTION + "completed", EMOTION + "failed"):
            turn.phrases.append(copy.deepcopy(event))
            phrase = OrukPhraseEmotionFrame(
                self.stream_id, turn.turn_id, turn.user_id, turn.offset, copy.deepcopy(event)
            )
            await self.push_frame(phrase)
            await self._call_event_handler("on_phrase_emotion", copy.deepcopy(phrase))
        elif kind.startswith("conversation.item.input_audio_speaker."):
            speaker = OrukSpeakerBoundaryFrame(
                self.stream_id, turn.turn_id, turn.user_id, turn.offset, copy.deepcopy(event)
            )
            await self.push_frame(speaker)
            await self._call_event_handler("on_speaker_boundary", copy.deepcopy(speaker))
        elif kind == "session.usage":
            # Provider receipt, not all microphone bytes (which include gated silence).
            await self.start_stt_usage_metrics(
                STTUsage(audio_seconds=event["usage"]["audio_seconds"])
            )

    async def _complete(self, turn: _Turn, result: TurnResult) -> None:
        if self._wait_for_emotions:
            await self._final(
                turn,
                {
                    "type": TRANSCRIPT + "completed",
                    "transcript": result.transcript,
                    "request_id": result.request_id,
                },
            )
        frame = OrukTurnCompletedFrame(
            self.stream_id, turn.turn_id, turn.user_id, turn.offset, copy.deepcopy(result)
        )
        await self.push_frame(frame)
        await self._call_event_handler("on_turn_completed", copy.deepcopy(frame))
