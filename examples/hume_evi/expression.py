"""Optional bounded audio sidecar; primary frames never wait for inference."""

import asyncio
import math

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    STTMuteFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat_oruk.realtime import (
    ENDPOINT,
    RealtimeOptions,
    new_http_session,
    run_turn,
    validate_endpoint,
)

from .config import Config
from .policy import Record, SCOPE_METADATA, SignalStore


class ExpressionAudioTap(FrameProcessor):
    """16 kHz only; reuse the pinned package's authenticated streaming transport."""

    def __init__(
        self,
        store: SignalStore,
        config: Config,
        *,
        api_key: str = "",
        endpoint: str = ENDPOINT,
    ):
        super().__init__()
        if config.signals_enabled and (
            not isinstance(api_key, str)
            or not 1 <= len(api_key) <= 4096
            or any(ord(c) < 33 or ord(c) > 126 for c in api_key)
        ):
            raise ValueError(
                "Set a nonempty server-side ORUK_API_KEY when enabling signals"
            )
        validate_endpoint(endpoint)
        self.store, self.config = store, config
        self._key, self._endpoint = api_key, endpoint
        self._session = None
        self._task = None
        self._active: Record | None = None
        self._current: Record | None = None
        self._queue: asyncio.Queue | None = None
        self._queued_bytes = 0
        self._prefix = bytearray()
        self._track = ""
        self._samples = 0
        self._last_chunk_seconds = 0.0
        self._speech = False
        self._muted = False
        self._format_valid = True
        self._closed = False
        self._limit = int(config.signal_buffer_seconds * 32_000)
        self.peak_queued_bytes = 0

    @property
    def active_task(self):
        return self._task

    def _offer(self, data):
        if self._queue is None or data == b"":
            return
        if data is not None and self._queued_bytes + len(data) > self._limit:
            self._abandon("signal_skipped_backpressure")
            return
        if data is not None:
            self._queued_bytes += len(data)
            self.peak_queued_bytes = max(self.peak_queued_bytes, self._queued_bytes)
        self._queue.put_nowait(data)

    def _abandon(self, code):
        if self._active is not None:
            self.store.status(self._active, code)
        if self._task is not None:
            self._task.cancel()
        self._active = None
        self._current = None
        self._queue = None
        self._queued_bytes = 0

    async def _run(self, record, queue):
        async def audio():
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                self._queued_bytes = max(0, self._queued_bytes - len(chunk))
                # The transport accepts binary chunks of at most 10,240 bytes.
                for offset in range(0, len(chunk), 10_240):
                    yield chunk[offset : offset + 10_240]

        try:
            async with asyncio.timeout(
                self.config.max_turn_seconds + self.config.signal_timeout + 2
            ):
                await run_turn(
                    session=self._session,
                    api_key=self._key,
                    endpoint=self._endpoint,
                    audio=audio(),
                    request_id=record.utterance_id,
                    options=RealtimeOptions(
                        finish_timeout=self.config.signal_timeout,
                        max_turn_seconds=self.config.max_turn_seconds,
                    ),
                    on_event=lambda event: self.store.observe(record, event),
                    connect_timeout=min(2, self.config.signal_timeout),
                    max_connect_retries=0,
                )
            if not self._closed:
                self.store.status(record, "completed")
        except asyncio.CancelledError:
            if record.status == "streaming":
                self.store.status(record, "cancelled_outcome_unknown")
            raise
        except Exception:
            self.store.status(record, "signal_failed_outcome_unknown")
        finally:
            if self._active is record:
                self._active = None
                self._current = None
                self._queue = None
                self._queued_bytes = 0

    def _start_turn(self):
        if self._task is not None and not self._task.done():
            self.store.trace.append(
                {"generation": self.store.generation, "status": "signal_skipped_busy"}
            )
            return
        if not self._track or not self._prefix:
            self.store.trace.append(
                {
                    "generation": self.store.generation,
                    "status": "signal_missing_audio_identity",
                }
            )
            return
        record = self.store.begin(self._track, self._samples - len(self._prefix) // 2)
        self._active = record
        self._current = record
        self._queue = asyncio.Queue()
        self._task = self.create_task(
            self._run(record, self._queue), name="optional_oruk_expression"
        )
        if self._prefix:
            self._offer(bytes(self._prefix))
        self._prefix.clear()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM or not self.config.signals_enabled:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, StartFrame) and not self._closed:
            self._session = new_http_session()
        elif isinstance(frame, (CancelFrame, EndFrame)):
            await self._shutdown()
        elif isinstance(frame, STTMuteFrame):
            self._muted = frame.mute
            if frame.mute:
                self._abandon("signal_muted_outcome_unknown")
                self._speech = False
                self._prefix.clear()
        elif not self._closed and isinstance(frame, InputAudioRawFrame):
            if (
                frame.sample_rate != 16000
                or frame.num_channels != 1
                or len(frame.audio) % 2
                or len(frame.audio) > 32_000
            ):
                self._abandon("signal_unsupported_audio")
                self._prefix.clear()
                # The original 16 kHz sample timeline is no longer attestable.
                # Keep forwarding primary audio, but require a new generation
                # before expression analysis can resume.
                self._format_valid = False
            else:
                self._samples += len(frame.audio) // 2
                self._last_chunk_seconds = len(frame.audio) / 32_000
                if self._muted or not self._format_valid:
                    await self.push_frame(frame, direction)
                    return
                track = str(
                    getattr(frame, "user_id", "") or frame.transport_source or "input"
                )
                if len(track) > 128:
                    self._abandon("signal_unsupported_track")
                    self._prefix.clear()
                    await self.push_frame(frame, direction)
                    return
                if track != self._track:
                    self._abandon("signal_track_changed")
                    self._prefix.clear()
                    self._track = track
                if self._speech and self._current is not None:
                    if (
                        self._samples - self._current.start_sample
                        > self.config.max_turn_seconds * 16000
                    ):
                        self._abandon("signal_turn_limit")
                    else:
                        self._offer(bytes(frame.audio))
                elif not self._speech:
                    self._prefix.extend(frame.audio)
                    del self._prefix[
                        :-16_000
                    ]  # Half a second, including VAD confirmation.
        elif (
            not self._closed
            and not self._muted
            and self._format_valid
            and isinstance(frame, VADUserStartedSpeakingFrame)
        ):
            if not self._speech:
                self._speech = True
                # A larger VAD confirmation interval cannot fit our prefix.
                if (
                    math.isfinite(frame.start_secs)
                    and 0 <= frame.start_secs
                    and frame.start_secs + self._last_chunk_seconds <= 0.5
                ):
                    self._start_turn()
                else:
                    self.store.trace.append(
                        {
                            "generation": self.store.generation,
                            "status": "signal_unsupported_vad_prefix",
                        }
                    )
        elif (
            not self._closed
            and not self._muted
            and self._format_valid
            and isinstance(frame, VADUserStoppedSpeakingFrame)
        ):
            self._speech = False
            if self._current is not None:
                frame.metadata[SCOPE_METADATA] = self.store.seal(
                    self._current, self._samples
                )
                self._offer(None)
                self._current = None
            self._prefix.clear()
        await self.push_frame(frame, direction)

    async def _shutdown(self):
        self._closed = True
        self._abandon("cancelled_outcome_unknown")
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._prefix.clear()
        self.store.close()

    async def cleanup(self):
        await self._shutdown()
        await super().cleanup()
