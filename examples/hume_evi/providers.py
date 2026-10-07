"""Adapters that let real Pipecat providers meet the gateway's provenance contract.

`gateway.py` refuses provider output it cannot attribute to issued input. Plain
Pipecat services carry no such metadata, so this module wraps them:

- `ScopedVAD` runs a Pipecat `VADAnalyzer` over the issued chunks and reports
  each utterance as an exact, contiguous chunk span.
- `ScopedUtteranceSTT` transcribes exactly that span's bytes with an injected
  transcriber, so a final transcript can never be assigned by timing or text
  similarity.
- `TurnTaggedLLMMixin` copies the issued turn onto each LLM response frame and
  replaces the provider's tool-call ID with the issued tool ID.
- `ScopedSynthesisTTS` synthesizes with an injected synthesizer and returns
  turn-tagged 48 kHz mono PCM16 chunks of at most 100 ms. It holds each response
  end until that response's audio has been emitted.

The transcriber and synthesizer helpers at the end bind these seams to the
Oruk realtime API and to OpenAI-compatible audio endpoints. These adapters are
local contract implementations. They do not qualify any provider's quality,
latency, billing, retention or voice rights. Every failure path fails closed
through an `ErrorFrame` that the gateway turns into `provider_error`.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import io
import re
import uuid
import wave
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

import numpy as np
import soxr
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.time import time_now_iso8601

from .gateway import AUDIO, TOOL, TURN

Transcriber = Callable[[bytes, int], Awaitable[str]]
Synthesizer = Callable[[str], AsyncIterator[tuple[bytes, int]]]

OUTPUT_RATE = 48_000
MAX_OUTPUT_BYTES = 9_600  # 100 ms of 48 kHz mono PCM16.


def _chunk_span(frame) -> dict | None:
    """Return the gateway's single-chunk scope, or None when the frame has none."""
    span = frame.metadata.get(AUDIO)
    if (
        type(span) is dict
        and set(span) == {"generation", "epoch", "first", "last"}
        and isinstance(span["generation"], str)
        and all(type(span[key]) is int for key in ("epoch", "first", "last"))
        and span["first"] == span["last"] >= 1
    ):
        return span
    return None


def _utterance_span(frame) -> dict | None:
    span = frame.metadata.get(AUDIO)
    if (
        type(span) is dict
        and set(span) == {"generation", "epoch", "first", "last"}
        and isinstance(span["generation"], str)
        and all(type(span[key]) is int for key in ("epoch", "first", "last"))
        and 1 <= span["first"] <= span["last"]
    ):
        return span
    return None


def _tag(frame, key, value):
    frame.metadata[key] = dict(value) if isinstance(value, dict) else value
    return frame


async def _fail(processor: FrameProcessor, code: str, direction: FrameDirection):
    """Fail closed toward the gateway bridge that observes provider errors."""
    await processor.push_frame(ErrorFrame(error=code), direction)


class ScopedVAD(FrameProcessor):
    """Report VAD boundaries as exact issued chunk spans.

    The analyzer still decides speech boundaries with its own parameters. This
    wrapper records which issued chunks those decisions cover. The analyzer
    scores whole frames and carries partial frames between calls, so an
    utterance starts at the chunk holding the first byte of the earliest frame
    that could have left QUIET. It ends at the chunk where the analyzer returns
    to QUIET, trailing silence included. A new PCM epoch (after typed input)
    discards any open utterance. Speech longer than `max_seconds` fails closed
    instead of being split, because a split would let the next VAD start retire
    the first part's transcript.
    """

    def __init__(self, analyzer: VADAnalyzer, *, max_seconds: float = 30.0):
        super().__init__()
        if isinstance(max_seconds, bool) or not 0 < max_seconds <= 300:
            raise ValueError("max_seconds must be in (0, 300]")
        self._analyzer = analyzer
        self._max_seconds = max_seconds
        self._max_bytes = 0
        self._frame_bytes = 0
        self._fed = 0  # Bytes handed to the analyzer, including carried partial frames.
        self._recent: deque = deque(maxlen=64)  # (epoch, seq, start_byte, end_byte)
        self._epoch: int | None = None
        self._first: int | None = None
        self._first_byte = 0
        self._last_stop: int | None = None
        self._speaking = False
        self._refused = False

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        self._analyzer.set_sample_rate(setup.audio_in_sample_rate)
        self._frame_bytes = 2 * self._analyzer.num_frames_required()
        self._max_bytes = int(self._max_seconds * setup.audio_in_sample_rate) * 2

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
            return
        span = _chunk_span(frame)
        if span is None:
            # The gateway validates issued audio before VAD; never guess a scope.
            await self.push_frame(frame, direction)
            await _fail(self, "vad_unscoped_audio", FrameDirection.DOWNSTREAM)
            return
        if span["epoch"] != self._epoch:
            self._epoch, self._first, self._speaking, self._last_stop = span["epoch"], None, False, None
        seq = span["first"]
        before = self._fed
        self._fed += len(frame.audio)
        self._recent.append((span["epoch"], seq, before, self._fed))
        state = await self._analyzer.analyze_audio(frame.audio)
        # Forward the chunk before its boundary so downstream STT has buffered it.
        await self.push_frame(frame, direction)
        if state == VADState.QUIET:
            if self._speaking:
                await self._stop(span, seq)
            self._first, self._refused = None, False
            return
        if self._refused:
            return
        if self._first is None:
            self._onset(span["epoch"], seq, before)
        if state == VADState.SPEAKING and not self._speaking:
            self._speaking = True
            started = VADUserStartedSpeakingFrame(start_secs=self._analyzer.params.start_secs)
            await self.push_frame(_tag(started, AUDIO, self._span(span, seq)))
        if self._fed - self._first_byte > self._max_bytes:
            self._speaking, self._first, self._refused = False, None, True
            await _fail(self, "vad_utterance_too_long", FrameDirection.DOWNSTREAM)

    def _onset(self, epoch, seq, before):
        # The first frame scored in this call starts at the oldest carried byte.
        onset = before - before % self._frame_bytes if self._frame_bytes else before
        self._first, self._first_byte = seq, before
        for chunk_epoch, chunk_seq, start, end in self._recent:
            # Never reach back into a chunk the previous utterance already owns.
            if (chunk_epoch == epoch and start <= onset < end
                    and (self._last_stop is None or chunk_seq > self._last_stop)):
                self._first, self._first_byte = chunk_seq, start
                return

    def _span(self, span, last):
        return {"generation": span["generation"], "epoch": span["epoch"], "first": self._first, "last": last}

    async def _stop(self, span, seq):
        stopped = VADUserStoppedSpeakingFrame(stop_secs=self._analyzer.params.stop_secs)
        await self.push_frame(_tag(stopped, AUDIO, self._span(span, seq)))
        self._speaking, self._first, self._last_stop = False, None, seq


class ScopedUtteranceSTT(FrameProcessor):
    """Transcribe exactly the issued chunks named by each VAD stop span.

    Keeps a bounded buffer of the current epoch's chunks (`max_seconds`, which
    must be at least ScopedVAD's). A VAD stop starts one owned transcription of
    that span; its final `TranscriptionFrame` echoes the same span. A newer VAD
    start, an interruption, a new epoch, cancellation or end cancels an
    in-flight transcription; the gateway would discard its result as stale
    anyway. Provider billing for a cancelled request stays unreconciled. Empty
    text produces no transcript. Missing audio, a provider failure, a timeout or
    a non-string result fails closed.
    """

    def __init__(self, transcribe: Transcriber, *, user_id: str = "user",
                 max_seconds: float = 30.0, timeout: float = 20.0):
        super().__init__()
        if isinstance(max_seconds, bool) or not 0 < max_seconds <= 300:
            raise ValueError("max_seconds must be in (0, 300]")
        if isinstance(timeout, bool) or not 0 < timeout <= 120:
            raise ValueError("timeout must be in (0, 120] seconds")
        self._transcribe = transcribe
        self._user_id = user_id
        self._max_seconds = max_seconds
        self._timeout = timeout
        self._sample_rate = 16_000
        self._max_bytes = int(max_seconds * self._sample_rate) * 2
        self._epoch: int | None = None
        self._chunks: OrderedDict[int, bytes] = OrderedDict()
        self._buffered = 0
        self._task: asyncio.Task | None = None
        self.requests = 0  # Transcriber invocations, for local accounting.

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        self._sample_rate = setup.audio_in_sample_rate
        self._max_bytes = int(self._max_seconds * self._sample_rate) * 2

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (CancelFrame, EndFrame, InterruptionFrame)):
            # A new user turn (typed or spoken) supersedes an unfinished transcript.
            await self._cancel()
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InputAudioRawFrame):
            span = _chunk_span(frame)
            if span is not None:
                if span["epoch"] != self._epoch:
                    await self._cancel()
                    self._epoch = span["epoch"]
                    self._chunks.clear()
                    self._buffered = 0
                self._chunks[span["first"]] = frame.audio
                self._buffered += len(frame.audio)
                # Keep at least one maximal utterance (ScopedVAD's bound) plus its onset chunk.
                while self._buffered - len(next(iter(self._chunks.values()))) >= self._max_bytes:
                    self._buffered -= len(self._chunks.popitem(last=False)[1])
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._cancel()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.push_frame(frame, direction)
            span = _utterance_span(frame)
            pcm = self._assemble(span)
            if pcm is None:
                await _fail(self, "stt_span_audio_unavailable", FrameDirection.DOWNSTREAM)
                return
            await self._cancel()
            self._task = self.create_task(self._run(dict(span), pcm), "scoped_transcription")
            return
        await self.push_frame(frame, direction)

    def _assemble(self, span) -> bytes | None:
        if span is None or span["epoch"] != self._epoch:
            return None
        parts = []
        for seq in range(span["first"], span["last"] + 1):
            chunk = self._chunks.get(seq)
            if chunk is None:
                return None
            parts.append(chunk)
        return b"".join(parts)

    async def _run(self, span, pcm):
        self.requests += 1
        try:
            async with asyncio.timeout(self._timeout):
                text = await self._transcribe(pcm, self._sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never forward provider exception text; it may contain request data.
            await _fail(self, "stt_provider_failed", FrameDirection.DOWNSTREAM)
            return
        if not isinstance(text, str):
            await _fail(self, "stt_invalid_result", FrameDirection.DOWNSTREAM)
            return
        text = text.strip()
        if not text:
            return
        final = TranscriptionFrame(text, self._user_id, time_now_iso8601(), finalized=True)
        await self.push_frame(_tag(final, AUDIO, span))

    async def _cancel(self):
        task, self._task = self._task, None
        if task is not None and not task.done():
            await self.cancel_task(task)

    async def cleanup(self):
        await self._cancel()
        await super().cleanup()


_TURN_SCOPE: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "oruk_evi_gateway_llm_turn", default=None
)


class TurnTaggedLLMMixin:
    """Mix into a Pipecat `LLMService` subclass placed behind the gateway.

    Example: `class GatewayOpenAILLM(TurnTaggedLLMMixin, OpenAILLMService): pass`.

    The issued `TURN`/`TOOL` pair is read from each `LLMContextFrame` and held in
    a context variable while that frame is processed. Response start/text/end
    frames pushed in that scope receive the turn. Frames pushed outside it stay
    untagged, which the gateway refuses. Exactly one function call per response
    is accepted, and it is given the issued tool ID before Pipecat records it in
    the context, so the provider's own ID never reaches the client. Parallel or
    malformed calls fail closed, except that Pipecat drops an unparseable call
    before dispatch, so one malformed call beside one valid call is not visible
    here. Services that stream from detached tasks without copying the current
    context, and Pipecat's `filter_incomplete_user_turns` mode, are not
    supported by this mixin.
    """

    async def process_frame(self, frame, direction):
        if not isinstance(frame, LLMContextFrame):
            await super().process_frame(frame, direction)
            return
        turn, tool = frame.metadata.get(TURN), frame.metadata.get(TOOL)
        scope = (turn, tool) if isinstance(turn, str) and isinstance(tool, str) else None
        token = _TURN_SCOPE.set(scope)
        try:
            await super().process_frame(frame, direction)
        finally:
            _TURN_SCOPE.reset(token)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame)
        ):
            scope = _TURN_SCOPE.get()
            if scope is not None:
                frame.metadata[TURN] = scope[0]
        await super().push_frame(frame, direction)

    async def run_function_calls(self, function_calls: Sequence[Any]):
        calls = list(function_calls)
        scope = _TURN_SCOPE.get()
        # Pipecat calls this only when the stream contained a call, and drops
        # calls whose arguments do not parse; an empty list is a malformed call.
        if scope is None or len(calls) != 1:
            await _fail(self, "llm_unsupported_tool_calls", FrameDirection.DOWNSTREAM)
            return
        await super().run_function_calls([dataclasses.replace(calls[0], tool_call_id=scope[1])])


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


class ScopedSynthesisTTS(FrameProcessor):
    """Synthesize turn-tagged LLM text into gateway-shaped audio.

    Text is grouped into sentences per turn and spoken in order by one owned
    worker. Each sentence yields TTSStarted, 48 kHz mono PCM16 chunks of at most
    100 ms, TTSText and TTSStopped, all tagged with the source turn. A response
    end is forwarded only after its audio. Interruption or cancellation drops
    queued and in-flight speech; bytes already pushed are not recalled. A
    graceful `EndFrame` first speaks what was already accepted, within the
    synthesis deadline. A synthesizer
    failure, invalid audio or timeout fails closed upstream, where the gateway's
    provider bridge observes it.
    """

    def __init__(self, synthesize: Synthesizer, *, max_chars: int = 2_000, timeout: float = 20.0,
                 max_pending: int = 32):
        super().__init__()
        if type(max_chars) is not int or not 1 <= max_chars <= 8_000:
            raise ValueError("max_chars must be an integer in [1, 8000]")
        if isinstance(timeout, bool) or not 0 < timeout <= 120:
            raise ValueError("timeout must be in (0, 120] seconds")
        if type(max_pending) is not int or not 1 <= max_pending <= 128:
            raise ValueError("max_pending must be an integer in [1, 128]")
        self._synthesize = synthesize
        self._max_chars = max_chars
        self._timeout = timeout
        self._max_pending = max_pending
        self._failed = False
        self._pending: dict[str, str] = {}
        self._queue: deque = deque()
        self._ready = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._busy = False
        self.requests = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._start_worker()
            return
        if isinstance(frame, EndFrame):
            await self._drain()
            await self._stop_worker()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, CancelFrame):
            await self._stop_worker()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            await self._stop_worker()
            self._pending.clear()
            await self.push_frame(frame, direction)
            self._start_worker()
            return
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if self._failed:
            return
        if isinstance(frame, LLMTextFrame):
            turn = frame.metadata.get(TURN)
            if not isinstance(turn, str):
                await _fail(self, "tts_untagged_text", FrameDirection.UPSTREAM)
                return
            if getattr(frame, "skip_tts", None):
                return
            text = self._pending.get(turn, "") + frame.text
            if len(text) > self._max_chars:
                await self._refuse("tts_text_too_long")
                return
            *complete, rest = _SENTENCE_END.split(text)
            for sentence in complete:
                if not self._enqueue(("speak", turn, sentence)):
                    await self._refuse("tts_pending_limit")
                    return
            self._pending[turn] = rest
            return
        if isinstance(frame, LLMFullResponseEndFrame):
            turn = frame.metadata.get(TURN)
            rest = self._pending.pop(turn, "") if isinstance(turn, str) else ""
            if rest.strip():
                if not self._enqueue(("speak", turn, rest)):
                    await self._refuse("tts_pending_limit")
                    return
            if not self._enqueue(("forward", frame)):
                await self._refuse("tts_pending_limit")
            return
        await self.push_frame(frame, direction)

    def _enqueue(self, item):
        if item[0] == "speak" and not item[2].strip():
            return True
        if self._failed or len(self._queue) >= self._max_pending:
            return False
        self._queue.append(item)
        self._ready.set()
        return True

    async def _refuse(self, code):
        self._failed = True
        self._pending.clear()
        self._queue.clear()
        await _fail(self, code, FrameDirection.UPSTREAM)

    def _start_worker(self):
        if self._worker is None:
            self._worker = self.create_task(self._work(), "scoped_synthesis")

    async def _drain(self):
        """Speak what was already accepted before a graceful end, within bounds."""
        for turn, rest in list(self._pending.items()):
            if not self._enqueue(("speak", turn, rest)):
                await self._refuse("tts_pending_limit")
                break
        self._pending.clear()
        try:
            async with asyncio.timeout(self._timeout * (len(self._queue) + 1)):
                while self._worker is not None and (self._queue or self._busy):
                    await asyncio.sleep(0.01)
        except TimeoutError:
            await _fail(self, "tts_drain_deadline", FrameDirection.UPSTREAM)

    async def _stop_worker(self):
        self._queue.clear()
        self._pending.clear()
        self._ready.clear()
        task, self._worker = self._worker, None
        if task is not None and not task.done():
            await self.cancel_task(task)

    async def _work(self):
        while True:
            await self._ready.wait()
            while self._queue:
                item = self._queue.popleft()
                self._busy = True
                try:
                    if item[0] == "forward":
                        await self.push_frame(item[1])
                    elif not await self._speak(item[1], item[2].strip()):
                        self._failed = True
                        self._pending.clear()
                        self._queue.clear()
                finally:
                    self._busy = False
            self._ready.clear()

    async def _speak(self, turn: str, text: str) -> bool:
        if len(text) > self._max_chars:
            await _fail(self, "tts_text_too_long", FrameDirection.UPSTREAM)
            return False
        self.requests += 1
        await self.push_frame(_tag(TTSStartedFrame(), TURN, turn))
        # One stateful resampler per sentence: no seams between provider chunks,
        # and the final flush keeps the filter's tail.
        resampler, stream_rate = None, None
        carry = b""
        out = b""
        input_bytes = 0
        try:
            async with asyncio.timeout(self._timeout), contextlib.aclosing(self._synthesize(text)) as stream:
                async for audio, rate in stream:
                    if (not isinstance(audio, (bytes, bytearray)) or len(audio) > 96_000 or type(rate) is not int
                            or not 8_000 <= rate <= 48_000 or rate != (stream_rate or rate)):
                        await _fail(self, "tts_invalid_audio", FrameDirection.UPSTREAM)
                        return False
                    input_bytes += len(audio)
                    if stream_rate is None:
                        stream_rate = rate
                        if rate != OUTPUT_RATE:
                            resampler = soxr.ResampleStream(rate, OUTPUT_RATE, 1, dtype="int16", quality="VHQ")
                    data = carry + bytes(audio)
                    even = len(data) - len(data) % 2
                    carry = data[even:]
                    if not even:
                        continue
                    pcm = data[:even]
                    if resampler is not None:
                        pcm = resampler.resample_chunk(np.frombuffer(pcm, dtype=np.int16), last=False).tobytes()
                    out += pcm
                    while len(out) >= MAX_OUTPUT_BYTES:
                        await self._audio(turn, out[:MAX_OUTPUT_BYTES])
                        out = out[MAX_OUTPUT_BYTES:]
                if carry or not input_bytes:
                    await _fail(self, "tts_incomplete_audio", FrameDirection.UPSTREAM)
                    return False
                if resampler is not None:
                    out += resampler.resample_chunk(np.zeros(0, dtype=np.int16), last=True).tobytes()
                    while len(out) >= MAX_OUTPUT_BYTES:
                        await self._audio(turn, out[:MAX_OUTPUT_BYTES])
                        out = out[MAX_OUTPUT_BYTES:]
        except asyncio.CancelledError:
            raise
        except Exception:
            await _fail(self, "tts_provider_failed", FrameDirection.UPSTREAM)
            return False
        if out:
            await self._audio(turn, out)
        await self.push_frame(_tag(TTSTextFrame(text=text, aggregated_by="sentence"), TURN, turn))
        await self.push_frame(_tag(TTSStoppedFrame(), TURN, turn))
        return True

    async def _audio(self, turn, pcm):
        frame = TTSAudioRawFrame(audio=pcm, sample_rate=OUTPUT_RATE, num_channels=1)
        await self.push_frame(_tag(frame, TURN, turn))

    async def cleanup(self):
        await self._stop_worker()
        await super().cleanup()


def pcm16_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap mono PCM16 in a WAV container without changing the samples."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return output.getvalue()


def oruk_realtime_transcriber(session, *, api_key: str, endpoint: str, options=None,
                              max_connect_retries: int = 0) -> Transcriber:
    """Transcribe each scoped utterance as one Oruk realtime turn.

    Uses `pipecat_oruk.realtime.run_turn`, which accepts completion only after a
    final transcript, the usage receipt and a clean close. Connection retries
    default to zero so an utterance is never silently resubmitted. Phrase
    emotions are off by default; enable them through `options` only with a
    consumer for those events. The caller owns `session` and closes it.
    """
    from pipecat_oruk.realtime import SAMPLE_RATE, RealtimeOptions, run_turn

    settings = options if options is not None else RealtimeOptions(phrase_emotions=False)

    async def transcribe(pcm: bytes, sample_rate: int) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError("oruk_realtime_requires_16k")

        async def chunks():
            for start in range(0, len(pcm), 3_200):
                yield pcm[start:start + 3_200]

        result = await run_turn(session=session, api_key=api_key, endpoint=endpoint,
                                audio=chunks(), request_id=uuid.uuid4().hex, options=settings,
                                on_event=lambda _event: None, max_connect_retries=max_connect_retries)
        return result.transcript

    return transcribe


def openai_transcriber(client, *, model: str = "gpt-4o-mini-transcribe",
                       language: str | None = None) -> Transcriber:
    """Transcribe each scoped utterance with an OpenAI-compatible endpoint.

    `client` is an `openai.AsyncOpenAI` owned and closed by the caller. Configure
    its retry policy explicitly: the SDK retries some failures by default.
    """

    async def transcribe(pcm: bytes, sample_rate: int) -> str:
        request: dict[str, Any] = {"model": model, "file": ("utterance.wav", pcm16_wav(pcm, sample_rate), "audio/wav")}
        if language:
            request["language"] = language
        result = await client.audio.transcriptions.create(**request)
        return result.text

    return transcribe


def openai_speech_synthesizer(client, *, model: str = "gpt-4o-mini-tts", voice: str = "alloy",
                              instructions: str | None = None) -> Synthesizer:
    """Stream OpenAI-compatible speech as 24 kHz mono PCM16.

    `client` is an `openai.AsyncOpenAI` owned and closed by the caller. Voice
    selection and any cloning rights are the customer's decision.
    """

    async def synthesize(text: str):
        request: dict[str, Any] = {"model": model, "voice": voice, "input": text, "response_format": "pcm"}
        if instructions:
            request["instructions"] = instructions
        async with client.audio.speech.with_streaming_response.create(**request) as response:
            async for chunk in response.iter_bytes(4_800):
                yield chunk, 24_000

    return synthesize
