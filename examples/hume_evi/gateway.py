"""Unbound EVI-shaped PCM profile. No listener, provider or credential default.

Only explicitly injected, cancellation-cooperative Pipecat services are usable.
LLM/TTS adapters must preserve TURN metadata from their input frames on every
output frame; untagged output refuses instead of being assigned to a newer turn.
This local profile is not unchanged React/WebM or full Hume EVI compatibility.
"""
from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass, replace
import io
import json
import math
import time
from typing import Awaitable, Callable
import uuid
import wave
import weakref

from aiohttp import WSMsgType, web
from pipecat.frames.frames import (
    InputAudioRawFrame, InterruptionFrame, LLMContextFrame,
    LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame,
    LLMMessagesAppendFrame, TranscriptionFrame, TTSAudioRawFrame,
    TTSTextFrame, TTSStartedFrame, TTSStoppedFrame, VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame, ErrorFrame,
)
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.workers.runner import WorkerRunner

from .config import Config
from .pipeline import build_pipeline

TURN = "oruk_evi_gateway_turn"
TOOL = "oruk_evi_gateway_tool"
AUDIO = "oruk_evi_gateway_audio"
PROFILE = "local-pcm-evi-v1"


@dataclass(frozen=True)
class GatewayLimits:
    """Finite local-profile defaults, not production capacity tuning."""
    sessions: int = 2
    messages: int = 2000
    turns: int = 8
    pending_output: int = 32
    input_bytes: int = 960_000
    output_bytes: int = 4_000_000
    session_seconds: float = 60
    settings_seconds: float = 3
    send_seconds: float = 2
    tool_seconds: float = 2

    def __post_init__(self):
        for field, low, high in (
            ("sessions", 1, 8), ("messages", 1, 6000), ("turns", 1, 32),
            ("pending_output", 1, 128), ("input_bytes", 640, 1_920_000),
            ("output_bytes", 128, 8_000_000),
        ):
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid_gateway_limits")
        for field in ("session_seconds", "settings_seconds", "send_seconds", "tool_seconds"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.05 <= value <= 120:
                raise ValueError("invalid_gateway_limits")


@dataclass(frozen=True)
class Providers:
    """Fresh owned processors per connection; no reused provider instances.

    LLM copies TURN from LLMContextFrame onto start/text/end output. For the one
    optional tool call it uses that frame's issued TOOL value as tool_call_id.
    VAD start/stop and STT final frames echo AUDIO: generation, epoch and exact
    first/last chunk sequence numbers. STT timing is never inferred from when a
    transcript arrives. TTS copies TURN onto every start/audio/text/stop frame and emits mono PCM16
    at exactly 48 kHz. A normal unadapted provider is not silently qualified.
    """
    stt: FrameProcessor
    llm: LLMService
    tts: FrameProcessor
    vad: FrameProcessor
    close: Callable[[], Awaitable[None]]


class ProfileError(ValueError):
    pass


def require(value, code="invalid_message"):
    if not value:
        raise ProfileError(code)


def message_json(text):
    require(isinstance(text, str) and len(text.encode("utf-8")) <= 16_384)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result)
            result[key] = value
        return result
    try:
        result = json.loads(text, object_pairs_hook=unique,
                            parse_constant=lambda _: require(False))
    except (ValueError, UnicodeError, RecursionError):
        raise ProfileError("invalid_message") from None
    require(type(result) is dict and type(result.get("type")) is str)
    # The accepted payloads below have depth <=3. Bound even refused JSON before
    # field checks; this parser is not an arbitrary-data recursion facility.
    stack, nodes = [(result, 0)], 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        require(nodes <= 64 and depth <= 4)
        if isinstance(value, dict):
            require(len(value) <= 16)
            stack.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            require(False)
    return result


def session_config(value, configured):
    require(set(value) <= {"type", "audio", "system_prompt"} and value["type"] == "session_settings")
    audio = value.get("audio")
    require(type(audio) is dict and set(audio) == {"encoding", "sample_rate", "channels"})
    require(audio["encoding"] == "linear16" and type(audio["sample_rate"]) is int
            and audio["sample_rate"] == 16000 and type(audio["channels"]) is int and audio["channels"] == 1)
    prompt = value.get("system_prompt", configured.system_prompt)
    require(isinstance(prompt, str) and 1 <= len(prompt) <= 4000)
    # No automatic expression-model requests or imported Hume/provider settings.
    return replace(configured, system_prompt=prompt, greeting="", signals_enabled=False)


async def joined(task):
    """Retain actual child closure through repeated caller cancellation.

    Non-cooperative providers retain the owner; no elapsed timeout proves they
    stopped. Root/application must retain a finite outer process owner too.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    try:
        result = task.result()
    except asyncio.CancelledError:
        result = None
    if cancelled:
        raise asyncio.CancelledError()
    return result


class Session:
    def __init__(self, ws, limits):
        self.ws, self.limits = ws, limits
        self.generation = uuid.uuid4().hex
        self.chat_id, self.group_id = uuid.uuid4().hex, uuid.uuid4().hex
        self.output = asyncio.Queue(maxsize=limits.pending_output)
        self.failure = asyncio.Event()
        self.code = None
        self.closed = False
        self.cleanup_complete = False
        self.acquisition_uncertain = False
        self.active = None
        self.active_finished = False
        self.turns = self.messages = self.input_bytes = self.output_bytes = 0
        self.output_frames = self.stale_frames = 0
        self.pending_input = 0
        self.samples = self.audio_epoch = 0
        self.audio_records = {}
        self.primary_span = self.primary_stopped = None
        self.primary_final = False
        self.transcripts = 0
        self.text_bytes = 0
        self.audio_index = 0
        self.tool_id = None
        self.tool_used = False
        self.tool_future = None
        self.continuation = None
        self.response_has_text = False
        self.worker = self.conversation = None
        self.runner = self.pipeline_task = self.bundle = None
        self.tasks = []

    def abort(self, code):
        if self.code is None:
            self.code = code
        self.failure.set()

    def emit(self, value, token=None):
        if self.closed or self.failure.is_set():
            return
        if token is not None and token != self.active:
            self.stale_frames += 1
            return
        encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
        size = len(encoded.encode("utf-8"))
        if size > 32_768 or self.output_bytes + size > self.limits.output_bytes or self.output.full():
            self.abort("output_backpressure")
            return
        self.output_bytes += size
        self.output.put_nowait((token, encoded))

    def interrupt(self):
        self.continuation = None
        if self.active is None or self.active_finished:
            return
        self.active = None
        self.tool_id = None
        if self.tool_future is not None and not self.tool_future.done():
            self.tool_future.cancel()
        while not self.output.empty():
            self.output.get_nowait()
        # This local profile specifies Unix seconds for the timestamp.
        self.emit({"type": "user_interruption", "time": time.time()})

    def new_turn(self, frame, continuation=False):
        if self.closed or self.failure.is_set():
            return
        if continuation:
            if self.active is None or self.active_finished or self.continuation != self.active:
                self.abort("orphan_continuation")
                return
            self.continuation = None
        else:
            if self.active is not None and not self.active_finished:
                self.abort("overlapping_turn")
                return
            self.turns += 1
            if self.turns > self.limits.turns:
                self.abort("turn_limit")
                return
            self.active = self.generation + "." + uuid.uuid4().hex
            self.tool_id = uuid.uuid4().hex
            self.tool_used = False
            self.active_finished = False
            self.audio_index = 0
            self.response_has_text = False
        frame.metadata[TURN] = self.active
        frame.metadata[TOOL] = self.tool_id

    def accept_output(self, frame):
        if self.closed or self.failure.is_set():
            return None
        token = frame.metadata.get(TURN)
        if not isinstance(token, str):
            self.abort("untagged_provider_output")
            return None
        if token != self.active:
            self.stale_frames += 1
            return None
        if self.active_finished:
            self.abort("output_after_turn_end")
            return None
        return token

    async def tool(self, params):
        token = self.active
        if (self.closed or self.failure.is_set() or token is None or self.active_finished
                or self.tool_used or params.tool_call_id != self.tool_id):
            self.abort("stale_or_duplicate_tool")
            return
        args = params.arguments
        if type(args) is not dict or set(args) != {"order_id"} or not isinstance(args["order_id"], str) or not 1 <= len(args["order_id"]) <= 32:
            self.abort("invalid_tool_arguments")
            return
        self.tool_used = True
        future = asyncio.get_running_loop().create_future()
        self.tool_future = future
        self.emit({"type": "tool_call", "name": "lookup_demo_order", "tool_type": "function",
                   "tool_call_id": self.tool_id, "parameters": json.dumps(args), "response_required": True}, token)
        try:
            async with asyncio.timeout(self.limits.tool_seconds):
                result = await future
        except TimeoutError:
            result = {"error": "tool_timeout"}
        finally:
            # Only this call's waiter is owned here. A late old finalizer must
            # never cancel a newer turn's mutable current-tool reference.
            if not future.done():
                future.cancel()
        if not self.closed and token == self.active and not self.failure.is_set():
            self.continuation = token
            await params.result_callback(result)

    def audio_span(self, frame):
        """Validate a trusted adapter's exact issued PCM span, never a clock guess."""
        span = frame.metadata.get(AUDIO)
        require(type(span) is dict and set(span) == {"generation", "epoch", "first", "last"}, "unscoped_primary")
        require(span["generation"] == self.generation and all(type(span[k]) is int for k in ("epoch", "first", "last")), "unscoped_primary")
        first, last = span["first"], span["last"]
        require(1 <= first <= last <= len(self.audio_records), "unscoped_primary")
        left, right = self.audio_records[first], self.audio_records[last]
        require(left[0] == right[0] == span["epoch"], "unscoped_primary")
        if span["epoch"] != self.audio_epoch:
            self.stale_frames += 1
            return None
        return (first, last, left[1], right[2])

    async def sender(self):
        while True:
            _token, encoded = await self.output.get()
            # Completed earlier turns retain queue order. Interruption removes
            # queued old messages, but cannot recall bytes already sent.
            async with asyncio.timeout(self.limits.send_seconds):
                await self.ws.send_str(encoded)
            self.output_frames += 1

    async def receive(self):
        async for packet in self.ws:
            if packet.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                return
            require(packet.type == WSMsgType.TEXT, "text_json_required")
            self.messages += 1
            require(self.messages <= self.limits.messages, "message_limit")
            value = message_json(packet.data)
            kind = value["type"]
            if kind == "audio_input":
                require(set(value) == {"type", "data"} and isinstance(value["data"], str))
                try:
                    pcm = base64.b64decode(value["data"], validate=True)
                except (ValueError, UnicodeError):
                    raise ProfileError("invalid_audio") from None
                require(0 < len(pcm) <= 3200 and len(pcm) % 2 == 0
                        and base64.b64encode(pcm).decode() == value["data"], "invalid_audio")
                require(self.input_bytes + len(pcm) <= self.limits.input_bytes, "audio_limit")
                require(self.pending_input < 16, "input_backpressure")
                self.input_bytes += len(pcm)
                self.pending_input += 1
                seq = len(self.audio_records) + 1
                start = self.samples
                self.samples += len(pcm) // 2
                self.audio_records[seq] = (self.audio_epoch, start, self.samples)
                frame = InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
                frame.metadata[AUDIO] = {"generation": self.generation, "epoch": self.audio_epoch, "first": seq, "last": seq}
                await self.worker.queue_frame(frame)
            elif kind == "user_input":
                require(set(value) == {"type", "text"} and isinstance(value["text"], str)
                        and 0 < len(value["text"]) <= 2000)
                self.text_bytes += len(value["text"].encode())
                require(self.text_bytes <= 16_000, "text_limit")
                self.audio_epoch += 1
                self.primary_span = self.primary_stopped = None
                self.interrupt()
                await self.worker.queue_frame(InterruptionFrame())
                # System-frame interruption clears queued ordinary frames.
                # Await the existing round-trip Uninterruptible flush marker
                # before admitting the new typed message into that pipeline.
                async with asyncio.timeout(self.limits.settings_seconds):
                    require(await self.worker.flush_pipeline(timeout=self.limits.settings_seconds), "interruption_unresolved")
                self.emit({"type": "user_message", "from_text": True, "interim": False,
                           "message": {"role": "user", "content": value["text"]}, "models": {},
                           "time": {"begin": self.samples / 16, "end": self.samples / 16}})
                await self.worker.queue_frame(LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": value["text"]}], run_llm=True))
            elif kind in {"tool_response", "tool_error"}:
                allowed = {"type", "tool_call_id", "content"} if kind == "tool_response" else {"type", "tool_call_id", "error"}
                require(set(value) == allowed and value.get("tool_call_id") == self.tool_id
                        and self.tool_used and self.tool_future is not None and not self.tool_future.done(), "unknown_tool_response")
                content = value["content" if kind == "tool_response" else "error"]
                require(isinstance(content, str) and len(content.encode()) <= 4096)
                self.tool_future.set_result({"content": content} if kind == "tool_response" else {"error": "client_tool_error"})
            else:
                raise ProfileError("unsupported_message")

    async def run(self, bundle, config):
        self.bundle = bundle
        try:
            await self.start_pipeline(bundle, config)
            await self.wait_pipeline()
        except ProfileError as error:
            self.abort(str(error))
        except TimeoutError:
            self.abort("deadline")
        except asyncio.CancelledError:
            self.abort("cancelled")
            raise
        except Exception:
            self.abort("owned_work_failed")
        finally:
            owner = asyncio.create_task(self.shutdown(), name="evi_profile_shutdown")
            await joined(owner)

    async def start_pipeline(self, bundle, config):
        self.conversation = build_pipeline(
            WireTransport(self), bundle.stt, bundle.llm, bundle.tts, config=config, vad=bundle.vad,
            tool_handler=self.tool, after_stt=UserObserver(self),
            before_llm=TurnBridge(self), after_llm=ProviderBridge(self),
        )
        self.worker = PipelineWorker(
            self.conversation.pipeline,
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=48000),
            enable_rtvi=False, cancel_on_idle_timeout=False,
            processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
        )
        ready = asyncio.Event()
        @self.worker.event_handler("on_pipeline_started")
        async def started(*_):
            ready.set()
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await self.runner.add_workers(self.worker)
        self.pipeline_task = asyncio.create_task(self.runner.run(), name="evi_profile_pipeline")
        self.tasks.append(self.pipeline_task)
        async with asyncio.timeout(self.limits.settings_seconds):
            await ready.wait()

    async def wait_pipeline(self):
        self.emit({"type": "chat_metadata", "chat_id": self.chat_id, "chat_group_id": self.group_id})
        receiver = asyncio.create_task(self.receive(), name="evi_profile_receive")
        sender = asyncio.create_task(self.sender(), name="evi_profile_send")
        failed = asyncio.create_task(self.failure.wait(), name="evi_profile_failure")
        self.tasks.extend((receiver, sender, failed))
        async with asyncio.timeout(self.limits.session_seconds):
            done, _ = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is not failed:
                task.result()
        if self.pipeline_task in done and not self.ws.closed:
            self.abort("pipeline_stopped")

    async def shutdown(self):
        self.closed = True
        if self.tool_future is not None and not self.tool_future.done():
            self.tool_future.cancel()
        # First stop admission/transport tasks; then join framework and the
        # provider factory's explicit ownership finalizer. No timeout detaches it.
        framework_closed = self.pipeline_task is None
        try:
            for task in self.tasks:
                if task is not self.pipeline_task and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in self.tasks if task is not self.pipeline_task), return_exceptions=True)
            if self.runner is not None and self.pipeline_task is not None:
                await self.runner.cancel()
                await joined(self.pipeline_task)
                framework_closed = True
        finally:
            provider_closed = True
            if self.bundle is not None:
                try:
                    await self.bundle.close()
                except BaseException:
                    provider_closed = False
                    self.abort("provider_cleanup_failed")
            self.cleanup_complete = provider_closed and framework_closed and not self.acquisition_uncertain
            if self.code and not self.ws.closed:
                try:
                    async with asyncio.timeout(self.limits.send_seconds):
                        await self.ws.send_json({"type": "error", "code": self.code, "slug": self.code,
                                                 "message": "PCM gateway profile could not complete this session."})
                except Exception:
                    pass
            await self.ws.close(code=1008 if self.code else 1000)


class WireInput(FrameProcessor):
    def __init__(self, session):
        super().__init__()
        self.session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM and isinstance(frame, ErrorFrame):
            # Pipecat services report failures upstream (push_error). This is the
            # last owned processor before the source, so an LLM or STT failure
            # cannot leave a turn silently open until the session deadline.
            self.session.abort("provider_error")
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, InputAudioRawFrame):
            self.session.pending_input -= 1
            try:
                span = self.session.audio_span(frame)
            except ProfileError as error:
                self.session.abort(str(error))
                return
            if span is None:
                return
        await self.push_frame(frame, direction)


class UserObserver(FrameProcessor):
    def __init__(self, session):
        super().__init__()
        self.session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        state = self.session
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame, TranscriptionFrame)):
                try:
                    span = state.audio_span(frame)
                except ProfileError as error:
                    state.abort(str(error))
                    return
                if span is None:
                    return
                if isinstance(frame, VADUserStartedSpeakingFrame):
                    if state.primary_span is not None and span[0] <= state.primary_span[0]:
                        state.abort("primary_span_replay")
                        return
                    state.primary_span, state.primary_stopped, state.primary_final = span, None, False
                    state.interrupt()
                elif state.primary_span is None or span[0] != state.primary_span[0]:
                    # A finalized old primary response cannot attach to new VAD.
                    state.stale_frames += 1
                    return
                elif isinstance(frame, VADUserStoppedSpeakingFrame):
                    if state.primary_stopped is not None:
                        state.abort("primary_stop_replay")
                        return
                    state.primary_stopped = span
            if isinstance(frame, TranscriptionFrame):
                if not frame.finalized:
                    return
                if span != state.primary_stopped or state.primary_final:
                    state.abort("primary_scope_mismatch")
                    return
                state.primary_final = True
                if not isinstance(frame.text, str) or len(frame.text.encode()) > 8000:
                    state.abort("invalid_primary_transcript")
                    return
                state.transcripts += 1
                if state.transcripts > state.limits.turns:
                    state.abort("turn_limit")
                    return
                state.emit({"type": "user_message", "from_text": False, "interim": False,
                            "message": {"role": "user", "content": frame.text}, "models": {},
                            "time": {"begin": span[2] / 16, "end": span[3] / 16}})
        await self.push_frame(frame, direction)


class TurnBridge(FrameProcessor):
    def __init__(self, session):
        super().__init__()
        self.session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            self.session.new_turn(frame)
            if self.session.closed or self.session.failure.is_set():
                return
        # Failure must not swallow CancelFrame/EndFrame/flush/control traffic:
        # the framework has to reach every owned processor to close it.
        await self.push_frame(frame, direction)


class ProviderBridge(FrameProcessor):
    def __init__(self, session):
        super().__init__()
        self.session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        state = self.session
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.UPSTREAM:
            state.new_turn(frame, continuation=True)
            if state.closed or state.failure.is_set():
                return
        if isinstance(frame, InterruptionFrame):
            state.interrupt()
        if isinstance(frame, ErrorFrame):
            state.abort("provider_error")
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame)):
            token = state.accept_output(frame)
            if token is None:
                return
            if isinstance(frame, LLMFullResponseStartFrame):
                state.response_has_text = False
            if isinstance(frame, LLMTextFrame):
                if not isinstance(frame.text, str) or len(frame.text.encode()) > 8000:
                    state.abort("invalid_assistant_text")
                    return
                state.response_has_text = True
                state.emit({"type": "assistant_message", "id": token, "from_text": False,
                            "is_quick_response": False, "message": {"role": "assistant", "content": frame.text},
                            "models": {}}, token)
        await self.push_frame(frame, direction)


class WireOutput(FrameProcessor):
    def __init__(self, session):
        super().__init__()
        self.session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        state = self.session
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (TTSAudioRawFrame, TTSTextFrame, TTSStartedFrame, TTSStoppedFrame, LLMFullResponseEndFrame)):
            token = state.accept_output(frame)
            if token is None:
                return
            if isinstance(frame, TTSAudioRawFrame):
                if (frame.sample_rate != 48000 or frame.num_channels != 1 or not isinstance(frame.audio, bytes)
                        or not 0 < len(frame.audio) <= 9600 or len(frame.audio) % 2):
                    state.abort("unsupported_tts_audio")
                    return
                output = io.BytesIO()
                with wave.open(output, "wb") as wav:
                    wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                    wav.writeframes(frame.audio)
                state.emit({"type": "audio_output", "id": token, "index": state.audio_index,
                            "data": base64.b64encode(output.getvalue()).decode()}, token)
                state.audio_index += 1
            elif isinstance(frame, LLMFullResponseEndFrame):
                # Tool-only LLM responses continue later; only final spoken
                # responses end the public assistant turn.
                if state.response_has_text:
                    state.emit({"type": "assistant_end"}, token)
                    state.active_finished = True
        await self.push_frame(frame, direction)


class WireTransport(BaseTransport):
    def __init__(self, session):
        super().__init__()
        self.incoming, self.outgoing = WireInput(session), WireOutput(session)

    def input(self):
        return self.incoming

    def output(self):
        return self.outgoing


class Gateway:
    def __init__(self, enabled, authenticate, providers, config, limits):
        self.enabled, self.authenticate, self.providers = enabled, authenticate, providers
        self.config, self.limits = config, limits
        self.active = {}
        self.sessions = {}
        self.stopping = False
        self.quarantined = False
        self.reports = deque(maxlen=32)
        self.seen_processors = weakref.WeakSet()

    async def handle(self, request):
        if not self.enabled or self.stopping:
            return web.Response(status=404)
        if self.quarantined:
            return web.Response(status=503, text="provider_cleanup_unresolved")
        # Query auth is required for the pinned browser-style handshake; never
        # assume Hume credentials work. Host application must redact URLs in
        # proxy/access logs. This module never logs or stores credential values.
        permitted = {"api_key", "fernSdkLanguage", "fernSdkVersion"}
        if (set(request.query) - permitted or any(len(request.query.getall(k)) != 1 for k in request.query)
                or len(request.query_string) > 2048):
            return web.Response(status=400, text="unsupported_handshake")
        credential = request.query.get("api_key", "")
        if not 16 <= len(credential) <= 512 or not credential.isascii():
            return web.Response(status=401, text="authentication_required")
        if len(self.active) >= self.limits.sessions:
            return web.Response(status=503, text="session_capacity")
        # Reserve before awaiting auth, so simultaneous handshakes are bounded.
        slot = object()
        self.active[slot] = asyncio.current_task()
        session = None
        try:
            try:
                async with asyncio.timeout(self.limits.settings_seconds):
                    accepted = await self.authenticate(credential)
                if accepted is not True:
                    return web.Response(status=401, text="authentication_refused")
            except (Exception, asyncio.CancelledError) as error:
                if isinstance(error, asyncio.CancelledError):
                    raise
                return web.Response(status=401, text="authentication_refused")
            finally:
                credential = None
            ws = web.WebSocketResponse(max_msg_size=16_384, compress=False, timeout=self.limits.send_seconds)
            await ws.prepare(request)
            session = Session(ws, self.limits)
            self.sessions[slot] = session
            try:
                async with asyncio.timeout(self.limits.settings_seconds):
                    first = await ws.receive()
                require(first.type == WSMsgType.TEXT, "settings_required")
                configured = session_config(message_json(first.data), self.config)
                # A returned or throwing factory may have acquired work. Until
                # a usable, non-reused finalizer is owned, closure is unknown.
                session.acquisition_uncertain = True
                bundle = self.providers(configured)
                require(isinstance(bundle, Providers) and callable(bundle.close), "invalid_provider_bundle")
                processors = (bundle.stt, bundle.llm, bundle.tts, bundle.vad)
                # Never call a reused bundle's close: it may belong to another
                # active connection. Retain uncertainty and quarantine instead.
                require(not any(isinstance(p, FrameProcessor) and p in self.seen_processors for p in processors), "provider_reuse")
                session.bundle = bundle
                session.acquisition_uncertain = False
                require(all(isinstance(p, FrameProcessor) for p in processors) and isinstance(bundle.llm, LLMService)
                        and len({id(p) for p in processors}) == 4, "invalid_provider_bundle")
                self.seen_processors.update(processors)
                await session.run(bundle, configured)
            except ProfileError as error:
                session.abort(str(error))
            except TimeoutError:
                session.abort("settings_deadline")
            except asyncio.CancelledError:
                session.abort("cancelled")
                raise
            except Exception:
                session.abort("session_failed")
            finally:
                if not session.closed:
                    await joined(asyncio.create_task(session.shutdown()))
            return ws
        finally:
            if session is not None:
                if not session.cleanup_complete:
                    self.quarantined = True
                self.reports.append({"generation": session.generation, "profile": PROFILE,
                                     "status": "closed" if session.code is None else session.code,
                                     "turns": session.turns, "input_bytes": session.input_bytes,
                                     "output_bytes": session.output_bytes, "stale_frames": session.stale_frames,
                                     "all_owned_tasks_done": all(task.done() for task in session.tasks),
                                     "provider_cleanup_joined": session.cleanup_complete,
                                     "provider_billing": "unreconciled"})
            del self.active[slot]
            self.sessions.pop(slot, None)

    async def shutdown(self, _app):
        self.stopping = True
        tasks = tuple(task for task in self.active.values() if task is not asyncio.current_task())
        for task in tasks:
            task.cancel()
        if tasks:
            await joined(asyncio.ensure_future(asyncio.gather(*tasks, return_exceptions=True)))


GATEWAY = web.AppKey("oruk_evi_pcm_gateway", Gateway)


def create_gateway(*, enabled=False, authenticate: Callable[[str], Awaitable[bool]] | None = None,
                   providers: Callable[[Config], Providers] | None = None,
                   config=Config(greeting=""), limits=GatewayLimits()):
    """Construct an unbound local profile. The caller owns its HTTP listener.

    Explicit trusted auth/provider injection is mandatory when enabled. The
    callbacks must not retain keys, log URLs or start unowned background work.
    No CLI, environment discovery, cloud binding, default provider or resume.
    """
    if type(enabled) is not bool or not isinstance(config, Config) or not isinstance(limits, GatewayLimits):
        raise ValueError("invalid_gateway_configuration")
    if enabled and (not callable(authenticate) or not callable(providers)):
        raise ValueError("explicit_auth_and_providers_required")
    gateway = Gateway(enabled, authenticate, providers, config, limits)
    app = web.Application(client_max_size=16_384)
    app[GATEWAY] = gateway
    app.router.add_get("/v0/evi/chat", gateway.handle)
    app.on_shutdown.append(gateway.shutdown)
    return app
