"""Named deterministic providers; all media is synthetic and networking is local."""

import asyncio
import copy
import ipaddress
import socket

import pytest
from pipecat.frames.frames import (
    FunctionCallFromLLM,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMAssistantPushAggregationFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from examples.hume_evi.policy import SCOPE_METADATA


@pytest.fixture
def loopback_only(monkeypatch):
    """Fail before DNS/connect for every non-loopback IPv4/IPv6 destination."""
    original_connect, original_ex, original_dns = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
    )

    def check(host):
        if host == "localhost":
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError(
            "Non-loopback network access prohibited by deterministic test"
        )

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            check(address[0])
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            check(address[0])
        return original_ex(sock, address)

    def dns(host, *args, **kwargs):
        if host is not None:
            check(host)
        return original_dns(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", dns)


async def eventually(predicate, timeout=4):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class PassThrough(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class SyntheticPrimarySTT(PassThrough):
    def __init__(self, texts=("Primary words one.", "Primary words two.")):
        super().__init__()
        self.texts = iter(texts)
        self.pcm = bytearray()
        self.finals = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, InputAudioRawFrame):
            self.pcm.extend(frame.audio)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            final = TranscriptionFrame(
                next(self.texts),
                "synthetic-primary-user",
                "2026-10-03T00:00:00Z",
                finalized=True,
            )
            if SCOPE_METADATA in frame.metadata:
                final.metadata[SCOPE_METADATA] = frame.metadata[SCOPE_METADATA]
            self.finals.append(final)
            await self.push_frame(final)


class SyntheticLLM(LLMService):
    """Uses real LLMService tool dispatch; only completion generation is synthetic."""

    def __init__(self, tool_args=None):
        super().__init__(
            settings=LLMSettings(
                model="synthetic",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
            )
        )
        self.inputs = []
        self.tool_args = tool_args
        self.called_tool = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.inputs.append(copy.deepcopy(frame.context.get_messages()))
            await self.push_frame(LLMFullResponseStartFrame())
            if self.tool_args is not None and not self.called_tool:
                self.called_tool = True
                await self.run_function_calls(
                    [
                        FunctionCallFromLLM(
                            "lookup_demo_order",
                            "synthetic-call-1",
                            self.tool_args,
                            frame.context,
                        )
                    ]
                )
            else:
                await self.push_frame(
                    LLMTextFrame(f"Synthetic reply {len(self.inputs)}.")
                )
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class SyntheticTTS(FrameProcessor):
    def __init__(self, chunks=1):
        super().__init__()
        self.texts = []
        self.chunks = chunks

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMTextFrame, TTSSpeakFrame)):
            self.texts.append(frame.text)
            marker = len(self.texts).to_bytes(2, "little")
            await self.push_frame(TTSStartedFrame())
            for _ in range(self.chunks):
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=marker * 480, sample_rate=24000, num_channels=1
                    )
                )
            await self.push_frame(
                TTSTextFrame(text=frame.text, aggregated_by="sentence")
            )
            await self.push_frame(TTSStoppedFrame())
            if isinstance(frame, TTSSpeakFrame):
                await self.push_frame(LLMAssistantPushAggregationFrame())
        else:
            await self.push_frame(frame, direction)


class SyntheticPlayback(BaseOutputTransport):
    """Real Pipecat output buffering/interruptions with a delayed synthetic sink."""

    def __init__(self, delay=0):
        super().__init__(
            TransportParams(audio_out_enabled=True, audio_out_sample_rate=24000)
        )
        self.written = []
        self.delay = delay
        self.cancelled_writes = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled_writes += 1
            raise
        self.written.append(frame.audio)
        return True


class SyntheticTransport(BaseTransport):
    def __init__(self, delay=0):
        super().__init__()
        self.incoming, self.outgoing = PassThrough(), SyntheticPlayback(delay)

    def input(self):
        return self.incoming

    def output(self):
        return self.outgoing
