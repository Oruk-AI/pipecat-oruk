"""Named synthetic providers and a task-owned loopback listener, never a model."""
import asyncio
import copy
from contextlib import asynccontextmanager

from aiohttp import ClientSession, ClientTimeout, web
from pipecat.frames.frames import (
    FunctionCallFromLLM, InputAudioRawFrame, LLMContextFrame,
    LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame,
    TranscriptionFrame, TTSAudioRawFrame, TTSTextFrame, TTSStartedFrame,
    TTSStoppedFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService

from examples.hume_evi.gateway import AUDIO, TURN, TOOL, GATEWAY, Providers, create_gateway
from hume_helpers import SyntheticLLM

KEY = "local-synthetic-gateway-key-not-a-provider-key"
SETTINGS = {"type": "session_settings", "audio": {"encoding": "linear16", "sample_rate": 16000, "channels": 1}}
ALLOWED_PORTS = set()


def tagged(frame, source):
    frame.metadata.update(copy.deepcopy(source.metadata))
    return frame


class NamedVAD(FrameProcessor):
    """One synthetic input chunk is one scope; no real speech boundary claim."""
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(tagged(VADUserStartedSpeakingFrame(start_secs=0), frame))
            await self.push_frame(frame, direction)
            await self.push_frame(tagged(VADUserStoppedSpeakingFrame(stop_secs=0), frame))
        else:
            await self.push_frame(frame, direction)


class NamedSTT(FrameProcessor):
    def __init__(self, owner):
        super().__init__()
        self.owner, self.calls, self.pcm = owner, 0, bytearray()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, InputAudioRawFrame):
            self.pcm.extend(frame.audio)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self.calls += 1
            result = tagged(TranscriptionFrame(f"Primary {self.calls}.", "synthetic-user", "2026-10-04T00:00:00Z", finalized=True), frame)
            if self.owner.mode == "unscoped_stt":
                result.metadata.pop(AUDIO, None)
            if self.owner.mode == "late_stt" and self.calls == 1:
                self.owner.held.set()
                async def later():
                    await self.owner.release.wait()
                    await self.push_frame(result)
                self.owner.background.append(asyncio.create_task(later()))
            else:
                await self.push_frame(result)


class NamedLLM(SyntheticLLM):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    async def response(self, source, n):
        frames = [LLMFullResponseStartFrame(), LLMTextFrame(f"Synthetic reply {n}."), LLMFullResponseEndFrame()]
        for frame in frames:
            if self.owner.mode != "untagged_llm":
                tagged(frame, source)
            await self.push_frame(frame)

    async def process_frame(self, frame, direction):
        await LLMService.process_frame(self, frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.inputs.append(copy.deepcopy(frame.context.get_messages()))
        self.owner.tokens.append(frame.metadata[TURN])
        n = len(self.inputs)
        if self.owner.mode == "late_llm" and n == 1:
            self.owner.held.set()
            async def later():
                await self.owner.release.wait()
                await self.response(frame, n)
            self.owner.background.append(asyncio.create_task(later()))
        elif self.owner.mode == "tool" and n == 1:
            await self.push_frame(tagged(LLMFullResponseStartFrame(), frame))
            await self.run_function_calls([FunctionCallFromLLM("lookup_demo_order", frame.metadata[TOOL], {"order_id": "DEMO-100"}, frame.context)])
            await self.push_frame(tagged(LLMFullResponseEndFrame(), frame))
        else:
            await self.response(frame, n)


class NamedTTS(FrameProcessor):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.accepted_texts = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            self.accepted_texts.append(frame.text)
            for result in (
                TTSStartedFrame(),
                TTSAudioRawFrame(audio=b"\x01\x00" * 960, sample_rate=24000 if self.owner.mode == "wrong_rate" else 48000, num_channels=1),
                TTSTextFrame(text=frame.text, aggregated_by="sentence"), TTSStoppedFrame(),
            ):
                tagged(result, frame)
                if self.owner.mode == "untagged_tts":
                    result.metadata.pop(TURN, None)
                await self.push_frame(result)
        else:
            await self.push_frame(frame, direction)


class NamedProviders:
    def __init__(self, mode="normal", cleanup_gate=None):
        self.mode, self.cleanup_gate = mode, cleanup_gate
        self.held, self.release, self.closing, self.closed = (asyncio.Event() for _ in range(4))
        self.background, self.tokens = [], []
        self.stt, self.llm, self.tts, self.vad = NamedSTT(self), NamedLLM(self), NamedTTS(self), NamedVAD()

    async def close(self):
        self.closing.set()
        for task in self.background:
            task.cancel()
        await asyncio.gather(*self.background, return_exceptions=True)
        if self.cleanup_gate is not None:
            await self.cleanup_gate.wait()
        self.closed.set()
        if self.mode == "close_failure":
            raise RuntimeError("synthetic finalizer acknowledgement lost")

    def bundle(self):
        return Providers(self.stt, self.llm, self.tts, self.vad, self.close)


@asynccontextmanager
async def local_gateway(*, mode="normal", cleanup_gate=None, enabled=True, limits=None, auth=None):
    made = []
    async def authenticate(key):
        return key == KEY
    def factory(_config):
        if mode == "factory_failure":
            raise RuntimeError("synthetic unknown acquisition")
        if mode == "reuse_bundle" and made:
            return made[0].bundle()
        owner = NamedProviders(mode, cleanup_gate)
        made.append(owner)
        if mode == "invalid_bundle":
            return Providers(owner.stt, object(), owner.tts, owner.vad, owner.close)
        return owner.bundle()
    kwargs = {"enabled": enabled, "authenticate": auth or authenticate, "providers": factory}
    if limits is not None:
        kwargs["limits"] = limits
    app = create_gateway(**kwargs)
    runner = web.AppRunner(app, access_log=None, handle_signals=False, shutdown_timeout=5)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    ALLOWED_PORTS.add(port)
    url = f"http://127.0.0.1:{port}/v0/evi/chat"
    try:
        async with ClientSession(timeout=ClientTimeout(total=5), trust_env=False) as client:
            yield client, url, app[GATEWAY], made, runner
    finally:
        # A failed assertion must not leave the explicitly held synthetic owner.
        for owner in made:
            owner.release.set()
        if cleanup_gate is not None:
            cleanup_gate.set()
        await runner.cleanup()
        ALLOWED_PORTS.discard(port)
        assert not app[GATEWAY].active
        assert all(owner.closed.is_set() for owner in made)
        assert all(task.done() for owner in made for task in owner.background)
