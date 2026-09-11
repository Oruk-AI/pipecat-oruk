from __future__ import annotations

import asyncio
import json
import os
import runpy
import sys
import wave
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/pipecat_stream_file.py"
AGENT = EXAMPLE.with_name("pipecat_agent.py")


@pytest.fixture(scope="module")
def pipecat_agent():
    return runpy.run_path(str(AGENT))


def make_wav(path, *, channels=1):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x12\x01" * 1600 * channels)


async def command(*args, endpoint=None):
    env = {**os.environ, "ORUK_API_KEY": "example-local-key"}
    if endpoint:
        env["ORUK_REALTIME_URL"] = endpoint
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(EXAMPLE),
        *map(str, args),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    return process.returncode, stdout.decode(), stderr.decode()


@pytest.mark.parametrize("wait", [False, True])
async def test_stream_file_executes_actual_pipeline(gateway, tmp_path, wait):
    path = tmp_path / "test.wav"
    make_wav(path)
    async with gateway() as server:
        args = [path, "--fast"] + (["--wait-for-emotions"] if wait else [])
        code, stdout, stderr = await command(*args, endpoint=server.endpoint)
        assert code == 0, stderr
        rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
        assert {row["type"] for row in rows} == {"interim", "final", "provider_event", "complete"}
        assert next(row for row in rows if row["type"] == "final")["text"] == "Final turn 0."
        assert bytes(server.audio[0]) == b"\x12\x01" * 1600


async def test_stream_file_failure_exits_nonzero_and_does_not_leak_key(gateway, tmp_path):
    path = tmp_path / "test.wav"
    make_wav(path)
    async with gateway(fail_upgrades=[401]) as server:
        code, stdout, stderr = await command(path, "--fast", endpoint=server.endpoint)
        assert code == 1
        assert "websocket_upgrade_failed" in stderr
        assert "example-local-key" not in stdout + stderr


async def test_stream_file_rejects_stereo_before_connecting(gateway, tmp_path):
    path = tmp_path / "stereo.wav"
    make_wav(path, channels=2)
    async with gateway() as server:
        code, _, stderr = await command(path, endpoint=server.endpoint)
        assert code == 1 and "mono PCM16" in stderr
        assert server.requests == 0


async def test_help_does_not_connect():
    code, stdout, _ = await command("--help")
    assert code == 0 and "--wait-for-emotions" in stdout


async def test_real_user_aggregator_receives_emotions_and_clears_them_for_text(
    pipecat_agent, gateway
):
    import copy

    from pipecat.frames.frames import (
        LLMContextFrame,
        LLMMessagesAppendFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from test_pipecat import audio, running, service

    class LLMInput(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.messages = asyncio.Queue()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
                self.messages.put_nowait(copy.deepcopy(frame.context.get_messages()))
            await self.push_frame(frame, direction)

    context = LLMContext([{"role": "system", "content": "Keep answers short."}])
    aggregator = pipecat_agent["EmotionAwareUserAggregator"](context)
    llm_input = LLMInput()
    async with gateway() as server:
        stt = service(server, wait_for_emotions=True)
        async with running([stt, aggregator, llm_input]) as (worker, _, _):
            await worker.queue_frames(
                [
                    audio(b"\0\0" * 320),
                    VADUserStartedSpeakingFrame(),
                    audio(b"\1\0" * 320),
                    VADUserStoppedSpeakingFrame(stop_secs=0.2),
                ]
            )
            messages = await asyncio.wait_for(llm_input.messages.get(), 4)
            assert messages[-2] == {"role": "user", "content": "Final turn 0."}
            assert messages[-1]["role"] == "system"
            assert '"happy":0.7' in messages[-1]["content"]
            assert "provisional phrase" not in messages[-1]["content"].lower()
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "A typed message."}],
                    run_llm=True,
                )
            )
            typed = await asyncio.wait_for(llm_input.messages.get(), 3)
            assert typed[-1]["content"] == "A typed message."
            assert not any("Oruk estimates" in m.get("content", "") for m in typed)


def test_agent_uses_only_current_known_labels_and_requires_explicit_models(
    pipecat_agent, monkeypatch
):
    assert pipecat_agent["LABELS"] == {
        "angry",
        "disgusted",
        "scared",
        "happy",
        "sad",
        "surprised",
        "neutral",
    }
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="Set LLM_MODEL"):
        pipecat_agent["required"]("LLM_MODEL")


async def test_complete_agent_builder_routes_two_speech_turns_to_audio_output(
    pipecat_agent, gateway
):
    import copy

    from pipecat.frames.frames import (
        InterruptionFrame,
        LLMContextFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        OutputAudioRawFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.transports.base_transport import BaseTransport
    from test_pipecat import SPEECH_FIXTURE, chunks, running, service

    class Passthrough(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    class Transport(BaseTransport):
        def __init__(self):
            super().__init__()
            self.incoming, self.outgoing = Passthrough(), Passthrough()

        def input(self):
            return self.incoming

        def output(self):
            return self.outgoing

    class LocalLLM(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.inputs = asyncio.Queue()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
                self.inputs.put_nowait(copy.deepcopy(frame.context.get_messages()))
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame("A local test response."))
                await self.push_frame(LLMFullResponseEndFrame())
            else:
                await self.push_frame(frame, direction)

    class LocalTTS(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMTextFrame):
                await self.push_frame(
                    OutputAudioRawFrame(
                        audio=b"\0\0" * 480,
                        sample_rate=24000,
                        num_channels=1,
                    )
                )
            else:
                await self.push_frame(frame, direction)

    with wave.open(str(SPEECH_FIXTURE), "rb") as wav:
        pcm = wav.readframes(wav.getnframes()) + b"\0\0" * 24000
    async with gateway() as server:
        stt, llm = service(server, wait_for_emotions=True), LocalLLM()
        pipeline, actual_stt, user = pipecat_agent["build_pipeline"](
            Transport(),
            llm,
            LocalTTS(),
            stt=stt,
        )
        assert actual_stt is stt and user is not None
        async with running([pipeline]) as (worker, down, _):
            await worker.queue_frames(chunks(pcm))
            first = await asyncio.wait_for(llm.inputs.get(), 4)
            first_audio = await down.wait_for(lambda f: isinstance(f, OutputAudioRawFrame))
            assert first[-1]["role"] == "system" and '"happy":0.7' in first[-1]["content"]
            assert first_audio.sample_rate == 24000
            await worker.queue_frame(InterruptionFrame())
            await worker.queue_frames(chunks(pcm))
            second = await asyncio.wait_for(llm.inputs.get(), 4)
            assert second[-2]["role"] == "user"
            assert second[-2]["content"] != first[-2]["content"]
            assert sum("Oruk estimates" in m.get("content", "") for m in second) == 1
        assert stt._http is None and len(server.audio) >= 2
