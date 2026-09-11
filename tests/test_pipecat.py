from __future__ import annotations

import asyncio
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
import soxr
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    MetricsFrame,
    STTMuteFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData, TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings
from pipecat.tests.utils import run_test as pipecat_run_test
from pipecat.workers.runner import WorkerRunner

from pipecat_oruk import (
    OrukCommitFrame,
    OrukPhraseEmotionFrame,
    OrukSTTService,
    OrukTurnCompletedFrame,
)
from pipecat_oruk.realtime import RealtimeOptions

SPEECH_FIXTURE = (
    Path(__file__).resolve().parent / "audio/angry.wav"
)


async def run_test(*args, **kwargs):
    # Pipecat's first pipeline prewarms imports; its utility's 1s default is too short.
    async with asyncio.timeout(20):
        return await pipecat_run_test(*args, start_timeout=8, **kwargs)


def audio(pcm: bytes, rate: int = 16_000, user: str = "alice") -> UserAudioRawFrame:
    return UserAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1, user_id=user)


def chunks(pcm: bytes, rate: int = 16_000, user: str = "alice") -> list[UserAudioRawFrame]:
    size = rate // 50 * 2
    return [audio(pcm[i : i + size], rate, user) for i in range(0, len(pcm), size)]


def service(server, **kwargs) -> OrukSTTService:
    return OrukSTTService(
        api_key="local-test-key",
        endpoint=server.endpoint,
        connect_timeout=1,
        retry_interval=0,
        shutdown_timeout=2,
        **kwargs,
    )


class Capture(FrameProcessor):
    def __init__(self, direction):
        super().__init__(enable_direct_mode=True)
        self.direction = direction
        self.frames = []
        self.changed = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == self.direction:
            self.frames.append(frame)
            self.changed.set()
        await self.push_frame(frame, direction)

    async def wait_for(self, predicate):
        async with asyncio.timeout(4):
            while True:
                for frame in self.frames:
                    if predicate(frame):
                        return frame
                self.changed.clear()
                await self.changed.wait()


@asynccontextmanager
async def running(processors, *, sample_rate=16_000, metrics=False):
    upstream, downstream = Capture(FrameDirection.UPSTREAM), Capture(FrameDirection.DOWNSTREAM)
    pipeline = Pipeline([upstream, *processors, downstream])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            enable_metrics=metrics,
            enable_usage_metrics=metrics,
            report_only_initial_ttfb=False,
        ),
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
        processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(*_):
        started.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(started.wait(), 5)
        yield worker, downstream, upstream
        if not task.done():
            await worker.queue_frame(EndFrame())
        await asyncio.wait_for(asyncio.shield(task), 5)
    finally:
        if not task.done():
            await worker.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("wait", [False, True])
async def test_real_pipeline_two_turns_and_late_emotion_identity(gateway, wait):
    first = np.arange(640, dtype=np.int16).tobytes()
    second = np.arange(330, dtype=np.int16).tobytes()
    async with gateway() as server:
        stt = service(server, vad=False, wait_for_emotions=wait)
        down, up = await run_test(
            stt,
            frames_to_send=[
                *chunks(first),
                OrukCommitFrame(),
                *chunks(second, user="bob"),
                OrukCommitFrame(),
            ],
        )
        assert [bytes(a) for a in server.audio] == [first, second]
        assert server.auth_headers == ["Bearer local-test-key"] * 2
        finals = [f for f in down if isinstance(f, TranscriptionFrame)]
        completed = [f for f in down if isinstance(f, OrukTurnCompletedFrame)]
        assert [f.text for f in finals] == ["Final turn 0.", "Final turn 1."]
        assert [f.user_id for f in finals] == ["alice", "bob"]
        assert all(f.finalized and f.language is None for f in finals)
        assert len(completed) == 2
        assert len({f.turn_id for f in completed}) == 2
        assert completed[1].audio_offset == len(first) / 32000
        assert all(len(f.metadata["oruk"]["phrases"]) == int(wait) for f in finals)
        completed[0].result.phrases[0]["emotions"][0]["score"] = 0
        if wait:
            assert finals[0].metadata["oruk"]["phrases"][0]["emotions"][0]["score"] == 0.7
        assert not any(isinstance(f, ErrorFrame) for f in up)
        assert stt._http is None and stt._worker is None


async def test_interim_before_commit_and_interruption_preserves_input(gateway):
    async with gateway() as server:
        stt = service(server, vad=False)
        async with running([stt]) as (worker, down, _):
            await worker.queue_frame(audio(b"\x00\x01" * 320))
            interim = await down.wait_for(lambda f: isinstance(f, InterimTranscriptionFrame))
            assert interim.text.startswith("Provisional")
            assert not any(isinstance(f, TranscriptionFrame) for f in down.frames)
            await worker.queue_frame(InterruptionFrame())
            await worker.queue_frame(audio(b"\x01\x01" * 320))
            await worker.queue_frame(OrukCommitFrame())
            await down.wait_for(lambda f: isinstance(f, OrukTurnCompletedFrame))
        assert bytes(server.audio[0]) == b"\x00\x01" * 320 + b"\x01\x01" * 320


async def test_vad_prefix_silence_gating_and_partial_frame(gateway):
    prefix = np.arange(16000, dtype=np.int16).tobytes()
    tail = b"\x13\x01" * 647
    async with gateway() as server:
        down, up = await run_test(
            service(server),
            frames_to_send=[
                *chunks(prefix),
                VADUserStartedSpeakingFrame(start_secs=0.2),
                *chunks(tail),
                VADUserStoppedSpeakingFrame(stop_secs=0.4),
                *chunks(b"\0\0" * 16000),
            ],
        )
        assert [bytes(a) for a in server.audio] == [prefix[-16000:] + tail]
        result = next(f for f in down if isinstance(f, OrukTurnCompletedFrame))
        assert result.audio_offset == 0.5
        assert not any(isinstance(f, ErrorFrame) for f in up)


@pytest.mark.parametrize("rate", [8000, 24000, 44100, 48000, 96000])
async def test_streaming_resampler_flushes_short_tail_for_each_turn(gateway, rate):
    samples = np.arange(int(rate * 0.17))
    waveform = (np.sin(2 * np.pi * 300 * samples / rate) * 9000).astype(np.int16)
    expected = soxr.resample(waveform, rate, 16000, quality="VHQ").astype(np.int32)
    async with gateway() as server:
        down, up = await run_test(
            service(server, vad=False),
            pipeline_params=PipelineParams(audio_in_sample_rate=rate),
            frames_to_send=[
                *chunks(waveform.tobytes(), rate),
                OrukCommitFrame(),
                *chunks(waveform.tobytes(), rate),
            ],
        )
        assert len([f for f in down if isinstance(f, OrukTurnCompletedFrame)]) == 2
        assert not any(isinstance(f, ErrorFrame) for f in up)
        for pcm in server.audio:
            got = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
            assert len(got) == len(expected) == 2720
            assert np.max(np.abs(got - expected)) <= 2  # Independent SoX int16 dither.


async def test_auth_rejected_once_and_redacted(gateway):
    async with gateway(fail_upgrades=[401]) as server:
        stt = service(server, vad=False)
        async with running([stt]) as (worker, _, up):
            await worker.queue_frame(audio(b"\0\0" * 320))
            error = await up.wait_for(lambda f: isinstance(f, ErrorFrame))
            assert error.category.value == "authentication"
            assert "local-test-key" not in error.error
            assert not stt.is_usable
        assert server.requests == 1 and server.audio == []


@pytest.mark.parametrize(
    "mode",
    [
        "disconnect",
        "invalid_json",
        "bad_score",
        "final_only",
        "usage_only",
        "error_after_usage",
        "hang",
    ],
)
async def test_failed_turn_is_not_completed_or_replayed(gateway, mode):
    async with gateway(mode=mode) as server:
        stt = service(
            server, vad=False, wait_for_emotions=True, options=RealtimeOptions(finish_timeout=0.1)
        )
        async with running([stt]) as (worker, down, up):
            await worker.queue_frame(audio(b"\0\0" * 320))
            await worker.queue_frame(OrukCommitFrame())
            await up.wait_for(lambda f: isinstance(f, ErrorFrame))
            assert not any(
                isinstance(f, (TranscriptionFrame, OrukTurnCompletedFrame)) for f in down.frames
            )
            assert not stt.is_usable
        assert server.requests == 1


async def test_pre_audio_capacity_retry_does_not_duplicate_pcm(gateway):
    async with gateway(fail_upgrades=[503, 429]) as server:
        down, _ = await run_test(
            service(server, vad=False), frames_to_send=[audio(b"\x12\x01" * 320)]
        )
        assert server.requests == 3
        assert [bytes(a) for a in server.audio] == [b"\x12\x01" * 320]
        assert len([f for f in down if isinstance(f, OrukTurnCompletedFrame)]) == 1


async def test_phrase_failure_does_not_fail_transcription(gateway):
    async with gateway(mode="phrase_failure") as server:
        down, up = await run_test(
            service(server, vad=False, wait_for_emotions=True),
            frames_to_send=[audio(b"\0\0" * 320)],
        )
        phrase = next(f for f in down if isinstance(f, OrukPhraseEmotionFrame))
        assert phrase.event["type"].endswith(".failed")
        assert len([f for f in down if isinstance(f, TranscriptionFrame)]) == 1
        assert not any(isinstance(f, ErrorFrame) for f in up)


async def test_duplicate_events_emit_once(gateway):
    async with gateway(mode="duplicate") as server:
        down, _ = await run_test(service(server, vad=False), frames_to_send=[audio(b"\0\0" * 320)])
        assert len([f for f in down if isinstance(f, TranscriptionFrame)]) == 1
        assert len([f for f in down if isinstance(f, OrukPhraseEmotionFrame)]) == 1


@pytest.mark.parametrize(
    "fault",
    ["stereo", "sample_rate", "odd", "oversize", "tracks", "capacity", "turn_length", "prefix"],
)
async def test_input_faults_fail_explicitly(gateway, fault):
    kwargs = {"vad": False}
    frames = [audio(b"\0\0" * 320)]
    if fault == "stereo":
        frames = [InputAudioRawFrame(audio=b"\0" * 1280, sample_rate=16000, num_channels=2)]
    elif fault == "sample_rate":
        frames = [audio(b"\0\0" * 320, 48000)]
    elif fault == "odd":
        frames = [audio(b"\0")]
    elif fault == "oversize":
        frames = [audio(b"\0\0" * 16001)]
    elif fault == "tracks":
        frames.append(audio(b"\0\0" * 320, user="bob"))
    elif fault == "capacity":
        kwargs["max_buffer_seconds"] = 0.01
    elif fault == "turn_length":
        kwargs["options"] = RealtimeOptions(max_turn_seconds=0.01)
    elif fault == "prefix":
        kwargs["vad"] = True
        frames.append(VADUserStartedSpeakingFrame(start_secs=0.6))
    async with gateway() as server:
        stt = service(server, **kwargs)
        async with running([stt]) as (worker, _, up):
            await worker.queue_frames(frames)
            await up.wait_for(lambda f: isinstance(f, ErrorFrame))
            assert not stt.is_usable


async def test_mute_drops_audio_and_rearms_vad(gateway):
    async with gateway() as server:
        down, _ = await run_test(
            service(server),
            frames_to_send=[
                STTMuteFrame(mute=True),
                *chunks(b"\x01\0" * 16000),
                VADUserStartedSpeakingFrame(),
                STTMuteFrame(mute=False),
                audio(b"\x02\0" * 320),
                VADUserStartedSpeakingFrame(),
                audio(b"\x03\0" * 320),
                VADUserStoppedSpeakingFrame(),
            ],
        )
        assert [bytes(a) for a in server.audio] == [b"\x02\0" * 320 + b"\x03\0" * 320]
        completed = next(f for f in down if isinstance(f, OrukTurnCompletedFrame))
        assert completed.audio_offset == 1.0


async def test_cancel_closes_active_socket_without_a_final(gateway):
    async with gateway(mode="hang") as server:
        stt = service(server, vad=False)
        async with running([stt]) as (worker, down, _):
            await worker.queue_frame(audio(b"\0\0" * 320))
            await asyncio.wait_for(server.first_audio.wait(), 2)
            await worker.queue_frame(CancelFrame())
            await asyncio.wait_for(server.closed.wait(), 2)
            assert not any(isinstance(f, OrukTurnCompletedFrame) for f in down.frames)
        assert stt._http is None and stt._worker is None
        assert stt._queued_bytes == 0


async def test_slow_emotion_handler_is_bounded_and_does_not_complete(gateway):
    async with gateway() as server:
        stt = service(server, vad=False, wait_for_emotions=True, event_handler_timeout=0.02)

        @stt.event_handler("on_phrase_emotion")
        async def slow_handler(*_):
            await asyncio.Event().wait()

        async with running([stt]) as (worker, down, up):
            await worker.queue_frames([audio(b"\0\0" * 320), OrukCommitFrame()])
            error = await up.wait_for(lambda f: isinstance(f, ErrorFrame))
            assert "event_handler_timeout" in error.error
            assert not any(isinstance(f, OrukTurnCompletedFrame) for f in down.frames)


async def test_pending_turn_limit_is_explicit(gateway):
    async with gateway(mode="hang") as server:
        stt = service(server, vad=False, max_pending_turns=1)
        async with running([stt]) as (worker, _, up):
            await worker.queue_frames(
                [
                    audio(b"\0\0" * 320),
                    OrukCommitFrame(),
                    audio(b"\1\0" * 320),
                ]
            )
            error = await up.wait_for(lambda f: isinstance(f, ErrorFrame))
            assert "too_many_pending_turns" in error.error


async def test_upstream_vad_copy_does_not_open_an_input_turn(gateway):
    async with gateway() as server:
        stt = service(server)
        async with running([stt]) as (worker, down, up):
            await worker.queue_frame(VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await up.wait_for(lambda f: isinstance(f, VADUserStartedSpeakingFrame))
            await worker.queue_frame(audio(b"\0\0" * 320))
            await down.wait_for(lambda f: isinstance(f, InputAudioRawFrame))
            assert not stt._speech_active and stt._current is None
        assert server.requests == 0


async def test_settings_change_between_turns_has_immutable_per_turn_options(gateway):
    async with gateway() as server:
        stt = service(server, vad=False)
        async with running([stt]) as (worker, down, up):
            await worker.queue_frame(audio(b"\0\0" * 320))
            await down.wait_for(lambda f: isinstance(f, InterimTranscriptionFrame))
            await worker.queue_frame(STTUpdateSettingsFrame(delta=STTSettings(language="fr")))
            await up.wait_for(lambda f: isinstance(f, ErrorFrame))
            assert stt.is_usable and stt._options.language == "auto"
            await worker.queue_frame(OrukCommitFrame())
            await down.wait_for(lambda f: isinstance(f, OrukTurnCompletedFrame))
            updated = asyncio.Event()

            @stt.event_handler("on_after_process_frame")
            async def on_update(_service, frame):
                if isinstance(frame, STTUpdateSettingsFrame):
                    updated.set()

            await worker.queue_frame(STTUpdateSettingsFrame(delta=STTSettings(language="fr")))
            await asyncio.wait_for(updated.wait(), 2)
            await worker.queue_frame(audio(b"\0\0" * 320))
        assert [c["language"] for c in server.configs] == ["auto", "fr"]


async def test_receipt_usage_and_turn_specific_latency_metrics(gateway):
    async with gateway() as server:
        stt = service(server)
        async with running([stt], metrics=True) as (worker, down, _):
            await worker.queue_frames([audio(b"\0\0" * 320), VADUserStartedSpeakingFrame()])
            await asyncio.wait_for(server.first_audio.wait(), 2)
            await worker.queue_frame(
                VADUserStoppedSpeakingFrame(stop_secs=0.4, timestamp=time.time())
            )
            completed = await down.wait_for(lambda f: isinstance(f, OrukTurnCompletedFrame))
            metric_frames = [f for f in down.frames if isinstance(f, MetricsFrame)]
            usage = [m for f in metric_frames for m in f.data if isinstance(m, STTUsageMetricsData)]
            latency = [
                f
                for f in metric_frames
                if any(isinstance(m, TTFBMetricsData) and m.value > 0 for m in f.data)
            ]
            assert len(usage) == 1 and usage[0].value.audio_seconds == 0.02
            assert len(latency) == 1
            assert latency[0].metadata["oruk"]["turn_id"] == completed.turn_id


async def test_real_silero_pipeline_preserves_selected_recording_bytes(gateway):
    with wave.open(str(SPEECH_FIXTURE), "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        speech = wav.readframes(wav.getnframes())
    pcm = b"\0\0" * 16000 + speech + b"\0\0" * 24000
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.4))
    )
    oracle = Capture(FrameDirection.DOWNSTREAM)
    async with gateway() as server:
        stt = service(server)
        down, up = await run_test(Pipeline([vad, oracle, stt]), frames_to_send=chunks(pcm))
        expected, received, start, active = [], bytearray(), 0, False
        for frame in oracle.frames:
            if isinstance(frame, InputAudioRawFrame):
                received.extend(frame.audio)
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                start, active = max(0, len(received) - 16000), True
            elif isinstance(frame, VADUserStoppedSpeakingFrame) and active:
                expected.append(bytes(received[start:]))
                active = False
        if active:
            expected.append(bytes(received[start:]))
        assert len(expected) >= 1 and expected[0].strip(b"\0")
        assert [bytes(a) for a in server.audio] == expected
        assert len([f for f in down if isinstance(f, OrukTurnCompletedFrame)]) == len(expected)
        assert not any(isinstance(f, ErrorFrame) for f in up)
