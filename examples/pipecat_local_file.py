"""Transcribe a mono 16 kHz PCM16 WAV through local Orukeet and Pipecat VAD."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import (
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner
from pipecat_oruk.local import OrukeetSTTService


class PrintResults(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, TranscriptionFrame
        ):
            print(
                json.dumps({"text": frame.text, "finalized": frame.finalized}),
                flush=True,
            )
        await self.push_frame(frame, direction)


async def transcribe(path: str, offline: bool) -> None:
    with wave.open(path, "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
            16000,
            1,
            2,
        ):
            raise ValueError("Expected a mono 16 kHz PCM16 WAV")
        if not 0 < source.getnframes() <= 16000 * 60:
            raise ValueError("This file example accepts up to 60 seconds of audio")
        pcm = source.readframes(source.getnframes())

    stt = OrukeetSTTService(local_files_only=offline)
    await stt.prewarm()
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.3))
    )
    worker = PipelineWorker(
        Pipeline([vad, stt, PrintResults()]),
        params=PipelineParams(audio_in_sample_rate=16000),
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
        processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
    )
    failures: list[str] = []

    @stt.event_handler("on_error")
    async def on_error(_service, frame):
        failures.append(frame.error)

    @worker.event_handler("on_pipeline_started")
    async def on_started(*_):
        # Trailing silence lets VAD close the last utterance before EndFrame.
        audio = pcm + bytes(32000)
        for offset in range(0, len(audio), 640):
            await worker.queue_frame(
                InputAudioRawFrame(
                    audio=audio[offset : offset + 640],
                    sample_rate=16000,
                    num_channels=1,
                )
            )
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()
    if failures:
        raise RuntimeError(failures[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument(
        "--offline", action="store_true", help="Require a complete Hugging Face cache"
    )
    args = parser.parse_args()
    try:
        asyncio.run(transcribe(args.wav, args.offline))
    except (ValueError, RuntimeError, OSError, wave.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
