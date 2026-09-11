"""Stream a WAV through a real Pipecat pipeline and print Oruk events as JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from dataclasses import asdict

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from pipecat_oruk import OrukEventFrame, OrukSTTService, OrukTurnCompletedFrame
from pipecat_oruk.realtime import ENDPOINT, RealtimeOptions


class PrintResults(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
                print(
                    json.dumps(
                        {
                            "type": "final" if isinstance(frame, TranscriptionFrame) else "interim",
                            "text": frame.text,
                            "user_id": frame.user_id,
                            "oruk": frame.metadata.get("oruk"),
                        }
                    ),
                    flush=True,
                )
            elif isinstance(frame, OrukEventFrame):
                print(
                    json.dumps(
                        {
                            "type": "provider_event",
                            "turn_id": frame.turn_id,
                            "audio_offset": frame.audio_offset,
                            "event": frame.event,
                        }
                    ),
                    flush=True,
                )
            elif isinstance(frame, OrukTurnCompletedFrame):
                print(
                    json.dumps(
                        {
                            "type": "complete",
                            "turn_id": frame.turn_id,
                            "audio_offset": frame.audio_offset,
                            "result": asdict(frame.result),
                        }
                    ),
                    flush=True,
                )
        await self.push_frame(frame, direction)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("wav", help="Mono PCM16 WAV, 8–96 kHz, at most 300 seconds")
    result.add_argument("--fast", action="store_true", help="Send a saved clip without pacing")
    result.add_argument(
        "--vad", action="store_true", help="Use real Silero turns (8 or 16 kHz WAV)"
    )
    result.add_argument("--diarize", action="store_true", help="Request turn-local speaker labels")
    result.add_argument("--wait-for-emotions", action="store_true")
    result.add_argument("--language", default="auto")
    return result


async def stream_file(args: argparse.Namespace, wav: wave.Wave_read) -> None:
    rate = wav.getframerate()
    duration = wav.getnframes() / rate
    stt = OrukSTTService(
        endpoint=os.environ.get("ORUK_REALTIME_URL", ENDPOINT),
        vad=args.vad,
        wait_for_emotions=args.wait_for_emotions,
        options=RealtimeOptions(language=args.language, diarize=args.diarize),
        max_buffer_seconds=min(120, max(30, duration + 1)) if args.fast else 30,
    )
    processors: list[FrameProcessor] = []
    if args.vad:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.processors.audio.vad_processor import VADProcessor

        processors.append(
            VADProcessor(
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(start_secs=0.2, stop_secs=0.4),
                )
            )
        )
    pipeline = Pipeline([*processors, stt, PrintResults()])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=rate),
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
        processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
    )
    started = asyncio.Event()
    failure: list[str] = []

    @worker.event_handler("on_pipeline_started")
    async def on_started(*_):
        started.set()

    @stt.event_handler("on_error")
    async def on_error(_service, frame):
        failure.append(frame.error)

    async def feed() -> None:
        try:
            await asyncio.wait_for(started.wait(), 20)
            beginning, samples = time.monotonic(), 0
            while pcm := await asyncio.to_thread(wav.readframes, max(1, rate // 50)):
                await worker.queue_frame(
                    InputAudioRawFrame(
                        audio=pcm,
                        sample_rate=rate,
                        num_channels=1,
                    )
                )
                samples += len(pcm) // 2
                if args.fast:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(max(0, samples / rate - (time.monotonic() - beginning)))
            await worker.queue_frame(EndFrame())
        except (OSError, EOFError, wave.Error, TimeoutError):
            failure.append("Could not read the WAV or start the Pipecat pipeline")
            await worker.cancel()

    runner = WorkerRunner()
    await runner.add_workers(worker)
    feeder = asyncio.create_task(feed())
    try:
        await runner.run()
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
    if failure:
        raise RuntimeError(failure[0])


def main() -> int:
    args = parser().parse_args()
    try:
        with wave.open(args.wav, "rb") as wav:
            rate = wav.getframerate()
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
                or not 8000 <= rate <= 96000
            ):
                raise ValueError("Expected a mono PCM16 WAV at 8–96 kHz")
            duration = wav.getnframes() / rate
            if not 0 < duration <= 300:
                raise ValueError("Input must contain audio and be at most 300 seconds")
            if args.fast and duration > 110:
                raise ValueError("Use paced mode for files over 110 seconds")
            if args.vad and rate not in (8000, 16000):
                raise ValueError("Silero mode requires an 8 or 16 kHz WAV")
            asyncio.run(stream_file(args, wav))
    except (ValueError, RuntimeError, OSError, wave.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
