"""A browser voice agent with Oruk STT and turn-matched phrase emotion context.

Run with Pipecat's development runner on localhost. Choose LLM/TTS models and
voice explicitly in the environment; this example does not guess their availability.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from pipecat_oruk import OrukSTTService
from pipecat_oruk.realtime import ENDPOINT

LABELS = {"neutral", "happy", "sad", "angry", "scared", "disgusted", "surprised"}


class EmotionAwareUserAggregator(LLMUserAggregator):
    """Add compact estimates only when they match the newly aggregated user text.

    The spoken transcript remains unchanged. No provider phrase text is promoted
    into instructions, and text-only messages cannot inherit an earlier estimate.
    """

    def __init__(self, context: LLMContext):
        super().__init__(
            context,
            params=LLMUserAggregatorParams(
                # VAD runs upstream so Oruk receives its boundaries after input audio.
                vad_analyzer=None,
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6),
                    ]
                ),
            ),
        )
        self._segments: list[tuple[str, dict[str, Any]]] = []
        self._overflow = False
        self._note: LLMContextMessage | None = None

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if frame.text.strip():
            if len(self._segments) >= 32:
                self._overflow = True
                self._segments.clear()
            if not self._overflow:
                self._segments.append((frame.text, copy.deepcopy(frame.metadata.get("oruk", {}))))
        await super()._handle_transcription(frame)

    async def push_context_frame(self, direction=FrameDirection.DOWNSTREAM):
        messages = self.context.get_messages()
        if self._note is not None:
            self.context.set_messages([m for m in messages if m is not self._note])
            self._note = None
        messages = self.context.get_messages()
        latest = next(
            (m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), None
        )
        combined = " ".join(" ".join(text.split()) for text, _ in self._segments)
        latest_content = latest.get("content") if latest else None
        notes = []
        if (
            not self._overflow
            and self._segments
            and latest
            and isinstance(latest_content, str)
            and " ".join(latest_content.split()) == combined
        ):
            for index, (_, metadata) in enumerate(self._segments):
                for phrase in metadata.get("phrases", []):
                    if not phrase.get("type", "").endswith(".completed"):
                        continue
                    scores = {
                        e["label"]: round(e["score"], 3)
                        for e in phrase["emotions"]
                        if e["label"] in LABELS
                    }
                    if scores:
                        notes.append(
                            {
                                "segment": index,
                                "start": phrase["start"],
                                "end": phrase["end"],
                                "scores": scores,
                            }
                        )
        self._segments.clear()
        self._overflow = False
        if notes:
            self._note = {
                "role": "system",
                "content": (
                    "For the user message immediately above, these are Oruk estimates of vocal "
                    "expression, not facts about feelings or calibrated probabilities. "
                    "Use them only as tentative context. Respond to what the person said; "
                    "do not announce or diagnose an emotion. Segment times are relative to each "
                    "audio segment. Estimates: " + json.dumps(notes[:32], separators=(",", ":"))
                ),
            }
            self.context.add_message(self._note)
        await super().push_context_frame(direction)


def build_pipeline(
    transport: BaseTransport,
    llm: FrameProcessor,
    tts: FrameProcessor,
    *,
    stt: OrukSTTService | None = None,
) -> tuple[Pipeline, OrukSTTService, EmotionAwareUserAggregator]:
    stt = stt or OrukSTTService(
        endpoint=os.environ.get("ORUK_REALTIME_URL", ENDPOINT),
        wait_for_emotions=True,
    )
    context = LLMContext(
        [
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Keep spoken answers brief. "
                "Tell the user you are an AI if asked; never claim to know their inner state.",
            }
        ]
    )
    user = EmotionAwareUserAggregator(context)
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.2))
    )
    return (
        Pipeline(
            [
                transport.input(),
                vad,
                stt,
                user,
                llm,
                tts,
                transport.output(),
                LLMAssistantAggregator(context),
            ]
        ),
        stt,
        user,
    )


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before starting the agent")
    return value


async def bot(runner_args: RunnerArguments) -> None:
    from pipecat.runner.utils import create_transport
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.tts import OpenAITTSService

    required("ORUK_API_KEY")
    key = required("OPENAI_API_KEY")
    llm = OpenAILLMService(
        api_key=key, settings=OpenAILLMService.Settings(model=required("LLM_MODEL"))
    )
    tts = OpenAITTSService(
        api_key=key,
        settings=OpenAITTSService.Settings(
            model=required("TTS_MODEL"),
            voice=required("TTS_VOICE"),
        ),
    )
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        },
    )
    pipeline, _, _ = build_pipeline(transport, llm, tts)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000),
        enable_rtvi=True,
        processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(*_):
        await worker.cancel()

    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint, handle_sigterm=runner_args.handle_sigterm
    )
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
