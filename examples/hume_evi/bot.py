"""Local development runner; server-side provider keys, explicit model choices."""

import os
import json
from pathlib import Path

from .config import load_config


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before starting the agent")
    return value


def settings():
    config = load_config(
        os.environ.get(
            "HUME_STARTER_CONFIG", str(Path(__file__).with_name("config.example.json"))
        )
    )
    if config.signal_policy != "trace_only":
        raise ValueError(
            "This CLI has no verified provider audio-scope resolver; use trace_only"
        )
    if required("STT_PROVIDER") != "deepgram":
        raise ValueError(
            "The CLI implements deepgram; inject another customer STT through build_pipeline"
        )
    names = (
        "DEEPGRAM_API_KEY",
        "STT_MODEL",
        "OPENAI_API_KEY",
        "LLM_MODEL",
        "TTS_MODEL",
        "TTS_VOICE",
    )
    values = {name: required(name) for name in names}
    if config.signals_enabled:
        values["ORUK_API_KEY"] = required("ORUK_API_KEY")
    return config, values


async def bot(runner_args):
    # The development runner re-enables DEBUG after main(); disable its sinks
    # again for each session before processing any conversation or provider key.
    from loguru import logger

    logger.remove()
    # Validate everything before constructing network-capable services.
    config, values = settings()
    from pipecat.pipeline.worker import (
        PipelineParams,
        PipelineWorker,
        ProcessorUnusablePolicy,
    )
    from pipecat.runner.utils import create_transport
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.transports.base_transport import TransportParams
    from pipecat.workers.runner import WorkerRunner
    from .pipeline import build_pipeline

    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True, audio_out_enabled=True
            )
        },
    )
    session = build_pipeline(
        transport,
        DeepgramSTTService(
            api_key=values["DEEPGRAM_API_KEY"],
            settings=DeepgramSTTService.Settings(model=values["STT_MODEL"]),
        ),
        OpenAILLMService(
            api_key=values["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(model=values["LLM_MODEL"]),
        ),
        OpenAITTSService(
            api_key=values["OPENAI_API_KEY"],
            settings=OpenAITTSService.Settings(
                model=values["TTS_MODEL"], voice=values["TTS_VOICE"]
            ),
        ),
        config=config,
        api_key=values.get("ORUK_API_KEY", ""),
    )
    worker = PipelineWorker(
        session.pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000),
        enable_rtvi=True,
        processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def connected(*_):
        await session.connected(worker)

    @transport.event_handler("on_client_disconnected")
    async def disconnected(*_):
        await session.disconnected(worker)

    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint,
        handle_sigterm=runner_args.handle_sigterm,
    )
    await runner.add_workers(worker)
    await runner.run()
    print(
        json.dumps(
            {
                "event": "hume_starter_session_closed",
                "generation": session.store.generation,
                "signals": list(session.store.trace),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    # Framework/provider debug logs can include transcript/tool text. Default to
    # no framework logs; applications must design their own redacted telemetry.
    from loguru import logger

    logger.remove()
    from pipecat.runner.run import main

    main()
