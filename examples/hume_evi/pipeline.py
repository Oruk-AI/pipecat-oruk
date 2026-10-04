"""Injectable voice pipeline using Pipecat 1.8.1 public APIs."""

from dataclasses import dataclass

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .config import Config
from .expression import ExpressionAudioTap
from .policy import (
    PrimaryScopeBridge,
    ScopeResolver,
    SignalContextGate,
    SignalNoteExpiry,
    SignalStore,
)


async def lookup_demo_order(params: FunctionCallParams):
    """Synthetic read-only data. Customer tools require their own authorization."""
    args = params.arguments
    if (
        not isinstance(args, dict)
        or set(args) != {"order_id"}
        or not isinstance(args["order_id"], str)
        or len(args["order_id"]) > 32
    ):
        await params.result_callback({"error": "invalid_arguments"})
        return
    result = (
        {"order_id": "DEMO-100", "status": "packed", "synthetic": True}
        if args["order_id"] == "DEMO-100"
        else {"error": "not_found", "synthetic": True}
    )
    await params.result_callback(result)


@dataclass
class Conversation:
    pipeline: Pipeline
    context: LLMContext
    store: SignalStore
    tap: ExpressionAudioTap
    stt: FrameProcessor
    config: Config
    greeted: bool = False

    async def connected(self, worker):
        if not self.greeted and not self.store.closed:
            self.greeted = True
            if self.config.greeting:
                await worker.queue_frame(
                    TTSSpeakFrame(self.config.greeting, append_to_context=True)
                )

    async def disconnected(self, worker):
        await worker.cancel()


def build_pipeline(
    transport,
    stt: FrameProcessor,
    llm: LLMService,
    tts: FrameProcessor,
    *,
    config: Config = Config(),
    vad: FrameProcessor | None = None,
    scope_resolver: ScopeResolver | None = None,
    api_key: str = "",
    endpoint: str | None = None,
    tool_handler=None,
    after_stt: FrameProcessor | None = None,
    before_llm: FrameProcessor | None = None,
    after_llm: FrameProcessor | None = None,
) -> Conversation:
    """Each call owns a new session. Never reuse these services after disconnect.

    The customer supplies primary STT. There is no automatic transcript alignment.
    A resolver must identify exact audio scopes, not emission time or text similarity.
    """
    if vad is None:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.processors.audio.vad_processor import VADProcessor

        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(start_secs=0.2, stop_secs=0.2)
            )
        )
    store = SignalStore()
    tap = ExpressionAudioTap(
        store, config, api_key=api_key, **({"endpoint": endpoint} if endpoint else {})
    )
    tool = FunctionSchema(
        name="lookup_demo_order",
        description="Look up a synthetic demo order; no customer records.",
        properties={
            "order_id": {
                "type": "string",
                "description": "Synthetic identifier, for example DEMO-100",
            }
        },
        required=["order_id"],
    )
    llm.register_function(
        "lookup_demo_order",
        lookup_demo_order if tool_handler is None else tool_handler,
        cancel_on_interruption=True,
        timeout_secs=3.0,
    )
    context = LLMContext(
        [{"role": "system", "content": config.system_prompt}],
        tools=ToolsSchema(standard_tools=[tool]),
    )
    user = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            vad_analyzer=None,
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.2)]
            ),
        ),
    )
    gate = SignalContextGate(store, config.signal_policy)
    pipeline = Pipeline(
        [processor for processor in [
            transport.input(),
            vad,
            tap,
            stt,
            PrimaryScopeBridge(store, scope_resolver),
            after_stt,
            user,
            gate,
            before_llm,
            llm,
            after_llm,
            SignalNoteExpiry(gate),
            tts,
            transport.output(),
            LLMAssistantAggregator(context),
        ] if processor is not None]
    )
    return Conversation(pipeline, context, store, tap, stt, config)
