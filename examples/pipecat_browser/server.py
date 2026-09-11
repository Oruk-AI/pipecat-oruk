"""Loopback-only browser demo for the Oruk Pipecat adapter.

Uses real WebRTC and production Oruk STT. No microphone, LLM or TTS is used.
The public Oruk quickstart sample is the only input offered by this QA page.
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from aiohttp import web
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from pipecat_oruk import OrukEventFrame, OrukSTTService, OrukTurnCompletedFrame

ROOT = Path(__file__).resolve().parent
SESSIONS = {}
ORIGIN = "http://127.0.0.1:3119"


def record(row):
    with (ROOT / "server-events.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")


class Results(FrameProcessor):
    def __init__(self, connection, session_id):
        super().__init__()
        self.connection, self.session_id = connection, session_id
        self.started = time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        row = None
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
                row = {
                    "type": "final" if isinstance(frame, TranscriptionFrame) else "interim",
                    "text": frame.text,
                    "metadata": frame.metadata.get("oruk"),
                }
            elif isinstance(frame, OrukEventFrame):
                row = {"type": "phrase", "turn_id": frame.turn_id, "event": frame.event}
            elif isinstance(frame, OrukTurnCompletedFrame):
                row = {"type": "complete", "turn_id": frame.turn_id, "result": asdict(frame.result)}
        if row:
            row.update(
                session_id=self.session_id, elapsed=round(time.monotonic() - self.started, 3)
            )
            record(row)
            self.connection.send_app_message(row)
        await self.push_frame(frame, direction)


async def run(connection, session_id):
    transport = SmallWebRTCTransport(
        connection,
        TransportParams(audio_in_enabled=True, audio_in_channels=1, audio_out_enabled=False),
    )
    speech = OrukSTTService(wait_for_emotions=True)
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.4))
    )
    pipeline = Pipeline(
        [transport.input(), vad, speech, Results(connection, session_id), transport.output()]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000),
        enable_rtvi=False,
        processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
        idle_timeout_secs=60,
    )
    SESSIONS[session_id]["worker"] = worker

    @transport.event_handler("on_client_connected")
    async def connected(*_):
        row = {"type": "connected", "session_id": session_id}
        record(row)
        connection.send_app_message(row)

    @transport.event_handler("on_client_disconnected")
    async def disconnected(*_):
        record({"type": "disconnected", "session_id": session_id})
        await worker.cancel()

    @speech.event_handler("on_error")
    async def error(_speech, frame):
        row = {"type": "error", "session_id": session_id, "error": frame.error}
        record(row)
        connection.send_app_message(row)

    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    try:
        await asyncio.wait_for(runner.run(), 90)
    finally:
        await worker.cancel()
        await connection.disconnect()
        SESSIONS.pop(session_id, None)
        record({"type": "closed", "session_id": session_id})


async def offer(request):
    if request.headers.get("Origin") != ORIGIN:
        raise web.HTTPForbidden(text="Use the local demo page")
    if len(SESSIONS) >= 2:
        raise web.HTTPTooManyRequests(text="Close the existing session first")
    if request.content_length and request.content_length > 300_000:
        raise web.HTTPRequestEntityTooLarge(max_size=300_000, actual_size=request.content_length)
    data = await request.json()
    if data.get("type") != "offer" or not isinstance(data.get("sdp"), str):
        raise web.HTTPBadRequest()
    session_id = str(uuid.uuid4())
    connection = SmallWebRTCConnection(ice_servers=[], connection_timeout_secs=20)
    try:
        await connection.initialize(data["sdp"], data["type"])
    except BaseException:
        await connection.disconnect()
        raise
    SESSIONS[session_id] = {"connection": connection}
    task = asyncio.create_task(run(connection, session_id))
    SESSIONS[session_id]["task"] = task
    task.add_done_callback(
        lambda t: (
            record({"type": "task_error", "error": str(t.exception())})
            if not t.cancelled() and t.exception()
            else None
        )
    )
    return web.json_response({**connection.get_answer(), "session_id": session_id})


async def close(request):
    if request.headers.get("Origin") != ORIGIN:
        raise web.HTTPForbidden()
    data = await request.json()
    entry = SESSIONS.get(data.get("session_id"))
    if entry:
        if entry.get("worker"):
            await entry["worker"].cancel()
        await entry["connection"].disconnect()
    return web.json_response({"closed": True})


async def page(_):
    return web.FileResponse(ROOT / "index.html")


async def sample(_):
    return web.FileResponse(ROOT / "sample.wav")


async def logo(_):
    return web.FileResponse(ROOT / "oruk-primary.png")


async def recording(request):
    if request.headers.get("Origin") != ORIGIN or request.content_type != "video/webm":
        raise web.HTTPForbidden()
    body = await request.read()
    if not body.startswith(b"\x1a\x45\xdf\xa3"):
        raise web.HTTPBadRequest(text="Expected WebM recording")
    name = "webrtc-demo-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".webm"
    (ROOT / name).write_bytes(body)
    record({"type": "recording_saved", "file": name, "bytes": len(body)})
    return web.json_response({"file": name})


async def shutdown(_):
    for entry in list(SESSIONS.values()):
        if entry.get("worker"):
            await entry["worker"].cancel()
        await entry["connection"].disconnect()


app = web.Application(client_max_size=64_000_000)
app.add_routes(
    [
        web.get("/", page),
        web.get("/demo.js", lambda _: web.FileResponse(ROOT / "demo.js")),
        web.get("/sample.wav", sample),
        web.get("/logo.png", logo),
        web.post("/offer", offer),
        web.post("/close", close),
        web.post("/recording", recording),
    ]
)
app.on_shutdown.append(shutdown)
if __name__ == "__main__":
    if not (ROOT / "sample.wav").is_file() or not (ROOT / "oruk-primary.png").is_file():
        raise SystemExit("Download the sample and branding assets using the README commands first")
    if not os.environ.get("ORUK_API_KEY"):
        raise SystemExit("Set ORUK_API_KEY in the server environment")
    web.run_app(app, host="127.0.0.1", port=3119, print=None)
