<img src="https://oruk.ai/brand/kit/oruk-primary.png" alt="oruk" width="160">

# Oruk speech and phrase emotion for Pipecat

Stream speech to Oruk's realtime preview and receive interim transcripts, final text, and independent phrase-emotion estimates. This integration is developed by **oruk labs**, the company providing the API. It is community-maintained by Oruk; it is not maintained or endorsed by Pipecat.

**Release candidate.** The adapter is tested with Pipecat 1.8.1 on Python 3.11, 3.12, 3.13, and 3.14. The package accepts Pipecat 1.8.x; other patch releases require their own recorded checks. Automated tests use a simulated gateway with the real framework. A separate [production browser demonstration](https://github.com/Oruk-AI/pipecat-oruk/blob/main/docs/DEMO.md) verifies WebRTC, real transcription and phrase estimates, two consecutive turns, cancellation after an interim result, reconnection, and backend metering. This is not an accuracy benchmark or a verified LLM/TTS conversation.

See the [Oruk service guide in Pipecat’s documentation](https://docs.pipecat.ai/api-reference/server/services/stt/oruk) for setup, configuration, and a pipeline example.

## Install

Use a Python virtual environment and pin the release candidate explicitly:

```sh
python -m pip install 'pipecat-oruk==0.1.0rc1'
```

For the browser transport dependencies, install `pipecat-oruk[agent]==0.1.0rc1`. The examples below live in this repository; clone it to run them. To develop the adapter from a checkout, run `python -m pip install '.'` from the repository root.

Set `ORUK_API_KEY` in the server environment using a key from your Oruk developer portal. Keys are sent in an authenticated WebSocket upgrade, never in a URL or browser bundle. The distribution is named `pipecat-oruk` and uses `pipecat_oruk` imports.

## Local Orukeet transcription

The optional `OrukeetSTTService` runs the [Orukeet](https://huggingface.co/oruk/orukeet)
INT8 ONNX model on CPU. It needs no API key and keeps audio on the machine running
Pipecat. Install the local extra from this checkout (it is not in `0.1.0rc1`):

```sh
python -m pip install '.[local]'
python examples/pipecat_local_file.py recording.wav
```

The file example accepts mono PCM16 WAVs at 16 kHz, up to one minute long, and
prints final text from a real Silero VAD pipeline. For an application, set its
input sample rate to 16 kHz and place the local service after `VADProcessor`:

```python
from pipecat_oruk.local import OrukeetSTTService

stt = OrukeetSTTService()
await stt.prewarm()  # In your async setup, before accepting audio.
# Pipeline([transport.input(), vad, stt, ...])
```

Recognition starts when VAD ends an utterance. The model supports 25 languages
with automatic detection, but emits only final text: no interim results,
language code, word timestamps, speaker labels, or phrase-emotion estimates.
Language forcing is rejected. One instance serializes native inference;
canceling a task discards its result, while cleanup waits for CPU work to finish.
The pipeline's normal cleanup releases the model.

The first load fetches required files and notices from Hugging Face at revision
`1751fce6ecde442f14543cf1804800c49b3e415c`, checking all four runtime files against
pinned SHA-256 hashes. Its required `config.json` uses Hugging Face's normal model
download accounting. Complete cached loads make no HTTP requests. Pass
`cache_dir="..."` to choose a Hub cache and `local_files_only=True` to require
cached files; the file example offers `--offline`. A corrupt cache fails with a
message identifying the file to remove and download again.

The model weights use **CC BY-SA 4.0** with NVIDIA foundation attribution. The
weight license, notices, and converter/preprocessor licenses are downloaded
alongside the model; this package's code remains MIT-licensed. The hosted
`OrukSTTService` and its dependencies are unchanged unless the local extra is
explicitly installed.

## Try an audio recording

```sh
python examples/pipecat_stream_file.py recording.wav
```

Use a mono PCM16 WAV at 8–96 kHz, up to five minutes long. Audio is sent at playback speed. The example prints JSON for interim text, final text, phrase events, and successful turn completion. It returns a nonzero exit status on failure.

Options:

- `--vad`: use Silero to split utterances; the file must be 8 or 16 kHz for this detector.
- `--wait-for-emotions`: include late phrase results in the final transcript's metadata.
- `--diarize`: request speaker labels within each Oruk turn.
- `--language fr`: configure a supported Oruk locale; the default is automatic recognition.
- `--fast`: send a saved recording without playback pacing, limited to 110 seconds.

## Try the browser speech demo

The [local browser example](https://github.com/Oruk-AI/pipecat-oruk/blob/main/examples/pipecat_browser/README.md) needs only an Oruk key. It streams the public sample through WebRTC, shows the returned estimates and receipts, and can record its own demonstration. No LLM or TTS account is required for this speech-only example. Install the agent extra shown below to include the browser transport dependencies.

## Put Oruk into a pipeline

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat_oruk import OrukSTTService

vad = VADProcessor(
    vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.2))
)
stt = OrukSTTService(wait_for_emotions=True)

pipeline = Pipeline([
    transport.input(),
    vad,
    stt,
    user_aggregator,
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
```

The transport, aggregators, LLM, and TTS belong to your application. The complete [browser agent example](https://github.com/Oruk-AI/pipecat-oruk/blob/main/examples/pipecat_agent.py) assembles them. Set the pipeline's input rate to 16 kHz and avoid adding a second VAD inside the user aggregator. Oruk commits on the upstream VAD stop. Pipecat forwards audio before emitting a VAD start, so the adapter retains a bounded half-second prefix by default. Its prefix must cover the detector's start-confirmation interval plus one input chunk; an insufficient prefix is reported as an error instead of silently clipping the opening.

For externally segmented audio, use `OrukSTTService(vad=False)` and queue `OrukCommitFrame()` after the utterance. This is a system frame, ordered with incoming audio. `EndFrame` also flushes the current turn and waits for its result. An assistant `InterruptionFrame` preserves input recognition; `CancelFrame` aborts it.

## Run the browser agent

```sh
python -m pip install '.[agent]'
python examples/pipecat_agent.py --transport webrtc --host 127.0.0.1 --port 7860
```

Before connecting, configure `ORUK_API_KEY`, `OPENAI_API_KEY`, `LLM_MODEL`, `TTS_MODEL`, and `TTS_VOICE` in the server environment. Choose model and voice names available to your account. The example uses Pipecat's OpenAI services for response generation and speech output; it does not call them during installation or `--help`. Open Pipecat's local runner interface, connect, and speak. This is a development example, not an Internet-facing deployment configuration.

The agent waits for phrase events and matches their metadata to the text being aggregated into the current user message. Only known labels, scores, and segment times reach its instruction context. The original transcript stays unchanged. The added note explicitly treats estimates as tentative, and it is removed for a subsequent text-only message. The example's local framework test verifies this handoff; it does not establish better conversations or emotion-recognition accuracy.

## Read results

| Output | Meaning |
| --- | --- |
| `InterimTranscriptionFrame` | Provisional text accumulated for this turn. It can be revised. |
| `TranscriptionFrame` | The provider's final transcript, with `finalized=True`. `metadata["oruk"]` identifies its stream, turn, audio offset, model, and available phrases. |
| `OrukPhraseEmotionFrame` | A completed or failed phrase estimate. Read the original `event`, including `phrase_id`, start/end, scores, and provider request ID. |
| `OrukSpeakerBoundaryFrame` | An optional speaker boundary. Speaker identity is not assigned to the entire transcript. |
| `OrukTurnCompletedFrame` | Final text and usage were received and the WebSocket closed successfully. The result includes all phrases sorted by time. |

Phrase estimates are independent of transcript deltas and may arrive after final text. Join by stream/turn/request identity and phrase ID; never pair the nearest delta with an emotion result. Phrase times are relative to the turn; `audio_offset` places that turn on the incoming audio timeline, including skipped idle time. Speaker IDs restart with each connection. This integration does not claim word alignment or transcript-confidence scores.

With the default `wait_for_emotions=False`, final text is delivered immediately and its metadata is an immutable snapshot. Listen for later phrase frames or the `on_phrase_emotion` event. With `True`, final text waits for a clean turn completion and carries all returned phrase events. Interim text remains immediate in either mode.

```python
@stt.event_handler("on_phrase_emotion")
async def on_phrase(service, frame):
    # Store or display this turn's estimate according to your application's needs.
    handle_phrase(frame.turn_id, frame.event)
```

Handlers are awaited in order and must finish within `event_handler_timeout` (two seconds by default). `on_speaker_boundary` and `on_turn_completed` use the same pattern. Each handler receives its own copy. Queue substantial application work separately within your own bounded worker.

## Configure and troubleshoot

Use `RealtimeOptions` for initial phrase segmentation, language, diarization, and turn/completion limits. `OrukSTTService.Settings(language="fr")` is also supported. A `STTUpdateSettingsFrame` can change the language between turns; changing the fixed model or changing language mid-utterance is rejected without corrupting the active turn.

Input must be mono PCM16 at the pipeline's fixed rate. Sixteen-kHz samples are transmitted byte for byte. Other admitted rates use streaming SoX VHQ conversion to the endpoint's 16 kHz, with the filter tail flushed at each commit. This conversion does not preserve frequencies above the endpoint's bandwidth. The original file is unchanged. Each input frame is limited to one second.

The default outstanding-audio limit is 30 seconds, with at most 32 pending turns and 128 pending provider events. Overflow is an explicit error, not silent sample loss. Feed microphone audio at playback speed. Each turn is limited to 300 seconds by default; a graceful pipeline stop has a separate 30-second limit. All limits are configurable within documented constructor bounds.

Transient connection failures can be retried only before the first audio write attempt. Once delivery may have happened, the utterance is never replayed. A failed turn marks this service unusable and follows the worker's `ProcessorUnusablePolicy`; use `CANCEL` for the supplied example. Start a fresh service after correcting the cause. A phrase-emotion failure alone does not discard a valid transcript.

Usage metrics come from the provider receipt, not idle microphone bytes. A receipt can precede a terminal error, so usage alone is not proof of successful completion. The optional latency metric measures speech end to delivered final text for the matching turn. No production P99 is advertised: Pipecat uses its framework fallback until you supply a measured `ttfs_p99_latency`.

## Local verification and release status

The tests cover real streaming before commit, independent turns, VAD prefix and partial frames, five sample-rate conversions, failure/timeout/cancellation, bounded buffers, mute, immutable metadata, receipt-based metrics, language updates, and the actual user-aggregator handoff. The CLI is exercised as a subprocess against the local gateway. The test recording and its separate attribution are in `tests/audio/`. It is not covered by the program's MIT license.

```sh
python -m pip install '.[agent,test,local]' 'build>=1,<2' 'ruff>=0.15,<1' 'pyright>=1.1,<2'
python -m pytest -q
python -m build
```

The production browser check covers two consecutive turns, cancellation after an interim result, reconnection, and backend metering. Its 45-second recording and request receipts are in [the demonstration record](https://github.com/Oruk-AI/pipecat-oruk/blob/main/docs/DEMO.md). A full spoken assistant using an LLM and TTS, including assistant barge-in, still needs separate verification. This package is a release candidate; it is not an upstream Pipecat release.
