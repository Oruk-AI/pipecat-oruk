# Vapi custom transcriber with Oruk

This source example connects Vapi's caller audio to Oruk realtime transcription.
Vapi continues to run your selected LLM and voice. It does not restore a Hume
voice, expose Hume emotion dimensions, or add emotion fields to Vapi's transcript
protocol. Oruk is not affiliated with Hume or Vapi.

The wire implementation is locally tested against synthetic loopback providers.
An actual Vapi call, served Oruk revision, latency, interruption behavior,
account billing and deployment have not been qualified. Use a separate test
assistant and consented test audio before considering a customer cutover.

## Protocol and source basis

Read on October 6, 2026:

- [Vapi custom transcriber](https://docs.vapi.ai/customization/custom-transcriber)
  defines the start/audio/response wire shapes and credential configuration.
- [Vapi's Gradium recipe](https://docs.vapi.ai/customization/custom-transcriber/gradium)
  emphasizes caller-channel extraction and partial transcripts for interruption.
- [Vapi voice pipeline](https://docs.vapi.ai/customization/voice-pipeline-configuration)
  describes speaking and endpointing controls; tune them with measured calls.

This example accepts only the documented stereo, raw linear16, 16 kHz profile.
It extracts channel 0 and preserves samples split across WebSocket messages.
Channel 1 is never sent to Oruk. A different profile closes the connection;
there is no guessed resampling, channel assignment or silent trimming.

Native transcript deltas become cumulative Vapi `partial` messages. A `final`
is emitted only after the native turn returns nonempty final text, usage and
normal close. An empty, incomplete or failed turn closes the call's transcription connection.
The Vapi server's acceptance of this failure behavior is part of the pilot.

## Configure a test deployment

1. Check out the exact reviewed commit of this example and retain that SHA in
   your deployment manifest. Examples are source files, not files included in
   the `pipecat-oruk` wheel. From the repository, install the existing agent
   dependencies with `python -m pip install '.[agent,test]'`. The tested
   framework version is Pipecat 1.8.1. Lock the resolved environment before
   deployment; the optional local ASR model is not needed.
2. Generate a dedicated inbound bearer credential, separate from the Oruk key.
   Store it as a Vapi Custom Credential. The credential authenticates Vapi to
   your bridge; the Oruk key remains in the bridge's secret store.
3. Construct the app in your deployment entry point. The following example
   explicitly enables it on loopback; importing the module opens no listener.

```python
import os
from aiohttp import web
from examples.vapi.transcriber import create_app

app = create_app(
    inbound_token=os.environ["VAPI_TRANSCRIBER_BEARER"],
    oruk_api_key=os.environ["ORUK_API_KEY"],
    enabled=True,
    max_sessions=2,
)
web.run_app(app, host="127.0.0.1", port=8088, access_log=None)
```

4. Put an owned TLS endpoint in front of that listener and preserve the bearer
   upgrade header. Expose only `/api/custom-transcriber`. Configure redacted
   access/error logging and finite WebSocket limits at the proxy. Do not log
   credentials, PCM or transcript payloads. Use separately authorized serving
   resources; protected customer machines are outside this deployment recipe.
5. Apply this fragment to a separate Vapi test assistant, replacing the endpoint
   and credential ID. It changes only that assistant's transcriber configuration.

```json
{
  "transcriber": {
    "provider": "custom-transcriber",
    "server": {
      "url": "wss://YOUR_OWNED_HOST/api/custom-transcriber",
      "credentialId": "YOUR_CUSTOM_BEARER_CREDENTIAL_ID"
    }
  }
}
```

No call, assistant update, credential creation or deployment is performed by
the example or tests. Preserve the previous transcriber configuration for
rollback. Choose a supported voice independently if replacing Octave.

## Limits and behavior

- Per connection: ten-minute wall/audio limit; 15-second receive idle limit;
  five seconds to supply the start message; 64 KB maximum input message.
- Caller VAD uses Silero, with 0.2-second start and 0.6-second stop settings.
  Twenty analysis frames of pre-roll are retained. These are initial settings,
  not measured customer speech thresholds.
- Each utterance has one native connection, at most 30 seconds of audio and ten
  seconds to finish after commit. Retries are disabled. A disconnect cancels
  and joins in-flight work; it does not commit or replay buffered speech.
- Audio and output queues hold at most 64 items each; a stalled consumer closes
  the session instead of accumulating unbounded audio or text. Native text is
  capped at 16,000 characters, and each outbound write has a two-second timeout.
- Utterance completion is serialized. Audio transport can apply backpressure
  while waiting for completion; qualify fast successive turns and overlap.
- The app opens at most two sessions by default and refuses additional calls.
  It is disabled unless explicitly enabled. Each session owns a fresh analyzer,
  HTTP client and task group; shutdown stops admission and joins active work.
- Memory is transient in this bridge. This does not establish Vapi, Oruk,
  reverse-proxy or deployment-level retention; verify those independently.

## Qualification and cutover

Run `python -m pytest -q tests/test_vapi_transcriber.py` from the source checkout.
The tests use synthetic audio, a deterministic VAD and fake native responses.
They verify channel separation, split samples, cumulative partials, final
completion, call isolation, no replay, bounded backpressure and joined cleanup.
They do not measure Silero accuracy or call quality.

Before switching a customer, record the exact source/dependency/model revisions
and actual audio profile. Exercise a consented web or telephone test call with
caller-only speech, assistant-only speech, interruption, short replies, silence,
rapid consecutive turns and a lost provider connection. Measure first partial,
final and voice-response latency, check that the assistant never transcribes
itself, and reconcile each native request against usage and the invoice. Verify
that authentication failure and capacity exhaustion cause the intended Vapi
behavior. Rehearse restoring the old transcriber on the test assistant. Record
technical-owner acceptance before changing production traffic.
