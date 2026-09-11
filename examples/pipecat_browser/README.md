# Browser speech demo

This example connects a browser to Pipecat SmallWebRTC and the production Oruk
realtime API. It shows interim and final text, phrase estimates, completed-turn
receipts, cancellation after an interim result, and reconnection. Oruk handles
speech recognition; this example does not use an LLM, TTS, or microphone.

The server binds only to `127.0.0.1:3119`. Keep the API key in its environment.
This is a local development example, not a hosted multi-user service.

## Run

Install the package with its `agent` extra, then fetch the two public assets:

```sh
curl --fail --location https://oruk.ai/samples/oruk-quickstart.wav -o examples/pipecat_browser/sample.wav
curl --fail --location https://oruk.ai/brand/kit/oruk-primary.png -o examples/pipecat_browser/oruk-primary.png
python examples/pipecat_browser/server.py
```

Set `ORUK_API_KEY` before the last command. Open <http://127.0.0.1:3119/>, connect,
and stream the sample. The audio is sent at playback speed through WebRTC.
A continuous silent track keeps the connection ready between utterances.

**Record 45-second demo** starts a repeatable browser recording: two complete
utterances, a disconnect after the next interim result, and a new session with
one more complete utterance. The canvas is a live view of actual received events.
Only the supplied sample is recorded; system audio and the microphone are not
captured. The unedited WebM is saved beside this example. Failures remain visible
in the recording rather than being replaced with expected output.

A cancelled request can still be metered. The demonstrated cancelled turn used
1.58 seconds of API audio; see the release verification record. Phrase scores
estimate vocal expression and are not calibrated probabilities of feelings.

The sample and logo come from Oruk's public website and are separate from the
MIT-licensed program. This example makes no independent accuracy claim.
