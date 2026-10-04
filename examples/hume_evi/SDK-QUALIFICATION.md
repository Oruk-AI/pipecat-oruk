# Opt-in public SDK fixture

This source fixture exercises the public npm `hume@0.15.17` `HumeClient` against
the actual local gateway and named synthetic VAD/STT/LLM/TTS. It is separate from
the normal test suite and does not silently skip when dependencies are missing.
It does not install dependencies or accept provider credentials.

An operator must supply an already acquired, independently pinned Node 22 and the
Hume package directory from the fixture's exact lockfile. The fixture checks the
package, ChatClient, ChatSocket, WebSocket implementation and lockfile hashes
before importing the SDK. These selected checks supplement the operator's full
dependency acquisition receipt; they are not a complete dependency tree verifier.

```sh
HUME_GATEWAY_NODE=/absolute/path/to/node \
HUME_GATEWAY_SDK_ROOT=/absolute/path/to/npm-package/node_modules/hume \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -B -m pytest -q -s tests/hume_gateway_sdk_opt_in.py
```

Run under a finite outer process-group supervisor (60 seconds is sufficient for
the two finite fixture cases unless a concrete failure requires investigation).
The Python fixture starts one non-detached Node child per case, retains bounded
output, and terminates/kills and joins that direct child on failure. The outer
supervisor is still responsible for process-group containment on abrupt parent
death. No arbitrary daemon ownership or hard OS scheduling deadline is claimed.

The SDK selects `environment.evi = ws://127.0.0.1:<owned-port>/v0/evi`, with
`reconnectAttempts: 0`. The unmodified public methods send explicit `linear16`,
16 kHz mono session settings, two typed turns and one PCM turn; a separate case
uses the exact tool-call ID through `sendToolResponseMessage`. Received camelCase
objects must preserve primary text, empty model scores and millisecond intervals;
each audio result is independently decoded as a 48 kHz, mono PCM16 WAV with the
expected actual samples. This is an SDK serialization and local transport check,
not a real model, voice quality, browser, unchanged React/WebM or public endpoint
qualification.

The fixture wraps and delegates the real native Node WebSocket constructor and
socket connect calls, allowing only the single owned IPv4 loopback endpoint.
It refuses other fetch/TLS/DNS paths and observes attempted violations as sticky
failures. Plain local WebSockets establish no TLS/SNI/certificate conclusion.
The SDK's `ChatSocket.close()` invokes its public close handler immediately; the
fixture separately waits for the underlying native WebSocket and socket closure,
then the Python side checks provider finalization and gateway-owner release.

Only synthetic credentials appear in memory. Access logging is disabled, and
failure output contains fixed phase/code fields rather than raw exceptions or
query URLs. Assertions and schemas are not weakened if the real SDK rejects a
message: preserve the failed run and diagnose the source boundary.
