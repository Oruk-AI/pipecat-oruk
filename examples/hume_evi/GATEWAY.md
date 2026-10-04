# Local PCM EVI-shaped gateway

`gateway.create_gateway()` constructs a disabled, unbound aiohttp application.
It never starts a listener, selects a model, loads a key, or reads deployment
configuration. An embedding host must explicitly set `enabled=True` and supply
trusted authentication and a factory returning fresh `Providers`. This is the
`local-pcm-evi-v1` profile, not full Hume EVI parity or a deployed migration.

The wire field names were read from the public **Hume npm 0.15.17** generated
chat types/serializers. The finite tests use actual local aiohttp WebSockets and
Pipecat 1.8.1 with named synthetic providers. They do not qualify a public SDK
client, a browser, TLS, a live STT/LLM/TTS provider, or production operation.

## Supported wire contract

The only route is `GET /v0/evi/chat`. The query accepts one `api_key` plus the
optional SDK tracking fields `fernSdkLanguage` and `fernSdkVersion`. The key is
passed only to the injected async authentication callback; it is not a Hume key
exchange. The embedding host must redact query URLs in access/proxy logs and
configure third-party provider logging appropriately. This module records no
keys, prompts, transcripts, audio or tool payloads in its bounded reports.

After authentication, the first text JSON message must be:

```json
{"type":"session_settings","audio":{"encoding":"linear16","sample_rate":16000,"channels":1}}
```

An optional `system_prompt` replaces the configured prompt (1–4,000 characters).
Other settings, settings updates, voice/model/config IDs, URLs, arbitrary tools,
resume identifiers and extra fields refuse. Input is 16 kHz mono signed PCM16
little endian, base64 in `audio_input.data`, 1–1,600 samples per message. There
is no WebM decoder. `user_input.text` also works (1–2,000 characters).

Outputs are `chat_metadata`, finalized `user_message`, `assistant_message`,
`audio_output`, `assistant_end`, `user_interruption`, and fixed-code `error`.
Every audio output is actually encoded as a mono PCM16 **48 kHz WAV**; a provider
returning another rate refuses rather than being relabeled. Audio indexes reset
per turn and the message/audio ID retains that turn's identity. `models` is `{}`:
the gateway does not manufacture a Hume 48-label measurement or map Oruk labels.

Primary `user_message.time` is in **milliseconds**, derived from an explicitly
correlated server-issued PCM span (samples / 16). It describes that input span,
not inferred word timings, an automatically aligned transcript, or provider
emission time. Text-only input uses the current accepted input cursor. This
local profile specifies Unix seconds for `user_interruption.time`.

The one fixed `lookup_demo_order` tool accepts `{"order_id":"..."}` (1–32
characters). Its tool call uses an authority-issued per-turn `tool_call_id`;
only one matching `tool_response` (`content`) or `tool_error` (`error`) is
accepted. Content is limited to 4 KiB. A reply after interruption, a duplicate,
an unknown ID, or an unissued provider call refuses. No customer tool executes
in the gateway; the authenticated client owns that external action. The tool
deadline yields an explicit `tool_timeout` result to the LLM, never a fabricated
successful result. Tool-only intermediate LLM responses do not end the turn.

## Provider adapter and ownership contract

The existing injectable pipeline remains the owner. Its optional
`after_stt` / `before_llm` / `after_llm` processors and `tool_handler` leave the
old pipeline defaults unchanged. The gateway forces expression signals off
and disables the greeting; it does not create another provider framework.

The factory must return four distinct fresh processors (STT, LLM, TTS, VAD)
plus an async `close` callback. It must acquire no unowned work before returning
the bundle. Pipecat cancellation is joined first; `close` then joins any owned
provider resources/callbacks not already handled by framework teardown.
`close` must be cancellation-cooperative and must not return while its work is
still active. Reusing a live or retained processor instance refuses.
An invalid fresh bundle with a usable finalizer is still owned and finalized.
An ambiguous/throwing acquisition or reused bundle quarantines new admission;
the gateway does not close another connection's processors or report that
unknown acquisition as successfully cleaned up. Failure gates stop new work
while continuing to forward framework terminal/control frames for teardown.

Adapters must implement these explicit provenance requirements:

- Every input audio frame has `AUDIO` metadata containing `generation`,
  `epoch`, `first`, and `last` chunk numbers. VAD start identifies the first
  issued chunk; VAD stop and the final STT frame echo the exact contiguous span.
  Missing, duplicate or inconsistent scope refuses. A late old scoped primary
  final is discarded rather than assigned to a newer input. A typed user turn
  retires the preceding PCM epoch.
- The LLM receives `TURN` and `TOOL` metadata on its context frame. It must copy
  `TURN` onto each response start/text/end. Its single optional tool call uses
  the issued `TOOL` ID. A continuation may follow only that retained tool result.
- TTS copies `TURN` onto start/audio/text/stop frames and returns ordered 48 kHz
  mono PCM16 chunks, each at most 100 ms. It preserves the response end until
  all that response's audio has been emitted. Untagged output fails closed;
  explicitly tagged old-turn output is discarded. This is an adapter contract,
  not a claim that arbitrary existing provider services already implement it.

An interruption clears queued old output and cancels the tool waiter through
the existing Pipecat cancellation path. Bytes already handed to the transport
cannot be recalled; their original IDs are never relabeled. A reconnect is a
new chat/group/generation with no replay or resume. SDK callers must explicitly
set `reconnectAttempts: 0`; the gateway does not control client retry policy.

## Real provider adapters

`providers.py` implements the contract above for ordinary Pipecat services,
so a deployment does not have to hand-write provenance handling:

| Gateway role | Adapter | Provider seam |
| --- | --- | --- |
| VAD | `ScopedVAD(analyzer)` | Any Pipecat `VADAnalyzer` (for example Silero). A span starts at the chunk holding the first byte of the earliest frame that could have left QUIET. Because of the analyzer's carried partial frame, that may be one chunk early, but never inside the previous utterance. It ends where the analyzer returns to QUIET, trailing silence included. Speech longer than `max_seconds` fails closed instead of being split. |
| STT | `ScopedUtteranceSTT(transcribe)` | `transcribe(pcm16, sample_rate) -> str` on exactly the span's bytes. `oruk_realtime_transcriber` sends each utterance as one Oruk realtime turn with no connection retry. `openai_transcriber` targets an OpenAI-compatible endpoint. |
| LLM | `class MyLLM(TurnTaggedLLMMixin, OpenAILLMService)` | Any `LLMService` that streams within `process_frame`. One tool call per response is given the issued tool ID before Pipecat records it. Parallel calls fail closed. |
| TTS | `ScopedSynthesisTTS(synthesize)` | `synthesize(text)` yields `(pcm16, rate)`. Output is resampled to 48 kHz, split into chunks of at most 100 ms, and tagged. A response end is held until its audio is out. `openai_speech_synthesizer` streams 24 kHz PCM from an OpenAI-compatible endpoint. |

The provider factory still owns every client and closes it in `close`.
Configure SDK retries explicitly: the OpenAI SDK retries some failures by
default, and Pipecat's OpenAI LLM client does too. Keep the STT `max_seconds`
at or above the VAD `max_seconds`. TTS resampling uses one flushed SoX stream
per sentence, so there are no seams at provider chunk edges.

Three limits apply to the LLM mixin:
- An interruption or a new user turn cancels an unfinished transcription.
- Pipecat drops an unparseable tool call before dispatch, so one malformed call next to one valid call is invisible to the mixin. A lone malformed call fails closed.
- Pipecat's optional `filter_incomplete_user_turns` mode pushes text outside the turn scope and is not supported.

Adapter failures push an `ErrorFrame`, and the gateway reports
`provider_error`:
- The STT adapter pushes its errors downstream. The TTS adapter pushes them upstream to `ProviderBridge`.
- Ordinary Pipecat services report errors upstream with `push_error`. `WireInput` now observes those too, so a failed LLM completion no longer leaves a turn silently open until the session deadline.
- Pipecat's websocket services also report each failed reconnect attempt as an `ErrorFrame`, even when a later attempt succeeds. Any such service ends the session on a brief network blip, as TTS errors already did through `ProviderBridge`.

`tests/test_hume_evi_providers.py` checks all of this over loopback. It drives
the pinned Pipecat `OpenAILLMService` and the OpenAI SDK against a local
OpenAI-compatible fixture, and it covers:
- exact transcription spans and bounded 48 kHz output;
- tool-ID reissue;
- parallel-call refusal;
- STT, LLM and TTS failure;
- a typed turn.

This is contract evidence only. It does not qualify any real provider's
accuracy, latency, cost, retention, voice rights or interruption feel. Those
still need the customer's selected providers and permitted audio.

## Bounds and failure behavior

Default local bounds are two concurrent sessions, 2,000 input messages, eight
turns, 960,000 input PCM bytes, 4,000,000 serialized output bytes, and 32 pending
output messages. Message JSON is at most 16 KiB, 64 nodes and depth four;
duplicate keys/NaN refuse. Output frames are at most 32 KiB. Input queue
admission allows at most 16 frames awaiting the input processor. Total input
and message caps also bound downstream admission; this is not a hard OS memory
sandbox. Defaults are local fixture policy, not production sizing.

Authentication/settings, sends, tools and the active session have finite
cooperative asyncio deadlines. The default active session is 60 seconds. A
deadline does not prove a provider stopped: shutdown remains pending until
framework and provider cleanup actually finish. The embedding process needs
its own finite supervisor; no wall-clock guarantee is claimed for a
non-cooperative callback or blocked kernel operation. Repeated handler
cancellation cannot detach its dedicated cleanup task. Application shutdown
joins registered handlers. If cleanup fails rather than proving closure, the
application is quarantined for new sessions with `provider_cleanup_unresolved`.
Provider billing and remote cancellation acknowledgement remain unreconciled.

No default listener or deployment configuration is included. No React/WebM,
resumption, arbitrary customer tools, provider replacement, native measurement
parity, or customer migration completion is implied.

## Local qualification

From the repository root, the owner of execution can run the new suite with
the already pinned Pipecat 1.8.1 / aiohttp 3.14.3 Python 3.12 environment:

```text
.venv/bin/python -m pytest -q tests/test_hume_evi_gateway.py
```

Run under a finite owned process-group supervisor with a minimal credential-free
environment. The tests themselves permit only the exact active loopback port,
retain a sticky external-network refusal counter, and close their listener,
clients and provider tasks in `finally`. They never call a model or provider.
The guard is installed after asyncio creates its platform event loop and is
removed before loop teardown; Windows' internal socketpair is outside that
scope. No arbitrary loopback ports are allowed during a test. The shared
synthetic-provider fixture disables Pipecat 1.8.1's optional NLTK cache warm-up
to avoid a first-run tokenizer download. These tests do not qualify NLTK
tokenization or a production image's tokenizer provisioning.
The pre-existing pipeline/tool interruption suites should also be run because
the new optional seam shares `pipeline.py`. Source authorship alone is not a
passing test result; qualification results belong to the saved run receipts.
