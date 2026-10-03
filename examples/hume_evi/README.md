# Hume EVI migration starter

A browser voice-agent example for **Pipecat 1.8.1** and **pipecat-oruk 0.1.0rc3**. The builder accepts your STT, LLM and TTS services. The runnable example selects Deepgram STT and OpenAI LLM/TTS explicitly; Oruk is an optional expression-analysis sidecar. It never substitutes its transcript for your primary STT output.

This is source code for building a replacement conversation stack, not an EVI protocol adapter. It has deterministic local lifecycle tests. Real provider responses, account entitlements, browser microphone/speaker behavior, latency and billing still require a separate end-to-end test with your authorized audio.

## Offline setup and tests

Run these commands from this repository's root. The checked-in hash lock was installed and tested with Python **3.12.13 on macOS 26.4.1 ARM64**. Its resolution targets macOS 14 or newer; other OS versions still need validation. It includes framework/provider dependencies and test tools. It does not install the optional Orukeet local-model extra or download its weights.

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes examples/hume_evi/requirements-macos-arm64-py312.lock
.venv/bin/python -m examples.hume_evi.bot --help
.venv/bin/python -m pytest -q tests/test_hume_evi_config.py tests/test_hume_evi_expression.py tests/test_hume_evi_pipeline.py
```

Package installation accesses the package registry. The help command and the new deterministic tests make no inference calls. Tests reject non-loopback socket connections and DNS, use an in-process fake Oruk WebSocket gateway, named synthetic providers, real Pipecat workers/aggregators/tool dispatch, and a real framework playback queue. The help subprocess blocks all socket connections.

For another OS/Python lane, resolve `requirements.in` into a separate hash lock and run the tests on that target. The supplied macOS lock is not a Linux deployment claim. To reproduce its resolution:

```sh
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile examples/hume_evi/requirements.in --generate-hashes --python-version 3.12 --python-platform aarch64-apple-darwin --output-file examples/hume_evi/requirements-macos-arm64-py312.lock
```

## Configure the live example

```sh
cp examples/hume_evi/.env.example .env
cp examples/hume_evi/config.example.json hume-starter.local.json
```

Edit `.env` locally with server-side credentials. It is ignored by Git. Choose available models and a TTS voice from your accounts; no model or voice entitlement is assumed:

| Setting | Required value |
| --- | --- |
| `STT_PROVIDER` | `deepgram` for this CLI |
| `STT_MODEL`, `DEEPGRAM_API_KEY` | Your chosen primary STT model and account key |
| `LLM_MODEL`, `OPENAI_API_KEY` | Your chosen LLM model and account key |
| `TTS_MODEL`, `TTS_VOICE` | Your chosen OpenAI TTS model and voice |
| `ORUK_API_KEY` | Only required if `signals_enabled` is explicitly set to `true` |

Do not put credentials in the browser, JSON configuration, command arguments or version control. Your local prompt/configuration may also be private; ignore it before adding customer-specific content. The example configuration contains only synthetic content.

The JSON accepts only `schema_version: 1`, `system_prompt`, `greeting`, `signals_enabled`, `signal_policy`, `signal_timeout`, `signal_buffer_seconds`, and `max_turn_seconds`. Unknown fields, duplicate keys and invalid limits fail rather than silently discarding settings. Manually map your Hume system prompt and first message to the first two fields. Hume voice IDs, credentials, tool URLs, session state, emotion labels and exported configurations are not imported. Review unsupported settings individually.

**The next command starts a live development app. Joining it connects to your selected providers and can incur charges.** Microphone audio goes to your primary STT; conversation text goes to the LLM; reply text goes to TTS. Enabling the Oruk sidecar also sends speech audio to Oruk and bills that service separately.

```sh
HUME_STARTER_CONFIG=hume-starter.local.json .venv/bin/python -m examples.hume_evi.bot -t webrtc --host localhost
```

Use the local URL printed by Pipecat. Keep this development server on localhost. Authentication, deployment, account quotas and production observability are separate work. Each browser session constructs fresh services, context and generation IDs. Disconnect cancels the worker; reconnect starts a new conversation and greeting, without replaying previous audio or history.

## Keep your primary STT

`pipeline.build_pipeline(transport, stt, llm, tts, ...)` preserves the objects you provide. Replace the CLI factories for another Pipecat-compatible provider, or call this builder from your own application. The starter uses public Pipecat processor/context APIs, without overriding a private STT or aggregator method.

```text
input → VAD → optional Oruk audio tap → your STT → scope bridge
      → user context → signal gate → your LLM → signal expiry
      → your TTS → output → assistant history

audio tap → one bounded optional task → Oruk /v1/realtime
```

The concrete runner uses mono PCM16 at 16 kHz input and 24 kHz output. The bundled VAD configuration confirms speech after 0.2 seconds. The sidecar retains a 0.5-second prefix and checks that the VAD confirmation plus last input chunk fits that prefix. Unsupported track identity skips that auxiliary turn. An invalid audio format disables signals for the rest of that generation, because the original sample timeline is no longer attestable. Primary frames continue in both cases. Muted 16 kHz samples still advance the original timeline without being sent to Oruk. The tap accepts chunks up to one second; VAD-prefix validation may require smaller chunks.

The synthetic `lookup_demo_order` tool accepts one `order_id` and knows only `DEMO-100`. It validates arguments, has a three-second timeout, is cancelled on interruption and performs no external action. The real Pipecat dispatcher rejects results after cancellation/timeout. Replace it only with explicitly authorized customer handlers and schemas; arbitrary Hume tool definitions are not executable imports.

## Optional expression signals

Signals are **off by default**. With `signals_enabled: true`, the CLI stays **trace-only**. Generic STT emission timestamps, transcript similarity and nearest-turn timing cannot establish that two providers processed the same audio. A real Deepgram audio-scope resolver has not been verified, so the CLI rejects `exact_scope`.

An application can explicitly supply a `scope_resolver` to the builder and set `signal_policy: "exact_scope"`. That resolver is responsible for identifying the exact immutable `AudioScope`: session generation, source track, sidecar utterance/request ID, and start/end positions in the original 16 kHz sample timeline. The audio tap places its scope on the VAD-stop frame's metadata as `oruk_migration_audio_scope`. Copying this blindly onto an asynchronous provider result does not prove correlation. The test STT owns the exact synthetic boundary, so its resolver is valid only for those tests.

The gate admits only already-completed signals whose scopes exactly match every primary segment in the user turn. It rejects missing identity, mismatched tracks, old generations, overlapping/out-of-order intervals and mismatched primary text. Text equality is a consistency check after identity, never an alignment method. It never waits for Oruk. Late results cannot trigger another LLM call or attach themselves to a later message.

Eligible notes contain bounded dynamic labels and finite scores; provider phrase text is excluded. They describe tentative acoustic expression estimates, not feelings, diagnoses or calibrated probabilities. Notes expire on response completion, tool continuation, new user speech, typed input, interruption and teardown. Primary text and the customer's system prompt remain unchanged.

The sidecar uses the published adapter's authenticated `wss://api.oruk.ai/v1/realtime?model=oruk-realtime` transport. It has one in-flight request, a bounded audio queue, a 60-second default audio limit, a two-second default finish deadline and a wall-clock deadline. An overlapping turn is skipped for signals while the primary conversation continues. Queue overflow, unsupported audio, transport errors and timeout cancel only the auxiliary request. There are **no automatic connection retries and no audio replay**.

Cancellation, a missing usage receipt, or an error after audio was sent does not prove zero billing. Every owned-request trace carries `billing_outcome: "unreconciled"`, including backpressure, format rejection and successful local completion. Status names describe local inference handling, not final settlement. A receipt received before a failure remains a receipt, not proof of successful inference. Check provider billing records before manually replaying any uncertain request.

The CLI disables framework debug sinks before each live session because those logs can contain transcripts and tool arguments. At normal session shutdown it prints a bounded JSON summary of generation/request IDs, status codes and available Oruk usage receipts. It does not print keys, audio, transcript text or phrase text. In-memory context/audio is sensitive; add your own redacted telemetry and retention controls before deployment. SDK/platform logging still deserves review in your deployment environment.

## What the tests establish

The new suite checks greeting and two-turn history; primary-provider preservation; real tool success, bad arguments, timeout, interruption and late-result rejection; queued playback interruption; disconnect/reconnect; timeout/failure isolation; exact scope rejection and expiry; dynamic/hostile labels; PCM fidelity including prefix and partial chunks; bounded backpressure; usage uncertainty; strict configuration; compatible provider constructors; credential redaction; and offline help/network guards.

These are deterministic framework tests with named provider doubles. They do not establish real provider accuracy, browser playback latency, customer account availability or exact primary-provider timing correlation. Before a customer cutover, separately record a real two-turn conversation, tool round trip, audible barge-in, cancellation, reconnect and timeout, with authorized audio and reconciled usage receipts from every selected provider.

Migration background: [Oruk Hume migration guide](https://oruk.ai/hume-migration). Package versions: [Pipecat 1.8.1](https://pypi.org/project/pipecat-ai/1.8.1/) and [pipecat-oruk 0.1.0rc3](https://pypi.org/project/pipecat-oruk/0.1.0rc3/).
