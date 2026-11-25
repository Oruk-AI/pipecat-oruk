# Changelog

## 0.1.0 — 2026-11-25

- First stable PyPI release (`pip install pipecat-oruk` without `--pre`), consolidating `main` with the `0.1.0rc4` Fourier local GGUF integration.
- Re-export `FourierSTTService` at top level (`from pipecat_oruk import FourierSTTService`).
- Add hosted `FourierSTTService(mode="hosted")` targeting `POST https://speech-api.oruk.ai/v1/audio/transcriptions` (`model="oruk-fourier-2b"`) via core `aiohttp` (zero extra dependencies, `ORUK_API_KEY`, English-only validation, safe error redaction).
- Link the Pipecat section of the Oruk SDK docs and the live Oruk Realtime API anchor (`https://oruk.ai/docs#realtime`), stating the hosted Realtime preview status (10-minute session cap; no published latency yet).

## 0.1.0rc4 — 2026-11-23

- Add optional local CPU English transcription with `FourierSTTService` (`pipecat-oruk[fourier]`, pinned `oruk/oruk-fourier-2b` Q8_0 GGUF with SHA-256 verification).

## 0.1.0rc3 — 2026-09-18

Documentation-only follow-up: point both hosted and local installation commands to the released package instead of the older hosted-only candidate or a source checkout. Model files, runtime code, dependencies and behavior are unchanged from `0.1.0rc2`.

## 0.1.0rc2 — 2026-09-18

- Add optional CPU Orukeet transcription after Pipecat VAD through the `local` extra.
- Verify pinned Hugging Face runtime files, retain weight notices, and support complete offline caches.
- Add a local WAV example and tests for final utterances, invalid input/settings, cache integrity, and cancellation/cleanup.
- Verify the installed local adapter in the manual release workflow and in the registry smoke test. Publishing remains a separate maintainer action.

## 0.1.0rc1 — 2026-09-11

Initial Oruk-maintained streaming STT and phrase-emotion integration, with VAD/manual turns, bounded buffering, guarded retries, usage metrics, and runnable examples. Production browser verification and a recorded demonstration are included in docs/.
