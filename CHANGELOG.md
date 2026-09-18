# Changelog

## 0.1.0rc3 — 2026-09-18

Documentation-only follow-up: point both hosted and local installation commands to the released package instead of the older hosted-only candidate or a source checkout. Model files, runtime code, dependencies and behavior are unchanged from `0.1.0rc2`.

## 0.1.0rc2 — 2026-09-18

- Add optional CPU Orukeet transcription after Pipecat VAD through the `local` extra.
- Verify pinned Hugging Face runtime files, retain weight notices, and support complete offline caches.
- Add a local WAV example and tests for final utterances, invalid input/settings, cache integrity, and cancellation/cleanup.
- Verify the installed local adapter in the manual release workflow and in the registry smoke test. Publishing remains a separate maintainer action.

## 0.1.0rc1 — 2026-09-11

Initial Oruk-maintained streaming STT and phrase-emotion integration, with VAD/manual turns, bounded buffering, guarded retries, usage metrics, and runnable examples. Production browser verification and a recorded demonstration are included in docs/.
