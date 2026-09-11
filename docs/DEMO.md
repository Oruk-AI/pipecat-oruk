# Production browser demonstration

[Watch the 45-second demonstration](pipecat-demo.mp4).

Recorded September 11, 2026 using Pipecat 1.8.1 and the production Oruk realtime
API. The browser's live canvas displays the actual returned text, phrase
estimates, and request receipts. Input is the public Oruk quickstart recording.

Three turns finish successfully. A fourth is disconnected after a real interim
transcript and has no completed-turn receipt. Reconnection opens a fresh session.
Backend metering matches all four requests, including 1.58 seconds on the cancelled
one. The unedited original WebM is retained in Oruk's execution evidence; the MP4
changes the container/codec for convenient playback and preserves the full timeline.

[Verification details and hashes](pipecat-verification.json). This is a transport
and service integration check, not an independent benchmark or customer outcome.
A spoken LLM/TTS assistant and assistant barge-in are separate, unverified paths.

Demo recording and branding © 2026 Oruk labs. They are separate from the
MIT-licensed program. The existing third-party test fixture has its own attribution.
