import asyncio
from dataclasses import replace

import pytest

from pipecat_oruk.realtime import (
    EMOTION,
    TRANSCRIPT,
    RealtimeError,
    RealtimeOptions,
    new_http_session,
    run_turn,
)


async def audio(pcm=b"\x01\x00" * 1_024):
    yield pcm


async def run(server, *, source=None, on_event=lambda _: None, **kwargs):
    async with new_http_session() as session:
        return await run_turn(
            session=session,
            api_key="test-key",
            endpoint=server.endpoint,
            audio=source if source is not None else audio(),
            request_id="req_test",
            options=RealtimeOptions(finish_timeout=0.2),
            on_event=on_event,
            connect_timeout=0.5,
            max_connect_retries=2,
            retry_interval=0,
            **kwargs,
        )


async def test_streams_before_commit_with_exact_pcm_and_header_auth(gateway):
    interim = asyncio.Event()
    seen = []
    pcm = bytes(range(256)) * 100

    async def source():
        yield pcm
        await asyncio.wait_for(interim.wait(), 2)

    def on_event(event):
        seen.append(event)
        if event["type"] == TRANSCRIPT + "delta":
            interim.set()

    async with gateway() as server:
        result = await run(server, source=source(), on_event=on_event)
        assert server.audio == [pcm]
        assert server.auth_headers == ["Bearer test-key"]
        assert server.queries == ["model=oruk-realtime"]
        assert server.configs[0]["sample_rate"] == 16_000
        assert result.transcript == "Final turn 0."
        assert result.audio_seconds == len(pcm) / 32_000
        assert result.usage["billable_seconds"] == 1
        assert result.phrases[0]["text"] != result.transcript
        assert seen[0]["type"] == TRANSCRIPT + "delta"


@pytest.mark.parametrize(
    "mode",
    [
        "disconnect",
        "invalid_json",
        "final_only",
        "usage_only",
        "hang",
        "bad_score",
        "conflicting_final",
        "error_after_usage",
    ],
)
async def test_incomplete_or_invalid_turn_fails_without_replaying_audio(gateway, mode):
    async with gateway(mode=mode) as server:
        with pytest.raises(RealtimeError) as error:
            await asyncio.wait_for(run(server), 3)
        assert not error.value.retryable
        assert server.requests == 1


async def test_only_pre_audio_transient_failure_retries(gateway):
    consumed = 0

    async def source():
        nonlocal consumed
        consumed += 1
        yield b"\x01\x00" * 100

    async with gateway(fail_upgrades=[503, 429]) as server:
        result = await run(server, source=source())
        assert server.requests == 3
        assert consumed == 1
        assert result.transcript == "Final turn 0."


async def test_auth_failure_is_not_retried_or_leaked(gateway):
    async with gateway(fail_upgrades=[401]) as server:
        with pytest.raises(RealtimeError) as error:
            await run(server)
        assert server.requests == 1
        assert error.value.status == 401
        assert "test-key" not in str(error.value)


async def test_redirect_never_receives_credentials(gateway):
    async with gateway(mode="redirect") as server:
        with pytest.raises(RealtimeError, match="websocket_redirect_refused"):
            await run(server)
        assert server.requests == 1


async def test_duplicate_events_are_idempotent_and_phrase_failure_is_auxiliary(gateway):
    for mode in ("duplicate", "phrase_failure"):
        seen = []
        async with gateway(mode=mode) as server:
            result = await run(server, on_event=seen.append)
            assert result.transcript == "Final turn 0."
            assert len(result.phrases) == 1
            assert sum(e["type"] == TRANSCRIPT + "completed" for e in seen) == 1
            assert sum(e["type"].startswith(EMOTION) for e in seen) == 1


async def test_cancel_closes_socket_and_audio_generator(gateway):
    source_closed = asyncio.Event()

    async def source():
        try:
            yield b"\x01\x00" * 100
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    async with gateway() as server:
        task = asyncio.create_task(run(server, source=source()))
        await asyncio.wait_for(server.first_audio.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(server.closed.wait(), 2)
        assert source_closed.is_set()


@pytest.mark.parametrize("pcm", [b"", b"\x01", "not bytes"])
async def test_invalid_audio_rejected(gateway, pcm):
    async with gateway() as server:
        with pytest.raises(RealtimeError, match="empty_audio|invalid_pcm16"):
            await run(server, source=audio(pcm))


async def test_audio_limit_is_explicit(gateway):
    async with gateway() as server, new_http_session() as session:
        with pytest.raises(RealtimeError, match="turn_too_long"):
            await run_turn(
                session=session,
                api_key="test-key",
                endpoint=server.endpoint,
                audio=audio(b"\x01\x00" * 1601),
                request_id="req_test",
                options=RealtimeOptions(max_turn_seconds=0.1),
                on_event=lambda _: None,
            )


@pytest.mark.parametrize(
    "options",
    [
        {"finish_timeout": 0},
        {"finish_timeout": float("nan")},
        {"max_turn_seconds": 600},
        {"phrase_silence_ms": 199},
        {"phrase_max_ms": True},
        {"diarize": 1},
        {"language": "../../oops"},
    ],
)
def test_configuration_rejects_invalid_values(options):
    with pytest.raises(ValueError):
        replace(RealtimeOptions(), **options)
