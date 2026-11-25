"""Exercise local and hosted English Fourier pipelines without downloading weights."""

import hashlib
import io
import sys
import wave
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from aiohttp import web
from pipecat.services.settings import STTSettings
from pipecat.transcriptions.language import Language

import pipecat_oruk
from pipecat_oruk import FourierSTTService, fourier


def _make_wav(pcm: np.ndarray | None = None, *, rate: int = 16000, channels: int = 1) -> bytes:
    if pcm is None:
        pcm = np.array([0, -32768, 16384], dtype="<i2")
    wav = io.BytesIO()
    with wave.open(wav, "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    return wav.getvalue()


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    weights = tmp_path / fourier.FILENAME
    weights.write_bytes(b"test model")
    monkeypatch.setattr(
        fourier, "SHA256", hashlib.sha256(weights.read_bytes()).hexdigest()
    )
    acquire = Mock(return_value=str(weights))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", acquire)
    calls = []
    engine = Mock()

    @contextmanager
    def session():
        def run(audio, **settings):
            calls.append((audio.copy(), settings))
            return SimpleNamespace(text="  transcript  ")

        yield SimpleNamespace(run=run)

    engine.session.side_effect = session
    make_model = Mock(return_value=engine)
    monkeypatch.setitem(
        sys.modules, "transcribe_cpp", SimpleNamespace(Model=make_model)
    )
    return weights, acquire, make_model, engine, calls


def test_top_level_export_and_release_version():
    assert FourierSTTService is fourier.FourierSTTService
    assert "FourierSTTService" in pipecat_oruk.__all__
    assert pipecat_oruk.__version__ == "0.1.0"


async def test_pinned_cache_and_independent_utterances(fake_engine):
    weights, acquire, make_model, engine, calls = fake_engine
    service = fourier.FourierSTTService(cache_dir="local", local_files_only=True)
    await service.prewarm()
    wav = _make_wav()
    for _ in range(2):
        (frame,) = [r async for r in service.run_stt(wav)]
        assert frame.text == "transcript" and frame.language == Language.EN
    acquire.assert_called_once_with(
        repo_id="oruk/oruk-fourier-2b",
        filename="Fourier-2B-Q8_0.gguf",
        revision="8bcce21695e4ba13569c48de6cedffcb8a7576f5",
        cache_dir="local",
        local_files_only=True,
    )
    make_model.assert_called_once_with(str(weights), backend="cpu")
    assert engine.session.call_count == 2
    for samples, settings in calls:
        np.testing.assert_array_equal(samples, [0.0, -1.0, 0.5])
        assert settings == {"language": "en", "timestamps": "none"}
    await service.cleanup()
    engine.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        await service.prewarm()


async def test_reject_corrupt_file_and_non_english(fake_engine):
    _, _, make_model, _, _ = fake_engine
    with pytest.raises(ValueError, match="English"):
        fourier.FourierSTTService(settings=STTSettings(language=Language.FR))
    fake_engine[0].write_bytes(b"corrupt")
    service = fourier.FourierSTTService(local_files_only=True)
    with pytest.raises(ValueError, match="checksum"):
        await service.prewarm()
    make_model.assert_not_called()
    await service.cleanup()


def test_long_turns_split_near_silence_and_keep_every_sample():
    rate = 16000
    pcm = np.full(61 * rate + 37, 0.75, dtype=np.float32)
    pcm[29 * rate : 29 * rate + 320] = 0
    chunks = list(fourier._split_audio(pcm, rate))
    assert 28 * rate < len(chunks[0]) <= 30 * rate
    assert len(chunks[0]) == 29 * rate + 320
    assert max(map(len, chunks)) <= 30 * rate
    np.testing.assert_array_equal(np.concatenate(chunks), pcm)


async def test_hosted_mode_posts_multipart_and_does_not_require_local_extra(monkeypatch):
    # Ensure hosted mode works even when transcribe_cpp is unavailable.
    monkeypatch.setitem(sys.modules, "transcribe_cpp", None)
    received: list[dict] = []

    async def handle_transcriptions(request: web.Request) -> web.Response:
        reader = await request.multipart()
        fields: dict[str, object] = {}
        while (part := await reader.next()) is not None:
            if part.name == "file":
                fields["file"] = await part.read()
                fields["filename"] = part.filename
            else:
                fields[part.name] = await part.text()
        received.append(
            {
                "auth": request.headers.get("Authorization"),
                "request_id": request.headers.get("X-Request-ID"),
                "fields": fields,
            }
        )
        return web.json_response(
            {
                "model": "oruk-fourier-2b",
                "text": "  Hosted Fourier transcript.  ",
                "model_revision": "8c2a8871f1304aa2f634da4fb95e24438c4945b9",
            },
            headers={"X-Request-ID": request.headers.get("X-Request-ID", "")},
        )

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", handle_transcriptions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    wav = _make_wav()
    service = FourierSTTService(
        mode="hosted",
        api_key="oruk-test-hosted-secret",
        base_url=f"http://127.0.0.1:{port}",
        settings=STTSettings(language="en-US"),
    )
    try:
        await service.prewarm()
        assert received == []
        (frame,) = [r async for r in service.run_stt(wav)]
        assert frame.text == "Hosted Fourier transcript."
        assert frame.language == Language.EN
        assert len(received) == 1
        assert received[0]["auth"] == "Bearer oruk-test-hosted-secret"
        assert received[0]["request_id"]
        assert received[0]["fields"] == {
            "model": "oruk-fourier-2b",
            "language": "en-US",
            "response_format": "json",
            "file": wav,
            "filename": "utterance.wav",
        }
        # Empty PCM returns zero frames without hitting network.
        empty_wav = _make_wav(np.array([], dtype="<i2"))
        assert [r async for r in service.run_stt(empty_wav)] == []
        assert len(received) == 1
    finally:
        await service.cleanup()
        await runner.cleanup()

    with pytest.raises(RuntimeError, match="closed"):
        await service.prewarm()


async def test_hosted_mode_redacts_api_key_on_http_error():
    async def handle_error(request: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"code": "invalid_api_key", "message": "secret leaked?"}},
            status=401,
            headers={"X-Request-ID": "req-err-1"},
        )

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", handle_error)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    secret = "oruk-super-secret-value-999"
    service = FourierSTTService(
        mode="hosted",
        api_key=secret,
        base_url=f"http://127.0.0.1:{port}",
    )
    try:
        with pytest.raises(RuntimeError) as exc_info:
            [r async for r in service.run_stt(_make_wav())]
        msg = str(exc_info.value)
        assert "401" in msg and "invalid_api_key" in msg and "req-err-1" in msg
        assert secret not in msg
    finally:
        await service.cleanup()
        await runner.cleanup()


def test_hosted_and_local_mode_validation(monkeypatch):
    monkeypatch.delenv("ORUK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="mode"):
        FourierSTTService(mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ORUK_API_KEY"):
        FourierSTTService(mode="hosted")
    with pytest.raises(ValueError, match="local mode"):
        FourierSTTService(mode="hosted", api_key="k", local_files_only=True)
    with pytest.raises(ValueError, match="local mode"):
        FourierSTTService(mode="hosted", api_key="k", cache_dir="/tmp/c")
    with pytest.raises(ValueError, match="hosted mode"):
        FourierSTTService(mode="local", api_key="k")
    with pytest.raises(ValueError, match="https"):
        FourierSTTService(mode="hosted", api_key="k", base_url="http://speech-api.oruk.ai")
    with pytest.raises(ValueError, match="English"):
        FourierSTTService(
            mode="hosted",
            api_key="k",
            settings=STTSettings(language=Language.FR),
        )
    with pytest.raises(ValueError, match="oruk-fourier-2b"):
        FourierSTTService(
            mode="hosted",
            api_key="k",
            settings=STTSettings(model="oruk-resonance"),
        )
