"""Exercise the optional local English pipeline without downloading weights."""

from contextlib import contextmanager
import hashlib
import io
import sys
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pipecat.transcriptions.language import Language
from pipecat.services.settings import STTSettings

from pipecat_oruk import fourier


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


async def test_pinned_cache_and_independent_utterances(fake_engine):
    weights, acquire, make_model, engine, calls = fake_engine
    service = fourier.FourierSTTService(cache_dir="local", local_files_only=True)
    await service.prewarm()
    pcm = np.array([0, -32768, 16384], dtype="<i2")
    wav = io.BytesIO()
    with wave.open(wav, "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    for _ in range(2):
        (frame,) = [r async for r in service.run_stt(wav.getvalue())]
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
