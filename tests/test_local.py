"""Local integrity, utterance boundaries, and cancellation without model downloads."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.tests.utils import run_test

pytest.importorskip("onnx_asr")
from pipecat_oruk import _local_model as _model
from pipecat_oruk.local import OrukeetSTTService


@pytest.fixture
def model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Small pinned files exercise the downloader without fetching real weights."""
    folder = tmp_path / _model.SUBFOLDER
    folder.mkdir(parents=True)
    payload = b"verified fixture"
    hashes = {name: hashlib.sha256(payload).hexdigest() for name in _model.FILE_HASHES}
    for name in (*hashes, *_model.LICENSE_FILES):
        (folder / name).write_bytes(payload)
    monkeypatch.setattr(_model, "FILE_HASHES", hashes)
    return folder


def test_complete_cache_does_not_request_http(
    model_cache: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        _model,
        "try_to_load_from_cache",
        lambda repo, name, **kwargs: str(model_cache / Path(name).name),
    )
    download = Mock(side_effect=AssertionError("complete cache must not request HTTP"))
    monkeypatch.setattr(_model, "snapshot_download", download)
    assert _model.download_model(local_files_only=True) == model_cache
    download.assert_not_called()


def test_missing_cache_downloads_pinned_required_files(model_cache: Path, monkeypatch):
    monkeypatch.setattr(_model, "try_to_load_from_cache", lambda *args, **kwargs: None)
    download = Mock(return_value=str(model_cache.parents[1]))
    monkeypatch.setattr(_model, "snapshot_download", download)
    assert _model.download_model(cache_dir="chosen-cache") == model_cache
    assert download.call_args.kwargs == {
        "repo_id": "oruk/orukeet",
        "revision": _model.REVISION,
        "cache_dir": "chosen-cache",
        "local_files_only": False,
        "allow_patterns": [
            f"{_model.SUBFOLDER}/{name}"
            for name in (*_model.FILE_HASHES, *_model.LICENSE_FILES)
        ],
    }


@pytest.mark.parametrize("filename", tuple(_model.FILE_HASHES))
def test_corruption_is_rejected_without_deleting_other_files(
    model_cache, monkeypatch, filename
):
    monkeypatch.setattr(
        _model,
        "try_to_load_from_cache",
        lambda repo, name, **kwargs: str(model_cache / Path(name).name),
    )
    (model_cache / filename).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _model.download_model(local_files_only=True)
    assert all((model_cache / name).exists() for name in _model.FILE_HASHES)
    (model_cache / filename).write_bytes(b"verified fixture")
    assert _model.download_model(local_files_only=True) == model_cache


def test_offline_miss_is_not_retried_online(monkeypatch):
    monkeypatch.setattr(_model, "try_to_load_from_cache", lambda *args, **kwargs: None)
    download = Mock(side_effect=FileNotFoundError("missing cached weights"))
    monkeypatch.setattr(_model, "snapshot_download", download)
    with pytest.raises(FileNotFoundError):
        _model.download_model(local_files_only=True)
    download.assert_called_once()
    assert download.call_args.kwargs["local_files_only"] is True


def wav_bytes(pcm=b"\0\0" * 160, rate=16000, channels=1):
    data = io.BytesIO()
    with wave.open(data, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return data.getvalue()


async def test_real_pipeline_preserves_separate_utterances_and_final_contract():
    service = OrukeetSTTService(local_files_only=True)
    recognize = Mock(side_effect=["Hello.", "Bonjour."])
    service._runtime = SimpleNamespace(recognize=recognize, close=lambda: None)
    chunks = [
        np.array([0, 16384, -16384], dtype=np.int16),
        np.array([8192, -8192], dtype=np.int16),
    ]
    frames = []
    for chunk in chunks:
        frames += [
            VADUserStartedSpeakingFrame(start_secs=0.2),
            InputAudioRawFrame(
                audio=chunk.tobytes(), sample_rate=16000, num_channels=1
            ),
            VADUserStoppedSpeakingFrame(stop_secs=0.2),
        ]
    async with asyncio.timeout(20):
        down, up = await run_test(
            service,
            frames_to_send=frames,
            pipeline_params=PipelineParams(audio_in_sample_rate=16000),
            start_timeout=8,
        )
    finals = [frame for frame in down if isinstance(frame, TranscriptionFrame)]
    assert [frame.text for frame in finals] == ["Hello.", "Bonjour."]
    assert all(frame.finalized and frame.language is None for frame in finals)
    assert not any(isinstance(frame, ErrorFrame) for frame in up)
    for call, chunk in zip(recognize.call_args_list, chunks):
        np.testing.assert_array_equal(call.args[0], chunk.astype(np.float32) / 32768.0)
        assert call.args[1] == 16000


async def test_empty_recognition_emits_no_transcript():
    service = OrukeetSTTService(local_files_only=True)
    service._runtime = SimpleNamespace(recognize=lambda *args: "", close=lambda: None)
    assert [frame async for frame in service.run_stt(wav_bytes())] == []
    await service.cleanup()


@pytest.mark.parametrize(
    "rate,channels,pcm", [(48000, 1, b"\0\0"), (16000, 2, b"\0" * 4), (16000, 1, b"\0")]
)
async def test_mismatched_input_rejected_before_buffering(rate, channels, pcm):
    service = OrukeetSTTService(local_files_only=True)
    with pytest.raises(ValueError, match="mono PCM16"):
        await service.process_audio_frame(
            InputAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=channels),
            FrameDirection.DOWNSTREAM,
        )
    assert not service._audio_buffer
    await service.cleanup()


@pytest.mark.parametrize(
    "settings", [STTSettings(language="en"), STTSettings(model="other")]
)
async def test_unsupported_settings_do_not_change_recognizer(settings):
    with pytest.raises(ValueError, match="automatic language"):
        OrukeetSTTService(settings=settings)
    service = OrukeetSTTService(local_files_only=True)
    service.push_error = AsyncMock()
    assert await service._update_settings(settings) == {}
    service.push_error.assert_awaited_once()
    await service.cleanup()


async def test_cancelled_native_work_finishes_before_cleanup():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocking(*args, **kwargs):
        started.set()
        assert release.wait(5)
        finished.set()
        return "stale transcript"

    service = OrukeetSTTService(local_files_only=True)
    service._runtime._recognizer = SimpleNamespace(recognize=blocking)

    async def consume():
        return [frame async for frame in service.run_stt(wav_bytes())]

    task = asyncio.create_task(consume())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cleanup = asyncio.create_task(service.cleanup())
        await asyncio.sleep(0)
        assert not cleanup.done()
    finally:
        release.set()
    await asyncio.wait_for(cleanup, 5)
    assert finished.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        await consume()
