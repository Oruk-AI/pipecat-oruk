"""Local English Fourier recognition for Pipecat's segmented speech pipeline."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np
from pipecat.services.settings import STTSettings
from pipecat.transcriptions.language import Language

from .local import _LocalSTTService

MODEL_ID = "oruk/oruk-fourier-2b"
REVISION = "8bcce21695e4ba13569c48de6cedffcb8a7576f5"
FILENAME = "Fourier-2B-Q8_0.gguf"
SHA256 = "6a7c419bd571f7b0e21da2d5ab1e3a1a3938a83470d622be57427af98f322981"


def _split_audio(audio: np.ndarray, rate: int):
    """Prefer a quiet 20 ms boundary near the end of a long model window."""
    limit = 30 * rate
    frame = rate // 50
    while len(audio) > limit:
        start = limit - 2 * rate
        window = audio[start:limit].reshape(-1, frame)
        power = (window * window).mean(axis=1)
        boundary = start + (int(np.argmin(power)) + 1) * frame
        yield audio[:boundary]
        audio = audio[boundary:]
    if audio.size:
        yield audio


class _FourierRuntime:
    def __init__(self, cache_dir: str | None, local_files_only: bool):
        try:
            from huggingface_hub import hf_hub_download
            from transcribe_cpp import Model
        except ImportError as exc:
            raise ImportError(
                "Install the fourier extra to use FourierSTTService."
            ) from exc
        self._download = hf_hub_download
        self._model_class = Model
        self._cache_dir = cache_dir
        self._offline = local_files_only
        self._model = None
        self._lock = threading.Lock()
        self._closed = False

    def _load(self):
        if self._closed:
            raise RuntimeError("Fourier STT is closed")
        if self._model is None:
            path = self._download(
                repo_id=MODEL_ID,
                revision=REVISION,
                filename=FILENAME,
                cache_dir=self._cache_dir,
                local_files_only=self._offline,
            )
            with Path(path).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                    raise ValueError("Fourier model checksum mismatch")
            self._model = self._model_class(str(path), backend="cpu")
        return self._model

    def prewarm(self):
        with self._lock:
            self._load()

    def recognize(self, audio: np.ndarray, sample_rate: int) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("Fourier STT is closed")
            if not audio.size:
                return ""
            if sample_rate != 16000:
                raise ValueError("Fourier requires 16 kHz input")
            model = self._load()
            parts = []
            for chunk in _split_audio(audio, sample_rate):
                with model.session() as session:
                    text = session.run(
                        chunk, language="en", timestamps="none"
                    ).text.strip()
                    if text:
                        parts.append(text)
            return " ".join(parts)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._model is not None:
                self._model.close()
                self._model = None


class FourierSTTService(_LocalSTTService):
    """Produce final English text from local, mono 16 kHz speech segments.

    The first segment or prewarm() loads the pinned Q8_0 file from Oruk.
    Set local_files_only=True to use an existing Hub cache offline.
    """

    Settings = STTSettings
    transcript_language = Language.EN

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        settings: STTSettings | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("sample_rate", 16000) != 16000:
            raise ValueError("FourierSTTService requires a 16 kHz input pipeline.")
        self._validate_settings(settings or STTSettings())
        super().__init__(
            sample_rate=16000,
            settings=STTSettings(model=MODEL_ID, language=Language.EN),
            **kwargs,
        )
        self._runtime = _FourierRuntime(cache_dir, local_files_only)

    @staticmethod
    def _validate_settings(settings: STTSettings) -> None:
        values = settings.given_fields()
        if (
            values.keys() - {"model", "language"}
            or values.get("model", MODEL_ID) != MODEL_ID
            or values.get("language", Language.EN) not in (Language.EN, "en")
        ):
            raise ValueError("Fourier uses a fixed English speech model.")
