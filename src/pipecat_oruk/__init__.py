"""Oruk-maintained Pipecat speech and phrase-emotion integration."""

from ._service import (
    OrukCommitFrame,
    OrukEventFrame,
    OrukPhraseEmotionFrame,
    OrukSpeakerBoundaryFrame,
    OrukSTTService,
    OrukTurnCompletedFrame,
)
from .fourier import FourierSTTService
from .realtime import RealtimeOptions

__version__ = "0.1.0"
__all__ = [
    "FourierSTTService",
    "OrukCommitFrame",
    "OrukEventFrame",
    "OrukPhraseEmotionFrame",
    "OrukSTTService",
    "OrukSpeakerBoundaryFrame",
    "OrukTurnCompletedFrame",
    "RealtimeOptions",
]
