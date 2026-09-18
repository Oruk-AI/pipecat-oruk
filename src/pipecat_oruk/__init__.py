"""Oruk-maintained Pipecat speech and phrase-emotion integration."""

from ._service import (
    OrukCommitFrame,
    OrukEventFrame,
    OrukPhraseEmotionFrame,
    OrukSpeakerBoundaryFrame,
    OrukSTTService,
    OrukTurnCompletedFrame,
)
from .realtime import RealtimeOptions

__version__ = "0.1.0rc2"
__all__ = [
    "OrukCommitFrame", "OrukEventFrame", "OrukPhraseEmotionFrame",
    "OrukSpeakerBoundaryFrame", "OrukSTTService", "OrukTurnCompletedFrame", "RealtimeOptions",
]
