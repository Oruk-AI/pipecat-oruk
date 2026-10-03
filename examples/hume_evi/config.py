"""Small explicit configuration; Hume exports and credentials are not imported."""

from dataclasses import dataclass
import json
from pathlib import Path

VAD_PREFIX_SECONDS = 0.5


@dataclass(frozen=True)
class Config:
    system_prompt: str = "You are a helpful voice assistant. Keep spoken answers brief."
    greeting: str = "Hello. How can I help?"
    signals_enabled: bool = False
    signal_policy: str = "trace_only"
    signal_timeout: float = 2.0
    signal_buffer_seconds: float = 2.0
    max_turn_seconds: float = 60.0

    def __post_init__(self):
        if (
            not isinstance(self.system_prompt, str)
            or not 1 <= len(self.system_prompt) <= 4000
        ):
            raise ValueError("Invalid system_prompt")
        if not isinstance(self.greeting, str) or len(self.greeting) > 500:
            raise ValueError("Invalid greeting")
        if (
            type(self.signals_enabled) is not bool
            or not isinstance(self.signal_policy, str)
            or self.signal_policy not in {"trace_only", "exact_scope"}
        ):
            raise ValueError("Invalid signal policy")
        for name, low, high in [
            ("signal_timeout", 0.05, 20),
            ("signal_buffer_seconds", VAD_PREFIX_SECONDS, 30),
            ("max_turn_seconds", 0.1, 60),
        ]:
            value = getattr(self, name)
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError("Invalid signal limit")


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as source:
        data = source.read(16_385)
    if len(data) > 16_384:
        raise ValueError("Configuration exceeds 16 KiB")

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate configuration key")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value.pop("schema_version") != 1
        ):
            raise ValueError("Unsupported configuration schema")
        if set(value) - Config.__dataclass_fields__.keys():
            raise ValueError(
                "Unsupported configuration fields; review Hume settings manually"
            )
        return Config(**value)
    except (UnicodeError, json.JSONDecodeError, RecursionError, TypeError):
        raise ValueError("Invalid configuration") from None
