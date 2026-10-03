import json
import socket

import pytest

from examples.hume_evi.bot import settings
from examples.hume_evi.config import Config, load_config
from hume_helpers import credential_free_subprocess_env, loopback_only  # noqa: F401

pytestmark = pytest.mark.usefixtures("loopback_only")


def test_network_guard():
    with pytest.raises(AssertionError, match="Non-loopback"):
        socket.getaddrinfo("api.example.com", 443)
    with socket.socket() as sock, pytest.raises(AssertionError, match="Non-loopback"):
        sock.connect(("1.1.1.1", 443))


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":true}',
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"unknown_secret":"never-print-this"}',
        "[]",
        "{",
        '{"schema_version":1,"signal_timeout":NaN}',
        '{"schema_version":1,"signal_policy":[]}',
        '{"schema_version":1,"signals_enabled":1}',
        '{"schema_version":1,"max_turn_seconds":61}',
        " " * 16385,
    ],
)
def test_strict_bounded_configuration(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(payload)
    with pytest.raises(ValueError) as error:
        load_config(path)
    assert "never-print-this" not in str(error.value)


def test_defaults_and_missing_credentials_are_offline(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1}))
    assert load_config(path) == Config()
    monkeypatch.setenv("HUME_STARTER_CONFIG", str(path))
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="STT_PROVIDER"):
        settings()
    path.write_text(json.dumps({"schema_version": 1, "signal_policy": "exact_scope"}))
    with pytest.raises(ValueError, match="no verified provider audio-scope resolver"):
        settings()


async def test_pinned_provider_constructors_are_compatible_and_offline():
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.tts import OpenAITTSService

    primary = DeepgramSTTService(
        api_key="synthetic-key",
        settings=DeepgramSTTService.Settings(model="synthetic-model"),
    )
    llm = OpenAILLMService(
        api_key="synthetic-key",
        settings=OpenAILLMService.Settings(model="synthetic-model"),
    )
    tts = OpenAITTSService(
        api_key="synthetic-key",
        settings=OpenAITTSService.Settings(
            model="synthetic-model", voice="synthetic-voice"
        ),
    )
    assert primary and llm and tts


def test_subprocess_environment_preserves_platform_essentials_only():
    essentials = {
        "Path": "synthetic-bin",
        "SystemRoot": "synthetic-windows",
        "WINDIR": "synthetic-windows",
        "TEMP": "synthetic-temp",
        "TMP": "synthetic-temp",
    }
    assert (
        credential_free_subprocess_env(
            {
                **essentials,
                "OPENAI_API_KEY": "synthetic-secret",
                "DEEPGRAM_API_KEY": "synthetic-secret",
                "ORUK_API_KEY": "synthetic-secret",
                "STT_PROVIDER": "synthetic-provider",
                "PYTHONPATH": "untrusted-import-path",
            }
        )
        == essentials
    )


def test_help_cannot_open_network_or_require_credentials(tmp_path):
    import os
    import subprocess
    import sys

    # The subprocess blocks even loopback: --help must not start any server.
    (tmp_path / "sitecustomize.py").write_text(
        "import socket\n"
        "def blocked(*a, **k): raise AssertionError('network forbidden in help')\n"
        "socket.socket.connect = blocked\nsocket.socket.connect_ex = blocked\nsocket.getaddrinfo = blocked\n"
    )
    env = {
        **credential_free_subprocess_env(os.environ),
        "PYTHONPATH": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "examples.hume_evi.bot", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--host" in result.stdout
