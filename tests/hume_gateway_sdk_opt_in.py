"""Opt-in actual npm SDK boundary. Explicit path is required; no missing-dependency skip.

Root supplies pinned HUME_GATEWAY_NODE and HUME_GATEWAY_SDK_ROOT paths. This
file is deliberately not named test_*. It owns one non-detached Node child per
case; root still owns the enclosing finite process group.
"""
import asyncio
import json
import os
from pathlib import Path

import pytest

from hume_gateway_helpers import local_gateway
from hume_gateway_helpers import exact_loopback_only  # noqa: F401 - autouse fixture
from hume_helpers import eventually


def selected_path(name, *, directory=False):
    value = os.environ.get(name, "")
    candidate = Path(value)
    assert value and candidate.is_absolute(), f"explicit_{name}_required"
    candidate = candidate.resolve(strict=True)
    assert candidate.is_dir() if directory else candidate.is_file()
    return str(candidate)


async def bounded_read(stream, maximum):
    result = bytearray()
    while chunk := await stream.read(4096):
        result.extend(chunk)
        assert len(result) <= maximum, "sdk_child_output_cap"
    return bytes(result)


async def child(node, sdk, port, mode, workdir):
    process = None
    reads = []
    env = {"TMPDIR": str(workdir), "PATH": str(Path(node).parent),
           "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "NO_COLOR": "1"}
    try:
        process = await asyncio.create_subprocess_exec(
            node, str(Path(__file__).with_name("hume_gateway_sdk.mjs")), sdk, str(port), mode,
            env=env, cwd=str(workdir), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            # No detached session: the root supervisor owns this whole group.
        )
        reads = [asyncio.create_task(bounded_read(process.stdout, 64 * 1024)),
                 asyncio.create_task(bounded_read(process.stderr, 64 * 1024))]
        async with asyncio.timeout(25):
            stdout, stderr, returncode = await asyncio.gather(*reads, process.wait())
        assert len(stderr) <= 64 * 1024
        # Do not print raw stderr, URLs, provider output or assertion values.
        try:
            result = json.loads(stdout)
        except (ValueError, UnicodeError):
            raise AssertionError("sdk_child_invalid_receipt") from None
        safe = {key: result.get(key) for key in ("ok", "phase", "error_code", "violations", "open_sockets", "open_websockets")}
        assert returncode == 0 and result.get("ok") is True, safe
        assert result["sdk"] == "hume@0.15.17"
        assert result["websocket_constructors"] == result["socket_connections"] == 1
        assert result["open_sockets"] == result["open_websockets"] == 0
        assert result["violations"] == []
        return result
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 2)
            except asyncio.TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), 2)
        for task in reads:
            if not task.done():
                task.cancel()
        await asyncio.gather(*reads, return_exceptions=True)
        assert process is None or process.returncode is not None, "sdk_child_not_reaped"


@pytest.mark.parametrize("mode", ["normal", "tool"])
async def test_public_hume_sdk_against_actual_gateway(mode, tmp_path):
    node = selected_path("HUME_GATEWAY_NODE")
    sdk = selected_path("HUME_GATEWAY_SDK_ROOT", directory=True)
    tmp_path.chmod(0o700)
    async with local_gateway(mode=mode) as (_, url, gateway, made, _):
        port = int(url.split(":")[2].split("/")[0])
        result = await child(node, sdk, port, mode, tmp_path)
        await eventually(lambda: not gateway.active)
        assert len(made) == len(gateway.reports) == 1
        assert made[0].closed.is_set()
        assert gateway.reports[0]["all_owned_tasks_done"]
        assert gateway.reports[0]["provider_cleanup_joined"]
        assert all(task.done() for task in made[0].background)
        returned_ids = [row["id"] for row in result["outcomes"]]
        assert returned_ids == list(dict.fromkeys(made[0].tokens))
        assert all(value.split(".")[0] == gateway.reports[0]["generation"] for value in returned_ids)
        if mode == "normal":
            assert [row["kind"] for row in result["outcomes"]] == ["typed_1", "typed_2", "pcm_input"]
            assert bytes(made[0].stt.pcm) == b"\x02\x00" * 320
            assert len(made[0].llm.inputs) == 3
            assert gateway.reports[0]["turns"] == 3 and gateway.reports[0]["input_bytes"] == 640
        else:
            assert [row["kind"] for row in result["outcomes"]] == ["tool_continuation"]
            assert "SDK synthetic packed." in json.dumps(made[0].llm.inputs[-1])
            assert len(made[0].tokens) == 2 and len(set(made[0].tokens)) == 1
            assert bytes(made[0].stt.pcm) == b""
        # Compact receipt is synthetic and contains no URL, query or transcript.
        print(json.dumps({"mode": mode, "sdk_result": result,
                          "provider_closed": True, "gateway_owner_closed": True}, sort_keys=True))
