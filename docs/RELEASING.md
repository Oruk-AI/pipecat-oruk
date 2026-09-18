# Release procedure

The current checkout prepares `0.1.0rc2`. Building or merging it does not publish a package. The existing manual **Publish Pipecat integration** workflow is the only publishing path documented here.

1. Review the diff and changelog. Keep `pyproject.toml`, `pipecat_oruk.__version__`, and the version references in `.github/workflows/publish.yml` synchronized. The workflow also verifies the installed package's version.
2. Require the package CI matrix to pass for the exact commit: Linux Python 3.11–3.14 and macOS/Windows Python 3.12. These tests exercise Pipecat 1.8.1 with a simulated hosted gateway and local-recognizer fixtures; they do not download model weights or replace real-model qualification.
3. From a clean Python 3.12 environment, reproduce the build and installed-wheel checks:

   ```sh
   python -m pip install '.[agent,test,local]' 'build>=1,<2' 'twine>=6,<8'
   python -m build
   python -m twine check --strict dist/*
   python -m pip install --force-reinstall --no-deps dist/*.whl
   python -c "import importlib.metadata, pipecat_oruk; from pipecat_oruk.local import OrukeetSTTService; assert pipecat_oruk.__version__ == importlib.metadata.version('pipecat-oruk') == '0.1.0rc2'"
   python -m pytest -q tests
   python -m pip check
   ```

4. Record real-model checks separately, with the candidate wheel installed: repeated speech, silence, cached offline reload, and a complete `VADProcessor`/`PipelineWorker` utterance. Use `examples/pipecat_local_file.py recording.wav --offline` with a previously verified cache. Record versions and fixture hashes. Never treat the source model's published WER as a measured accuracy or speed result for this adapter.
5. After review and a release decision, merge the candidate to `main`. An authorized maintainer can then manually dispatch **Publish Pipecat integration** on `main`. The `verify` job builds and tests the wheel, including the optional local dependencies. The `publish` job uses the existing `pypi` environment and trusted publishing; do not relax that environment's protection or add credentials to this repository.
6. Inspect the `registry` job. It installs the exact release with `[agent,test,local]`, checks both hosted and local imports and the version, then runs the tests and dependency check. PyPI versions cannot be overwritten: a failed post-publication check requires an assessed follow-up version, not a rerun with different bytes under the same version.
7. If the package is available and its registry checks pass, update the Pipecat community guide from its pinned source install to the released local extra. A GitHub release or tag is a separate decision; do not imply either exists before it does.

The model is acquired only when the local service loads. Its required pinned `config.json` uses Hugging Face's normal download accounting; complete cache hits make no requests. Package publication must not download model files just to increment that counter.
