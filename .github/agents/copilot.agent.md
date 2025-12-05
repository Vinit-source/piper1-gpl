---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config

name: copilot-agent
description: Copilot instructions for Piper TTS (piper1-gpl)
---

# Copilot Agent

<!-- Copilot instructions for Piper TTS (piper1-gpl) -->
# Piper — Copilot Instructions

This file gives an AI coding agent the key, actionable knowledge to be productive in this repository.
Be concise: refer to the exact files/commands below rather than general advice.

## Big picture
- **Purpose:** Piper is a local neural text-to-speech engine that embeds `espeak-ng` for phonemization and runs voice models (ONNX) to synthesize audio.
- **Major components:**
  -`notebooks/piper_finetuning_indian_english_from_gdrive.ipynb`: complete notebook for fine-tuning a voice from GDrive-hosted data.
  - `src/piper/`: primary Python package (CLI, HTTP server, download helpers, training code).
  - `libpiper/`: C/C++ API wrapper and native code used for lower-level integration.
  - `docs/`: user-facing docs (`BUILDING.md`, `CLI.md`, `TRAINING.md`, `API_*`).
  - `script/`: small Python wrappers used by developers (`dev_build`, `run`, `train`).

## Key files to read first
- `README.md` — project overview and links (already contains pointers to docs).
- `docs/BUILDING.md` — how to build the C-extension / wheels (CMake, scikit-build-core).
- `docs/CLI.md` — examples for running the CLI and downloading voices.
- `docs/TRAINING.md` — training and export workflow (PyTorch Lightning + monotonic align extension).
- `src/piper/__main__.py` and `src/piper/http_server.py` — entry points for CLI and HTTP API.
- `espeakbridge.c` — how espeak-ng is bridged into Python (important for phoneme handling).
- -`notebooks/piper_finetuning_indian_english_from_gdrive.ipynb`: complete notebook for fine-tuning a voice from GDrive-hosted data.

## Typical developer workflows (copyable commands)
- Create dev venv and install dev deps:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python3 -m pip install -e .[dev]`
- Build native extension (in-repo development):
  - `script/dev_build` (runs `python3 setup.py build_ext --inplace`)
- Run the CLI (fast iteration):
  - `script/run -- -m en_US-lessac-medium -- 'This is a test.'`
- Train a model (training deps + build monotonic align):
  - `python3 -m pip install -e .[train]`
  - `./build_monotonic_align.sh` (builds the cython/C extension used by training)
  - `script/train` (wrapper for `python3 -m piper.train fit`)
- Build release wheel:
  - `python3 -m build`

## Project-specific conventions & patterns
- The repository embeds `espeak-ng` to produce phonemes + a *terminator* — punctuation is passed to the model as phoneme tokens. See `docs/BUILDING.md` and `espeakbridge.c` for rationale.
- Native extensions are built with `scikit-build-core` + `cmake`. Treat `script/dev_build` and `setup.py build_ext` as the canonical local-build paths.
- Tests may rely on a built wheel (see `tox.ini` which references a wheel under `dist/`). For quick local tests, run `pytest` directly in the repo after a dev-build.
- Training uses PyTorch Lightning + a custom monotonic alignment C extension in `src/piper/train/vits/monotonic_align`. Always run `./build_monotonic_align.sh` or `python setup.py build_ext --inplace` before training from source.

## Integration points & external dependencies
- `espeak-ng` — phonemizer; project builds a bundled version and uses `espeak_TextToPhonemesWithTerminator`.
- ONNX runtime — voice models are exported to ONNX and loaded by the runtime. GPU support is optional; use `--cuda` with `onnxruntime-gpu`.
- C / CMake — the python package contains native code under `libpiper/` and `espeakbridge.c`.
- Voice artifacts: model `.onnx` files and `.json` config files stored per-voice. Use `python3 -m piper.download_voices` to fetch.

## Safety for code changes and suggested tests
- Small changes touching native code: run `script/dev_build` then `pytest`.
- Training/inference changes: run a minimal synthesis via `script/run` with a small voice to validate end-to-end.

## Helpful examples to reference when making changes
- CLI behavior: `src/piper/__main__.py` and `docs/CLI.md` (examples for `--cuda`, `--input-file`, raw phoneme injection `[[ ... ]]`).
- Training CLI and export: `src/piper/train/__main__.py`, `src/piper/train/export_onnx.py`, and `docs/TRAINING.md`.
- Native bridge: `espeakbridge.c` and `libpiper/src/piper.cpp` (see how Python and C++ interact).

## When to ask for human help
- If a change requires updating the CMake or scikit-build configuration, request a human review before merging.
- If you cannot reproduce a CI failure locally (native build differences across platforms), document steps and request maintainer help.

---
If any section is unclear or you want this shortened/expanded with more examples (e.g., a minimal dev checklist or sample `pytest` invocation), tell me which areas to adjust.
