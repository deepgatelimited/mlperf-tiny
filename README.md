# MLPerf Tiny v1.4: DeepGate submission

Zephyr-based DUT firmware for the **MLPerf Tiny v1.4** Inference benchmarks. A single Zephyr source tree builds for every supported board; the per-`(board, bench)` `libmodel.a` is produced separately by the DeepGate compiler (https://bitweaver.deepgate.ai).

## Supported boards

| Board | Zephyr target | Console UART | Probe |
|---|---|---|---|
| ST NUCLEO-H7A3ZI-Q | `nucleo_h7a3zi_q` | USART3 @ 921600 | ST-Link V3 |
| Silicon Labs xG24 Explorer Kit (EFR32MG24) | `xg24_ek2703a` | USART0 @ 115200 | J-Link OB |

## Prerequisites

The setup below has only been tested on macOS (Apple Silicon).

| Component | Version | Notes |
|---|---|---|
| **CMake, ninja, dtc, gperf** | CMake >= 3.20 | `brew install cmake ninja dtc gperf` |
| **libusb** | system | Needed by the MLCommons runner's `pyusb` backend. `brew install libusb`. |
| **ffmpeg** | system | Audio backend for `librosa` (AD01 WAV decoding). `brew install ffmpeg`. |
| **uv** | latest | Python venv + package manager. `brew install uv`. |
| **A clone of `mlcommons/tiny`** | master >= `5dae3296` | `git clone https://github.com/mlcommons/tiny` |

## Build and flash

```bash
git clone https://github.com/deepgatelimited/mlperf-tiny && cd mlperf-tiny

# Python venv (installs west + runner-side deps from pyproject.toml)
uv sync

# Zephyr workspace + SDK (per Zephyr's getting-started flow, with our manifest)
uv run west init -l manifest
uv run west update
uv run west packages pip | xargs uv pip install
uv run west sdk install -t arm-zephyr-eabi
uv run west zephyr-export

# Build + flash one (board, bench) pair. Group by board so you only swap
# which DUT is plugged in once.

# H7A3 / Image Classification (ic01)
uv run west build -b nucleo_h7a3zi_q firmware/ic01  && uv run west flash
# H7A3 / Image Classification (ic02, larger ResNet)
uv run west build -b nucleo_h7a3zi_q firmware/ic02  && uv run west flash
# H7A3 / Keyword Spotting (kws01)
uv run west build -b nucleo_h7a3zi_q firmware/kws01 && uv run west flash
# H7A3 / Visual Wake Words (vww01)
uv run west build -b nucleo_h7a3zi_q firmware/vww01 && uv run west flash
# H7A3 / Anomaly Detection (ad01)
uv run west build -b nucleo_h7a3zi_q firmware/ad01  && uv run west flash

# xG24 / Image Classification (ic01)
uv run west build -b xg24_ek2703a firmware/ic01  && uv run west flash
# xG24 / Image Classification (ic02, larger ResNet)
uv run west build -b xg24_ek2703a firmware/ic02  && uv run west flash
# xG24 / Keyword Spotting (kws01)
uv run west build -b xg24_ek2703a firmware/kws01 && uv run west flash
# xG24 / Visual Wake Words (vww01)
uv run west build -b xg24_ek2703a firmware/vww01 && uv run west flash
# xG24 / Anomaly Detection (ad01)
uv run west build -b xg24_ek2703a firmware/ad01  && uv run west flash
```

This repo ships precompiled `firmware/<bench>/model/<board>/{lib/libmodel.a, include/model.h}` for all 10 `(board, bench)` pairs. To regenerate from `models/closed/<bench>/schema.json`, upload it to <https://bitweaver.deepgate.ai>, pick the target board, and drop the returned files in `firmware/<bench>/model/<board>/{lib,include}/`.

## Run a benchmark

Fetch the evaluation datasets (the venv from the previous section is required):

```bash
export TINY=/path/to/your/mlcommons-tiny-clone
scripts/fetch_datasets.sh
```

`scripts/fetch_datasets.sh` defers to upstream `mlcommons/tiny`'s per-bench helpers; deviations (CIFAR-10 download for IC01, PIL transcode for VWW01 since upstream has no bin generator, `curl -L` for AD01 to handle Zenodo redirects) are inlined.

Drive `$TINY/benchmark/runner/main.py` against the flashed DUT. `runner/devices.yaml` lists both DUTs; plug in only the board you're targeting.

```bash
export REPO=$(pwd)
export DATASETS=$REPO/runner/tests/data
cd $TINY/benchmark/runner

# Accuracy (same script for both boards)
python main.py \
    --device_list=$REPO/runner/devices.yaml \
    --dataset_path=$DATASETS \
    --test_script=tests_accuracy.yaml --mode=a

# Performance (H7A3)
python main.py \
    --device_list=$REPO/runner/devices.yaml \
    --dataset_path=$DATASETS \
    --test_script=$REPO/runner/tests_performance_h7a3.yaml --mode=p

# Performance (xG24)
python main.py \
    --device_list=$REPO/runner/devices.yaml \
    --dataset_path=$DATASETS \
    --test_script=$REPO/runner/tests_performance_xg24.yaml --mode=p
```

Each run writes `log.txt`, `results.txt`, and `results.json` under `$TINY/benchmark/runner/sessions/<timestamp>/`.
