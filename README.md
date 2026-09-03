# Nightwatch

Sleep recording quality control and review tool powered by [somnio](https://github.com/alitsaberi/somnio).

Nightwatch loads a recording, optionally runs sleep scoring, EEG usability analysis, and edge-window eye movement detection, then surfaces metrics and plots via a CLI HTML report and a Streamlit UI.

**Supported formats:** ZMax multi-EDF directory and single-file EDF (via somnio `read_standard`).

Views are independent: raw traces, spectrogram, sleep scoring, artifacts, and eye movements each have their own channel picks. Empty picks skip that view.

## Install

```bash
pip install .
```

Sleep-scoring ONNX weights and sidecar `model.yaml` are **not** bundled. Provide your model path at runtime when sleep scoring is enabled (CLI `--model` or Streamlit sidebar).

## CLI

Inspect channels:

```bash
nightwatch inspect /path/to/recording.edf --format edf
```

ZMax full default suite (no channel flags):

```bash
nightwatch run /path/to/zmax_recording \
  --format zmax \
  --model /path/to/model.onnx \
  --output ./report.html
```

EDF with selected views:

```bash
nightwatch run /path/to/recording.edf \
  --format edf \
  --raw-channels C3,C4 \
  --spectrogram-channels C3 \
  --sleep-channels C3 \
  --model /path/to/model.onnx \
  --artifact-eeg C3 \
  --movement ACC \
  --eye-left C3 \
  --eye-right C4 \
  --output ./report.html
```

EDF with no view flags writes a recording-info-only report.

```bash
nightwatch --help
```

## Streamlit

```bash
streamlit run src/nightwatch/app.py
```

After installation:

```bash
streamlit run nightwatch.app
```

1. Choose format and path (folder for ZMax, `.edf` file for EDF), then **Load recording**.
2. Recording metadata appears immediately.
3. Enable views independently (raw, spectrogram, sleep, artifacts, eye movements).
4. **Run analysis**.

## Development

```bash
uv sync --group dev
pytest
```

## License

[MIT](LICENSE)
