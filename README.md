# Nightwatch

Sleep recording quality control and review tool powered by [somnio](https://github.com/alitsaberi/somnio) and [wu-sleep](https://github.com/alitsaberi/wu-sleep).

Nightwatch loads a recording, optionally runs sleep scoring, EEG usability analysis, and edge-window eye movement detection, then surfaces metrics and plots via a CLI HTML report and a Streamlit UI.

**Supported formats:** ZMax multi-EDF directory and single-file EDF (via somnio `read_standard`).

Views are independent: raw traces, spectrogram, sleep scoring, artifacts, and eye movements each have their own channel picks. Empty picks skip that view.

## Install

```bash
pip install .
```

Sleep scoring uses [WU-Sleep](https://github.com/alitsaberi/wu-sleep). Model weights live in the wu-sleep repo under `model/` (`ft-am`, `ft-noam`, `tfs-am`, `dt`) and are not bundled in the Python wheel. Point Nightwatch at that tree with `--model`, or use an editable wu-sleep install so the local `model/` directory can be discovered automatically.

## CLI

Inspect channels:

```bash
nightwatch inspect /path/to/recording.edf --format edf
```

ZMax full default suite (no channel flags):

```bash
nightwatch run /path/to/zmax_recording \
  --format zmax \
  --model /path/to/wu-sleep/model \
  --sleep-model ft-am \
  --output ./report.html
```

EDF with selected views:

```bash
nightwatch run /path/to/recording.edf \
  --format edf \
  --raw-channels C3,C4 \
  --spectrogram-channels C3 \
  --sleep-channels C3 \
  --model /path/to/wu-sleep/model \
  --sleep-model ft-am \
  --artifact-eeg C3 \
  --movement ACC \
  --eye-left C3 \
  --eye-right C4 \
  --output ./report.html
```

`--model` accepts either a strategy SavedModel directory (e.g. `.../model/ft-am`) or the parent `model/` root. With a parent root, `--sleep-model` selects among `ft-am`, `ft-noam`, `tfs-am`, and `dt`.

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
4. For sleep scoring, pick a WU-Sleep strategy and optionally a model directory.
5. **Run analysis**.

## Development

```bash
uv sync --group dev
pytest
```

## License

[MIT](LICENSE)
