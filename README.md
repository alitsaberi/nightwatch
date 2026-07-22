# Nightwatch

Sleep recording quality control and review tool powered by [somnio](https://github.com/alitsaberi/somnio).

Nightwatch loads a recording, runs sleep scoring, EEG usability analysis, and edge-window eye-movement detection, then surfaces metrics and plots via a CLI HTML report and a Streamlit UI.

**v1 input format:** ZMax multi-EDF (Hypnodyne layout). Loaders are isolated so other formats can be added later.

## Install

```bash
pip install .
```

Sleep-scoring ONNX weights and sidecar `model.yaml` are **not** bundled. Provide your model path at runtime (CLI `--model` or Streamlit sidebar).

## CLI

```bash
nightwatch run /path/to/recording \
  --format zmax \
  --model /path/to/model.onnx \
  --edge-minutes 30 \
  --usability-model default \
  --output ./report.html
```

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

## Development

```bash
uv sync --group dev
pytest
```

## License

[MIT](LICENSE)
