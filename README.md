# LLM Pricing Simulation Results

Exploratory simulator code and pilot outputs for studying pricing behavior in a repeated differentiated Bertrand duopoly. These artifacts are preliminary and are shared for transparency and further analysis, not as confirmatory evidence.

## Market

The simulator uses aggregate logit demand with two symmetric firms. The pilot calibration uses product quality `a = 2`, outside-option quality `a0 = 0`, differentiation `mu = 0.25`, marginal cost `1`, demand scale `100`, and a maximum price of `4.51`. The corresponding one-shot Nash price is approximately `1.473`; the joint-profit benchmark is approximately `1.925`.

## Pilot Runs

- `results/cost-pilot-01/`: both firms face a cost change at round 30. The arms are `costup` and `costdown`, with `costlo` and `costhi` as no-shock controls.
- `results/shock-pilot-01/`: one LLM trader faces a scripted rival whose price changes at round 30. The arms are `stepcut`, `stepraise`, and `stepnone`.

Each pilot has four episodes per arm, a 60-round horizon, and a 20-round history window. The LLM runs used the P1 prompt and DeepSeek V4.1 Flash. This scale is too small to support strong generalization or causal claims.

Each result folder includes:

- `manifest.json`: run settings, model identifier, and source hashes.
- `summary.csv` and `table.txt`: episode-level summaries and aggregate tables.
- `episodes/`: completed episode records.
- `traces/`: per-round records, including model-call traces where available.

## Code and Offline Check

The simulator is in `market_sim/`; `market_sim/run.py` is the experiment runner. The offline self-test does not make model calls:

```bash
python3 -m market_sim.selftest
```

Running LLM experiments requires a configured inference endpoint and may incur charges. Do not publish API keys or other credentials.

## Limitations and Data Review

These pilots use one model and prompt configuration, short runs, and four episodes per arm. They are exploratory and do not establish a robust treatment effect, a general claim about LLM pricing, or a mechanism. Treat comparisons as descriptive unless supported by a suitable design and analysis.

Before making this repository public, review `manifest.json` and the trace files. They can contain provider/endpoint metadata, prompts, and model-generated text. Remove any information you do not intend to disclose.