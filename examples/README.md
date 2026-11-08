# Examples and historical scripts

The active research pipeline lives in [`src/`](../src/). These examples preserve
the learning progression and optional tools without mixing them with the current
experiments. See the [project README](../README.md) for environment setup.

## Toy E-GraphSAGE

[`toy_egraphsage/`](toy_egraphsage/) progresses from endpoint mapping and manual
aggregation to a learned model and training:

- `build_toy_graph.py` and `inspect_graph.py`: graph construction and tensor inspection.
- `manual_aggregation.py`, `neighborhood_aggregation.py`, and `normalize_features.py`:
  outgoing/incident aggregation and feature standardization.
- `toy_egraphsage.py` and `train_toy_model.py`: model forward pass and toy training.
- `load_flows_csv.py`, `dataframe_to_graph.py`, and `train_from_csv.py`: the same
  educational pipeline using the tracked [`data/toy_flows.csv`](../data/toy_flows.csv).

With the project environment activated, scripts can run from their new directory:

```bash
cd examples/toy_egraphsage
python -B build_toy_graph.py
python -B manual_aggregation.py
python -B neighborhood_aggregation.py
python -B toy_egraphsage.py
python -B train_toy_model.py
python -B train_from_csv.py
```

These tiny examples illustrate mechanics; their training scores are not research
results. The main test suite still compares the original GNN aggregation against
the toy implementation.

## Optional data inspection

[`data_inspection/`](data_inspection/) contains `inspect_nf_unsw_features.py`
for working-sample feature statistics and `profile_nf_unsw_nb15.py` for full-dataset
label counts. Run them from the repository root when their inputs are available.
The profiler retains the original extracted bundle path under
`data/raw/nf_unsw_nb15/extracted/88695f0f620eb568_MOHANAD_A4706/data/`.
Dataset placement and provenance are documented in [`data/README.md`](../data/README.md).

## Historical archive

[`archive/`](archive/) preserves superseded scripts for provenance:

- `bulid_nf_unsw_graph.py`: the typo-named builder, including its superseded split-size assertions.
- `nf_unsw_to_graph.py`: the earlier whole-sample graph conversion.
- `inspect_nf_ton_iot.py`: historically named `src/inspect_nf_unsw_nb15.py`.
  That filename was incorrect: its documentation and input path target NF-ToN-IoT.

Archived source is frozen, including its original path assumptions, and is outside
the current runnable pipeline. The original layout is preserved in checkpoint
commit `24f4b58`. Historical reports and protocol hashes retain their recorded
paths and values; they have not been rewritten for this layout.
