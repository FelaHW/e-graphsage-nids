# NF-UNSW-NB15 evaluation checkpoint

Clean-split and graph-builder checkpoint verified on 2026-11-08. The first
E-GraphSAGE-style experiment is recorded in [egraphsage_experiment.md](egraphsage_experiment.md).

## Duplicate investigation

An exact duplicate means equality across every CSV column, including endpoints,
flow features, Label, and Attack. An extra copy is a row beyond the first copy
of a distinct complete row.

| Original sample | Rows | Unique rows | Duplicate groups | Extra copies |
| --- | ---: | ---: | ---: | ---: |
| Benign | 10,000 | 9,997 | 2 | 3 |
| Attack | 10,000 | 9,996 | 4 | 4 |
| Total | 20,000 | 19,993 | 6 | 7 |

There were five groups of two identical rows and one group of three.
Before regeneration, train/validation shared two distinct complete-row values
(three matching row pairs), train/test shared one, and validation/test shared
none. Training contained three extra duplicate copies and validation contained
one; test contained none within the split.

The splitter now removes seven extra copies before its existing stratified
70%/15%/15% split with seed 42. The original 20,000-row sample remains unchanged.
All 19,993 unique complete rows appear exactly once across the regenerated
splits. There are no exact duplicates within or between any splits.

| Clean split | Flows | Benign | Attack |
| --- | ---: | ---: | ---: |
| Training | 13,995 | 6,998 | 6,997 |
| Validation | 2,999 | 1,499 | 1,500 |
| Test | 2,999 | 1,500 | 1,499 |

The existing preprocessing script was rerun after splitting. Its five log1p
transforms are unchanged. Means and population standard deviations were fitted
only on the new training split and applied unchanged to validation and test.
An independent check confirmed those statistics and all normalized CSV values.

## Official MLP baseline

Use `mlp_baseline.pt` and `mlp_baseline_metrics.json` for future GNN comparisons.
The MLP source, eight inputs, architecture (8 -> 32 -> 16 -> 2), seed 42,
30 epochs, batch size 256, and learning rate 0.001 were unchanged.
The best checkpoint was selected by validation F1 (epoch 27, rounded F1 0.9806),
then evaluated once on the test split.

| Metric | Official clean split | Historical original split |
| --- | ---: | ---: |
| Accuracy | 0.976325441813938 | 0.9770 |
| Precision | 0.9594594594594594 | 0.9643088903309539 |
| Recall | 0.9946631087391594 | 0.9906666666666667 |
| F1 | 0.9767441860465116 | 0.9773100953633673 |

Official confusion matrix: `[[1437, 63], [8, 1491]]`.
Rows are true labels and columns are predicted labels, ordered [benign, attack].

The old F1 = 0.9773 is historical, not the official comparison target.
Its original artifacts are retained as `mlp_baseline_old_split.pt` and
`mlp_baseline_old_split_metrics.json`. The splits differ, so the score change
alone does not isolate the effect of removing duplicates.

## Verified graph checkpoint

`src/build_nf_unsw_graphs.py` builds three separate PyG Data objects in memory.
Nodes are (IP address, port), and each flow is one directed edge. Initial node
features are ones with eight columns. Labels are 0 = benign and 1 = attack.

Edge feature order: PROTOCOL, L7_PROTO, IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS,
TCP_FLAGS, FLOW_DURATION_MILLISECONDS.

| Split | Nodes | Edges | edge_index | edge_attr | edge_y | Label counts [0, 1] |
| --- | ---: | ---: | --- | --- | --- | --- |
| Training | 16,460 | 13,995 | [2, 13995] | [13995, 8] | [13995] | [6998, 6997] |
| Validation | 3,730 | 2,999 | [2, 2999] | [2999, 8] | [2999] | [1499, 1500] |
| Test | 3,771 | 2,999 | [2, 2999] | [2999, 8] | [2999] | [1500, 1499] |

The builder checks required columns, nonempty input, missing values, binary
labels, tensor shapes and types, finite edge features, label counts, and PyG
node-ID validity. Each split uses its own endpoint mapping.

## Remaining evaluation limits and next step

This remains a random flow split, not an endpoint-disjoint, session-grouped, or
time-separated evaluation. Train/validation share 338 endpoints, train/test
share 356, and validation/test share 255. Related flows can therefore occur
across splits even though exact complete rows do not. Similar feature vectors
from different flows are not removed. The balanced development sample also does
not represent the full dataset's natural class ratio.

For future graph evaluation, keep split graphs separate, fit only on training
data, select using validation F1, and evaluate test only after model selection.
Aggregating unlabeled features within a test graph would assume the whole test
traffic batch is available at prediction time; a real-time evaluation would
need to restrict context to traffic already observed.

Reusable toy code is in `src/toy_egraphsage.py`: `aggregate_connected_edges()`
averages incident edge features and `ToyEGraphSAGE.forward()` learns endpoint
embeddings before classifying edges. The toy code was not modified.

## Reproduction in the existing WSL environment

All four stages executed successfully in Ubuntu using the existing virtual
environment. The MLP ran on CPU with one OpenMP/MKL thread. A CUDA driver
compatibility warning did not prevent CPU training or graph verification.
These commands regenerate and overwrite the current artifacts:

```bash
cd /mnt/d/Felix_proj/for_git/e-graphsage-nids
source /home/felix/.venvs/e-graphsage-nids/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B src/split_nf_unsw_sample.py
python -B src/preprocess_nf_unsw_splits.py
python -B src/train_mlpbaseline.py
python -B src/build_nf_unsw_graphs.py
```
