# Fixed-split multi-seed comparison

Executed on 2026-11-08 with seeds 42, 7, 21, 84, and 123. All ten runs trained
from a fresh initialization. Dataset splits, preprocessing, graph construction,
model architectures, and all historical artifacts remained byte-for-byte
unchanged. No ablation was implemented or run.

## Fair-comparison check

The existing training procedures were inspected before running and preserved:

| Setting | MLP | E-GraphSAGE |
| --- | --- | --- |
| Architecture | 8 -> 32 -> 16 -> 2 | incident-edge mean, 16 -> 32 node layer, 64 -> 2 edge classifier |
| Trainable parameters | 850 | 674 |
| Epochs | 30 | 500 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.01 |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Training batching | shuffled batches of 256 | one full graph per step |
| Optimizer steps per epoch | 55 | 1 |
| Total optimizer steps | 1,650 | 500 |
| Checkpoint selection | best validation F1 each epoch | best validation F1 each epoch |
| F1 ties | keep earliest epoch | keep earliest epoch |
| Test evaluation | once after restoring the selected weights | once after restoring the selected weights |

Both models ran on CPU with one thread and deterministic algorithms. Python,
NumPy, and PyTorch were seeded for each run; the original trainers also perform
their normal seeding. MLP randomness includes initialization and training-batch
shuffling; the full-graph GNN has no minibatch shuffling.

The MLP's native script loads the fixed test tensors before training but never
uses them for optimization or selection. The wrapper checks that test evaluation
happens only after all training epochs and after restoration of the selected
weights. The GNN also delays loading its test graph until selection is finished.

This compares the existing training recipes, not architectures under an equal
training budget: epoch counts, learning rates, update counts, and example
exposures differ. Neither recipe was retuned using these test results.

## Per-seed results

All metrics below are test metrics except validation F1. Confusion matrices are
[[true benign/predicted benign, true benign/predicted attack],
[true attack/predicted benign, true attack/predicted attack]].

| Model | Seed | Best epoch | Best validation F1 | Accuracy | Precision | Recall | F1 | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MLP | 42 | 27 | 0.980573 | 0.976325 | 0.959459 | 0.994663 | 0.976744 | `[[1437,63],[8,1491]]` |
| E-GraphSAGE | 42 | 438 | 0.985851 | 0.977326 | 0.959538 | 0.996664 | 0.977749 | `[[1437,63],[5,1494]]` |
| MLP | 7 | 26 | 0.980224 | 0.976325 | 0.961837 | 0.991995 | 0.976683 | `[[1441,59],[12,1487]]` |
| E-GraphSAGE | 7 | 436 | 0.986799 | 0.981994 | 0.965828 | 0.999333 | 0.982295 | `[[1447,53],[1,1498]]` |
| MLP | 21 | 28 | 0.983152 | 0.976992 | 0.963684 | 0.991328 | 0.977310 | `[[1444,56],[13,1486]]` |
| E-GraphSAGE | 21 | 325 | 0.983284 | 0.979993 | 0.963894 | 0.997332 | 0.980328 | `[[1444,56],[4,1495]]` |
| MLP | 84 | 29 | 0.979215 | 0.976992 | 0.963684 | 0.991328 | 0.977310 | `[[1444,56],[13,1486]]` |
| E-GraphSAGE | 84 | 489 | 0.983574 | 0.979993 | 0.964493 | 0.996664 | 0.980315 | `[[1445,55],[5,1494]]` |
| MLP | 123 | 25 | 0.981854 | 0.977993 | 0.963754 | 0.993329 | 0.978318 | `[[1444,56],[10,1489]]` |
| E-GraphSAGE | 123 | 337 | 0.990753 | 0.983328 | 0.969540 | 0.997999 | 0.983563 | `[[1453,47],[3,1496]]` |

## Mean and sample standard deviation

Each mean is the arithmetic mean across the five independently initialized
runs. F1 standard deviations are sample standard deviations with ddof=1.
These are standard deviations across runs, not standard errors or confidence
intervals.

| Metric | MLP | E-GraphSAGE |
| --- | ---: | ---: |
| Mean test F1 | 0.977273093 | 0.980850010 |
| Sample std of test F1 | 0.000656206 | 0.002215178 |
| Mean accuracy | 0.976925642 | 0.980526842 |
| Mean precision | 0.962483512 | 0.964658682 |
| Mean recall | 0.992528352 | 0.997598399 |

Mean E-GraphSAGE F1 - mean MLP F1 = **+0.003576916**,
approximately +0.358 percentage points.

| Seed | MLP F1 | E-GraphSAGE F1 | Paired difference |
| --- | ---: | ---: | ---: |
| 42 | 0.976744 | 0.977749 | +0.001005 |
| 7 | 0.976683 | 0.982295 | +0.005612 |
| 21 | 0.977310 | 0.980328 | +0.003018 |
| 84 | 0.977310 | 0.980315 | +0.003005 |
| 123 | 0.978318 | 0.983563 | +0.005245 |

The GNN scored higher for all five seeds. The sample standard deviation of
paired F1 differences is 0.001882997.
The seed-42 paired difference uses the full-precision MLP value
0.9767441860465116; the original +0.001049 comparison used the rounded 0.9767.

## Interpretation

The original single-seed difference of about 0.001049 is smaller than the
GNN's observed between-seed F1 standard deviation of about 0.002215, so that
single comparison alone was weak evidence. Across the requested seeds,
however, every paired difference is positive, with a mean of about 0.003577.
These repeated results support a higher test F1 for the current GNN training
recipe on this fixed split, while also showing greater GNN seed variability.

This is not proof that graph structure itself caused the improvement. The
recipes and architectures differ, target edges still contribute their own
features, and no neighbor-removal ablation has been run. Five seeds on one
unchanged test set measure training randomness conditional on this dataset;
they do not measure uncertainty from new splits, new traffic, or deployment.
No statistical-significance claim is made.

The original balanced-sample, shared-endpoint, and batch/transductive-style
limitations continue to apply. Every test set was evaluated once per requested
model/seed. The ten test results are compared only after their checkpoints were
selected using validation F1; no preferred seed was selected for deployment.

## Proposed leave-one-edge-out aggregation (design only)

Keep the same eight normalized features, initial node features, node linear
layer, ReLU, edge classifier, and parameter count. Change only how the endpoint
context is constructed for the edge currently being classified.

For node v, let S_v be the sum of incident edge-feature vectors and d_v the
incident-edge count, computed using the current source-plus-destination
semantics. For target edge e with feature vector x_e, let c(v,e) be its incidence
multiplicity: 1 at each endpoint of an ordinary edge and 2 at the endpoint of a
self-loop. Construct the target-specific mean as:

```text
remaining_count = d_v - c(v,e)
mean_without_e = (S_v - c(v,e) * x_e) / remaining_count
mean_without_e = zero vector when remaining_count == 0
```

For e = (source, destination):

1. Obtain the source mean with e excluded and the destination mean with e
   excluded. Remove the target from both endpoint contexts, including both
   incidences for a self-loop.
2. Concatenate each context with that endpoint's existing initial node features.
3. Apply the same shared node linear layer and ReLU to those two vectors.
4. Concatenate the resulting source/destination embeddings and use the existing
   two-logit classifier. Do not add a shortcut containing the target features.

Exclusion must occur before the learned node transformation and ReLU, not by
subtracting the target features from an already-computed node embedding. A
single shared embedding per node is insufficient: its context now depends on
which target edge is being classified. Sums and counts can be computed once per
graph, followed by vectorized per-target subtraction, preserving approximately
O(number_of_edges * number_of_features) aggregation work.

Only the target edge instance is excluded. Other parallel or reverse flow edges
remain valid neighbors. Degree-one endpoints receive a zero neighborhood vector;
if both endpoints have no remaining neighbors, prediction can use only the
constant initial node features and learned biases. Report the fractions with
zero remaining context at one or both endpoints, and consider performance by
neighbor count when interpreting the eventual experiment.

When authorized, retrain this ablation from scratch on the same fixed graphs
and five seeds, apply exclusion consistently during training/validation/test,
select using validation F1, and evaluate test once per seed. Compare each seed
against the corresponding target-inclusive GNN run. Do not merely change the
forward pass of the already-trained inclusive model at test time.

This would test sensitivity to direct target-feature recovery. It would still
use other same-split flows, shared endpoints, and batch context, so it would
remain a batch experiment rather than solve strict real-time causal evaluation.

## Artifacts and reproducibility

- `summary.json`: all ten run records and full-precision aggregate statistics.
- `protocol.json`: seeds, training procedures, and protected source/input hashes.
- `mlp/seed_<seed>/` and `egraphsage/seed_<seed>/`: separate selected checkpoints,
  native trainer metrics, validation histories, run summaries, and training logs.
- GNN folders additionally retain their native full training histories.

The original GNN trainer's per-run log/native metrics retain its comparison
against the historical single-seed MLP reference 0.9767. Those lines are not
the multi-seed comparison; use this report and `summary.json` for paired results.

`src/run_multiseed_experiment.py` reuses both original `main()` functions.
Temporary module settings redirect files and seeds; evaluator wrappers verify
restored weight hashes without another model forward pass. All six synthetic
tests passed before running. Independent checks reconciled each confusion
matrix, each selected epoch/F1, and both sample standard deviations. All 40
pre-existing source/data/result files retained their original hashes.

Run with the existing WSL environment:

```bash
cd /mnt/d/Felix_proj/for_git/e-graphsage-nids
source /home/felix/.venvs/e-graphsage-nids/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B src/run_multiseed_experiment.py
```

The runner refuses to overwrite an existing `results/multiseed` folder. To
perform a separately authorized repetition, choose a fresh OUTPUT_ROOT before
starting; do not delete or replace historical results.

