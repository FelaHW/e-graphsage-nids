# Hybrid direct-edge + LOEO-neighborhood experiment

Executed on 2026-11-08 for seeds 42, 7, 21, 84, and 123. The fixed dataset splits,
normalization, graph construction, MLP, original GNN, LOEO implementation, and
all historical artifacts remain unchanged. SHA256 checks verified all 126
pre-existing source/data/result files. No shuffled-neighborhood controls were
implemented or run.

## Architecture and rationale

`src/hybrid_egraphsage_model.py` imports and calls the unchanged, tested
`loeo_egraphsage_model.aggregate_without_target()`. The target is excluded at
both endpoints, including both incidences for a self-loop; other parallel and
reverse flows remain valid neighbors.

For each target edge:

1. Concatenate its source's eight constant initial features with its eight
   source LOEO means; apply a shared Linear(16,32) and ReLU.
2. Apply the same node layer and ReLU to the corresponding destination inputs.
3. Concatenate the target's eight normalized flow features, the 32 source
   context values, and the 32 destination context values: 72 values in total.
4. Apply Linear(72,32), ReLU, Linear(32,16), ReLU, Linear(16,2).

The classifier hidden widths (32 and 16) match those of the existing MLP. This
gives direct flow features a nonlinear path even when both contexts are empty;
using only a linear 72->2 classifier would restrict that group's direct path
to a linear model. The architecture was specified before real training.

The hybrid has **3,442 parameters**, versus 850 for the MLP and 674 for each
previous GNN. No labels are read by the model. Direct target features enter only
through the explicit classifier input; the endpoint context excludes them.
Context representations are target-specific, not one final embedding per node.

## Exact training protocol

- Fresh initialization for each of seeds 42, 7, 21, 84, and 123.
- Seed Python, NumPy, and PyTorch; CPU with one thread and deterministic algorithms.
- CrossEntropyLoss and Adam, learning rate 0.01, betas (0.9, 0.999), epsilon
  1e-8, weight decay 0, AMSGrad off, maximize off, and decoupled weight decay
  off. The installed defaults leave foreach/fused unset and capturable/
  differentiable off.
- 500 full-graph epochs; one optimizer update per epoch, no scheduler or early
  stopping.
- Validation F1 after every epoch; save only on strict improvement, retaining
  the earliest checkpoint when F1 ties.
- Restore that checkpoint before loading test features/labels.
- One test evaluation and one test model forward pass per seed.

The GNN optimizer, learning rate, epoch count, and full-graph budget match the
previous GNN runs. The historical MLP still used 30 epochs, shuffled batches of
256, and Adam at 0.001. No previous recipe was changed, and no hybrid settings
were retuned after viewing test results. Architecture size and MLP training
budget remain comparison limitations.

All recorded losses were finite. Training was not uniformly stable: seed 21's
training loss rose from 0.036469 at the selected epoch to 0.094967 at epoch 500.
The selected checkpoint remained the best by validation F1. These observations
limit conclusions to the fixed recipe rather than every possible hybrid setup.

## Tests before training

All 22 synthetic tests passed before training, including six new tests in
`tests/test_hybrid_egraphsage.py`:

- Correct 72-dimensional input, parameter count, output shape, and finite
  forward/loss/backward values.
- Changing the target's features leaves its own context embeddings unchanged,
  while an explicit direct-feature path can respond to those features.
- Labels can change or be absent without changing model predictions.
- Subgroup membership uses topology, not zero-valued means, and removes both
  incidences for a self-loop.
- Subgroup metrics reuse exactly one test forward pass; their confusion matrices
  sum to the overall confusion matrix.
- Validation-only selection, restored weights, delayed test loading, and one
  test evaluation in a synthetic training run.

The previous LOEO tests continue to cover degree-one nodes, parallel/reverse
flows, self-loops, target-feature exclusion, and finite gradients.

## Per-seed hybrid results

Confusion matrices use true labels as rows and predictions as columns, ordered
[benign, attack]. All metrics except validation F1 are test metrics.

| Seed | Best epoch | Best validation F1 | Accuracy | Precision | Recall | F1 | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 492 | 0.975212 | 0.973658 | 0.956298 | 0.992662 | 0.974141 | `[[1432,68],[11,1488]]` |
| 7 | 445 | 0.978460 | 0.972991 | 0.952746 | 0.995330 | 0.973573 | `[[1426,74],[7,1492]]` |
| 21 | 481 | 0.975482 | 0.966989 | 0.944162 | 0.992662 | 0.967805 | `[[1412,88],[11,1488]]` |
| 84 | 355 | 0.978126 | 0.977993 | 0.961365 | 0.995997 | 0.978375 | `[[1440,60],[6,1493]]` |
| 123 | 291 | 0.977139 | 0.977326 | 0.961911 | 0.993996 | 0.977690 | `[[1441,59],[9,1490]]` |

## Four-model five-seed comparison

Entries are arithmetic mean ± sample standard deviation (ddof=1) across seeds.
Historical runs were read from their existing result files, not re-evaluated.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.976926 ± 0.000683 | 0.962484 ± 0.001875 | 0.992528 ± 0.001446 | 0.977273 ± 0.000656 |
| Original target-inclusive GNN | 0.980527 ± 0.002281 | 0.964659 ± 0.003607 | 0.997598 ± 0.001116 | 0.980850 ± 0.002215 |
| LOEO neighborhood-only GNN | 0.909303 ± 0.002820 | 0.936438 ± 0.005928 | 0.878185 ± 0.001012 | 0.906368 ± 0.002616 |
| Hybrid direct-edge + LOEO context | 0.973791 ± 0.004390 | 0.955296 ± 0.007281 | 0.994129 ± 0.001521 | 0.974317 ± 0.004208 |

Primary full-precision F1 differences:

- Hybrid - MLP = **-0.002956423**
  (-0.296 percentage points).
- Hybrid - original GNN = **-0.006533340**
  (-0.653 percentage points).

| Seed | Hybrid - MLP F1 | Hybrid - original GNN F1 |
| --- | ---: | ---: |
| 42 | -0.002603433 | -0.003607938 |
| 7 | -0.003110493 | -0.008722488 |
| 21 | -0.009505217 | -0.012522991 |
| 84 | 0.001064741 | -0.001940124 |
| 123 | -0.000627714 | -0.005873156 |

The hybrid exceeded the MLP in **1/5** paired seeds and
the original GNN in **0/5**. Seed 84 was the
only hybrid-over-MLP win. Averaged over seeds, hybrid recall was slightly higher
than MLP recall, but lower precision outweighed that improvement in F1.

## Hybrid subgroup analysis

Empty means both endpoint incidence counts are zero after excluding the target.
A non-empty neighborhood whose normalized feature mean happens to be zero is
still classified as non-empty. Group membership is based solely on topology.

The existing evaluator's single test forward output is captured with a temporary
forward hook. Overall and subgroup metrics therefore use the exact same selected
checkpoint and predictions; subgroup analysis does not forward the model again.

| Group | Edges | Benign / attack | Accuracy | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Both endpoints empty | 800 | 622 / 178 | 0.987250 ± 0.005108 | 0.978329 ± 0.009439 | 0.964045 ± 0.016185 | 0.971101 ± 0.011652 |
| At least one endpoint non-empty | 2,199 | 878 / 1321 | 0.968895 ± 0.006089 | 0.952397 ± 0.008842 | 0.998183 ± 0.001148 | 0.974737 ± 0.004819 |

Metrics are mean ± sample standard deviation over the five seeds. Counts and
class distributions are identical across seeds. Both-empty flows comprise
26.68% of test traffic.

Per-seed subgroup results:

| Seed | Group | Edges | Accuracy | Precision | Recall | F1 | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | Both endpoints empty | 800 | 0.980000 | 0.965517 | 0.943820 | 0.954545 | `[[616,6],[10,168]]` |
| 42 | At least one endpoint non-empty | 2199 | 0.971351 | 0.955137 | 0.999243 | 0.976693 | `[[816,62],[1,1320]]` |
| 7 | Both endpoints empty | 800 | 0.988750 | 0.982857 | 0.966292 | 0.974504 | `[[619,3],[6,172]]` |
| 7 | At least one endpoint non-empty | 2199 | 0.967258 | 0.948958 | 0.999243 | 0.973451 | `[[807,71],[1,1320]]` |
| 21 | Both endpoints empty | 800 | 0.988750 | 0.988439 | 0.960674 | 0.974359 | `[[620,2],[7,171]]` |
| 21 | At least one endpoint non-empty | 2199 | 0.959072 | 0.938703 | 0.996972 | 0.966960 | `[[792,86],[4,1317]]` |
| 84 | Both endpoints empty | 800 | 0.993750 | 0.983240 | 0.988764 | 0.985994 | `[[619,3],[2,176]]` |
| 84 | At least one endpoint non-empty | 2199 | 0.972260 | 0.958515 | 0.996972 | 0.977365 | `[[821,57],[4,1317]]` |
| 123 | Both endpoints empty | 800 | 0.985000 | 0.971591 | 0.960674 | 0.966102 | `[[617,5],[7,171]]` |
| 123 | At least one endpoint non-empty | 2199 | 0.974534 | 0.960670 | 0.998486 | 0.979213 | `[[824,54],[2,1319]]` |

For the both-empty group, the context embeddings are learned constants shared
by all its flows, so discrimination comes from the direct flow features.
For the other group, at least one endpoint supplies remaining flow context.
The two groups have substantially different class balances (22.25% attack in
both-empty versus about 60.07% attack in non-empty) and can differ in difficulty.
Their absolute F1 values alone do not identify a causal benefit from context.
No historical-model subgroup evaluation was added.

## Interpretation and limits

**The current neighborhood representation does not improve direct flow
classification in this experiment.** The hybrid mean F1 of about 0.974317 is
below both the MLP's 0.977273 and the original GNN's 0.980850, and it is less
consistent across seeds than the MLP. These runs do not provide evidence of
incremental neighbor-context value for this hybrid configuration.

This does not establish that all neighborhood representations are unhelpful.
The hybrid has more parameters, and its full-graph training recipe differs
from the MLP's recipe. The comparison is conditional on the chosen architecture
and settings; five seeds on one test split do not establish broader statistical
or deployment generalization.

Validation/test predictions still aggregate other flows in their own split.
The target is excluded from context, but parallel/reverse related flows can
remain. Shared endpoints/related traffic across splits and the balanced
development sample remain limitations. This is **batch/transductive-style
evaluation**, not a strict real-time causal IDS evaluation. Conclusions apply
only to this dataset and fixed split.

## Artifacts and verification

All new results are separate under `results/hybrid_multiseed/`:

- `summary.json`: all hybrid runs, four-model statistics, paired comparisons,
  and subgroup means/sample standard deviations.
- `protocol.json`: exact fixed recipe, architecture, and protected-file hashes.
- `seed_<seed>/checkpoint.pt`: selected hybrid weights, class, dimensions, epoch,
  validation F1, seed, and direct/context input flags.
- `seed_<seed>/run_summary.json`: global/subgroup metrics, confusion matrices,
  graph summaries, configuration, and evaluation safeguards.
- `seed_<seed>/training_history.json`: every training/validation epoch.
- `seed_<seed>/test_predictions.json`: labels, predicted classes, and empty-context
  membership from the single test evaluation, enabling forward-free auditing.
- `seed_<seed>.log`: per-seed training output.

Independent checks reconstructed subgroup membership from CSV endpoint counts,
recomputed overall and subgroup metrics from saved predictions, reconciled
selected epochs with validation histories, and verified subgroup mean/std
calculations. No additional real test model forward pass was used.

Reproduction with the existing WSL environment:

```bash
cd /mnt/d/Felix_proj/for_git/e-graphsage-nids
source /home/felix/.venvs/e-graphsage-nids/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B src/run_hybrid_experiment.py
```

The runner refuses to replace an existing output folder. A newly authorized
repetition must choose a fresh OUTPUT_ROOT.

