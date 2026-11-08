# Leave-one-edge-out E-GraphSAGE ablation

Executed on 2026-11-08 for seeds 42, 7, 21, 84, and 123. The existing MLP and
target-inclusive GNN runs are reused as the historical comparison; neither was
retrained or overwritten. All 100 pre-existing source, dataset, and result files
retained their SHA256 hashes. No split, preprocessing, graph builder, or control
model changes were made.

## Model and aggregation

`src/loeo_egraphsage_model.py` adds `LOEOEGraphSAGE`, inheriting the control's
constructor, node layer, and edge classifier. It has the same 674 parameters,
dimensions, parameter names, and seed-based initial weights. Its forward pass
constructs target-specific endpoint embeddings rather than one embedding per
node.

For each graph, accumulate incident feature sums S_v and incidence counts d_v,
counting each directed flow once at its source and once at its destination.
For target e with features x_e, remove c copies from each endpoint's sum/count,
where c=1 for an ordinary edge and c=2 for a self-loop:

```text
remaining_sum = S_v - c * x_e
remaining_count = d_v - c
neighbor_mean_without_target = remaining_sum / remaining_count
neighbor_mean_without_target = zero when remaining_count == 0
```

Other parallel and reverse-direction edges remain neighbors. Sums and
subtraction use float64 to reduce cancellation error; the resulting means
return to float32 before entering the unchanged learned layers. Empty-context
vectors are explicitly set to zero.

For each edge separately, concatenate its source's eight initial node features
with its eight target-excluded neighborhood means, then apply the existing
Linear(16,32) and ReLU. Do the same for its destination using the same node layer.
Concatenate those two 32-value representations and apply Linear(64,2).
No direct target features enter the classifier. Labels are never used in
aggregation or the forward pass.

## Tests before training

All 16 synthetic tests passed before running the five real seeds. Ten new LOEO
tests in `tests/test_loeo_egraphsage.py` cover:

1. Multiple incident edges and different contexts for different targets.
2. Target-only endpoints producing zero neighborhoods.
3. Parallel flow edges remaining neighbors.
4. Reverse-direction flow edges remaining neighbors.
5. Self-loops removing both incidences, including a lone self-loop.
6. Perturbing target features without changing its own context or prediction;
   the derivative of its own logits with respect to its own features is zero.
7. Predictions unchanged when labels change or are absent.
8. Finite logits, loss, parameter gradients, and feature gradients.
9. Identical layer dimensions, parameter count, and seed initialization versus
   the intact control.
10. Validation-only checkpoint selection, best-weight restoration, delayed test
    loading, and exactly one synthetic test evaluation.

The original six control/runner tests also passed.

## Training protocol

The original GNN recipe was preserved: CPU, one thread, deterministic algorithms,
seeded Python/NumPy/PyTorch initialization, CrossEntropyLoss, Adam at 0.01, and
500 full-graph epochs. No learning-rate, architecture, or training-budget tuning
was done after seeing test results.

Each seed started from a fresh model initialization. Validation F1 was evaluated
after each optimizer step; strict improvement selected a checkpoint and retained
the earliest epoch on ties. The shared `EvaluationTracker` verified selected
weights before the single test evaluation. Test graphs and labels were loaded
only after training and selection finished.

`src/run_loeo_experiment.py` reuses the existing graph loader, evaluator,
prediction checks, graph summaries, and holdout tracker. The ablation training
loop is separate so the control implementation and historical files remain
intact.

## Five-seed LOEO results

All metrics except validation F1 are test metrics. Confusion matrices have true
labels as rows and predictions as columns, ordered [benign, attack].

| Seed | Best epoch | Best validation F1 | Accuracy | Precision | Recall | F1 | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 169 | 0.921811 | 0.905969 | 0.930035 | 0.877919 | 0.903226 | `[[1401,99],[183,1316]]` |
| 7 | 380 | 0.923447 | 0.910637 | 0.938703 | 0.878586 | 0.907650 | `[[1414,86],[182,1317]]` |
| 21 | 420 | 0.922760 | 0.906636 | 0.930742 | 0.878586 | 0.903912 | `[[1402,98],[182,1317]]` |
| 84 | 432 | 0.923341 | 0.912304 | 0.943966 | 0.876584 | 0.909028 | `[[1422,78],[185,1314]]` |
| 123 | 491 | 0.923394 | 0.910970 | 0.938746 | 0.879253 | 0.908026 | `[[1414,86],[181,1318]]` |

## Three-model comparison

Entries are across-seed arithmetic means ± sample standard deviations (ddof=1).
These standard deviations measure training-seed variation on one fixed test
set, not uncertainty from new datasets or splits.

| Metric | MLP | Target-inclusive GNN | LOEO GNN |
| --- | ---: | ---: | ---: |
| F1 | 0.977273093 ± 0.000656206 | 0.980850010 ± 0.002215178 | 0.906368410 ± 0.002615966 |
| Accuracy | 0.976925642 ± 0.000683358 | 0.980526842 ± 0.002281111 | 0.909303101 ± 0.002819529 |
| Precision | 0.962483512 ± 0.001874647 | 0.964658682 ± 0.003606654 | 0.936438424 ± 0.005927892 |
| Recall | 0.992528352 ± 0.001446263 | 0.997598399 ± 0.001116291 | 0.878185457 ± 0.001011725 |

Primary F1 comparisons, using full-precision means:

- LOEO - MLP = **-0.070904683**, or
  **-7.090 percentage points**.
- LOEO - original GNN = **-0.074481600**, or
  **-7.448 percentage points**.

| Seed | LOEO - MLP F1 | LOEO - original GNN F1 |
| --- | ---: | ---: |
| 42 | -0.073518380 | -0.074522885 |
| 7 | -0.069033190 | -0.074645185 |
| 21 | -0.073397947 | -0.076415721 |
| 84 | -0.068282077 | -0.071286943 |
| 123 | -0.070291823 | -0.075537265 |

LOEO scored below both comparison models for all five seeds. The MLP training
recipe still differs from both GNN recipes (30 epochs, shuffled batches of 256,
Adam at 0.001). The target-inclusive and LOEO GNNs use the same dimensions,
initialization scheme, and training recipe.

## Interpretation

**Neighborhood-only information is not sufficient to outperform direct per-flow
features in this experiment.** Removing target-feature reuse reduces mean F1
from about 0.980850 to 0.906368. The original GNN advantage over the MLP does not
survive this ablation.

The strong control results therefore depend on retaining the target flow's own
information through aggregation. This ablation does not establish that
neighboring flows contain no useful signal, or prove that the entire small
control-over-MLP gain came only from target reuse: it removes the target's basic
flow attributes as well as their aggregation path. The results do not isolate
whether neighborhood information helps when direct target features remain
available.

Endpoint sparsity helps explain the ablation's information loss:

| Split | Edges | Source context empty | Destination context empty | Both contexts empty | Both empty fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 13,995 | 12,592 | 3,399 | 3,342 | 23.88% |
| validation | 2,999 | 2,734 | 771 | 766 | 25.54% |
| test | 2,999 | 2,748 | 806 | 800 | 26.68% |

These counts depend on topology, not labels, and are identical across seeds.
In particular, 800 of 2,999 test flows have no remaining context at either
endpoint. With initial node features fixed to ones, the model can only produce
the same learned constant prediction for those flows. For many more edges,
only one endpoint supplies remaining neighborhood information.

Conclusions apply only to this dataset, fixed split, simple architecture, and
training recipe. Validation/test predictions still use other flows in their own
split, so this remains **batch/transductive-style evaluation**, not a strict
real-time causal IDS experiment. Parallel/reverse flows can retain information
related to the target, and endpoints/related traffic can overlap across splits.
No generalization or statistical-significance claim beyond these runs is made.

## Artifacts and verification

All files are separate under `results/loeo_multiseed/`:

- `summary.json`: five LOEO records and full-precision three-model statistics.
- `protocol.json`: recipe, aggregation semantics, and protected-file hashes.
- `seed_<seed>/checkpoint.pt`: selected weights tagged as LOEO, architecture
  dimensions, seed, selected epoch, and validation F1.
- `seed_<seed>/run_summary.json`: test metrics, confusion matrix, graph summaries,
  exclusion protocol, context coverage, and best-weight hash.
- `seed_<seed>/training_history.json`: all 500 training/validation records.
- `seed_<seed>.log`: that seed's training output.

Independent checks reconciled all five confusion matrices with their metrics,
selected epochs/F1 with complete validation histories, finite recorded losses,
and means/sample standard deviations for all three models. No additional real
test forward pass was performed during verification.

Reproduce using the existing WSL environment:

```bash
cd /mnt/d/Felix_proj/for_git/e-graphsage-nids
source /home/felix/.venvs/e-graphsage-nids/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B src/run_loeo_experiment.py
```

The runner refuses to overwrite an existing output folder. A newly authorized
repetition must select a fresh OUTPUT_ROOT before running.

