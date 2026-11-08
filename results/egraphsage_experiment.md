# First E-GraphSAGE-style experiment

Executed on 2026-11-08 in the existing Ubuntu/WSL environment. The MLP baseline,
dataset splits, normalized CSVs, normalization statistics, verified graph
builder, and toy model/training files were not changed. SHA256 checks confirmed
that the existing inputs and baseline artifacts retained their original bytes.

## Architecture

`src/egraphsage_model.py` follows the pipeline in `src/toy_egraphsage.py`:

1. Mean the eight features of all incoming and outgoing edges at each node.
2. Concatenate those eight means with the eight initial node features (ones).
3. Apply Linear(16, 32) and ReLU to create each endpoint embedding.
4. Concatenate the source and destination embeddings for each directed edge.
5. Apply Linear(64, 2) to produce benign/attack logits.

The model has **674 trainable parameters**. Tensor `index_add_` computes the
same incident-edge sums/counts as the toy Python loop. Parallel edges remain
distinct. Self-loops contribute at both endpoints, matching the toy semantics.
Neither aggregation nor the forward pass reads `edge_y`.

This is a simple one-layer educational E-GraphSAGE-style model, not the complete
paper architecture. Graph loading reuses `dataframe_to_graph()` from
`src/build_nf_unsw_graphs.py`; graph construction is not duplicated.

## Fixed training protocol

- CPU, one thread, seed 42 for Python/NumPy/PyTorch, deterministic algorithms.
- CrossEntropyLoss and Adam with learning rate 0.01.
- 500 epochs, one full-training-graph optimizer step per epoch.
- Validation F1 evaluated each epoch with gradients disabled.
- Save on strict validation F1 improvement; retain the earliest tied epoch.
- Load the selected checkpoint before loading the test graph.
- Evaluate the real test graph exactly once, after training and selection.

Training labels were [6998 benign, 6997 attack]. Logits were checked for shape
[number_of_edges, 2], labels for [number_of_edges], and all logits/losses for
finite values. Training loss decreased from 0.789809 to 0.045516 over 500 epochs.

**Best epoch: 438. Best validation F1: 0.9858506087528792.**
Validation loss at that epoch was 0.057078. Later epochs reached the same best
F1, so they did not replace the earlier checkpoint.

## Single test evaluation

| Metric | E-GraphSAGE-style model |
| --- | ---: |
| Accuracy | 0.9773257752584195 |
| Precision | 0.9595375722543352 |
| Recall | 0.9966644429619747 |
| F1 | 0.9777486910994765 |

Confusion matrix: `[[1437, 63], [5, 1494]]`. Rows are true labels and columns
are predictions, both ordered [benign, attack].

Requested comparison against the official rounded MLP F1:

```text
MLP baseline: F1 = 0.9767
E-GraphSAGE: F1 = 0.9777486910994765
Difference: +0.001048691099476473
```

This is about +0.105 percentage points of F1. Compared with the MLP confusion
matrix, this run recovers three additional attacks with the same 63 false
positives. The result is slightly better on this fixed test split. A single
seed and comparison between different architectures/training schedules do not
establish that neighbor information caused the improvement. A controlled
ablation and repeated runs would be needed to isolate that contribution.
No hyperparameters were changed after seeing the test result.

## Batch evaluation assumptions and limitations

The target edge contributes **its own edge features** to both endpoint means.
Validation and test predictions also use features of **other edges from the
same split**, without using those edges' labels. Graphs are never merged across
splits, and preprocessing continues to use the previously fitted training-only
statistics. Test labels are unavailable to the training/selection loop.

This is a **batch/transductive-style graph evaluation**, not a strict real-time
causal IDS evaluation. A prediction may use flows from its batch that would not
yet have occurred in a deployment setting. The architecture can also receive
the target flow's own signal through aggregation, so the experiment does not
isolate pure neighbor-only information.

The existing random flow split allows shared endpoints and related sessions
across splits. Exact complete-row duplicates have been removed, but evaluation
is not endpoint-disjoint or time-separated. The balanced development sample
does not reproduce the full dataset's natural class ratio.

## Artifacts and verification

- `egraphsage.pt`: selected weights, model dimensions, best epoch and F1.
- `egraphsage_metrics.json`: test metrics, configuration, graph summaries,
  input hashes, comparison, and evaluation assumptions.
- `egraphsage_training_history.json`: all 500 training/validation records.
- `tests/test_egraphsage.py`: three synthetic tests checking aggregation against
  expected means and the toy implementation, label-independent predictions and
  finite gradients, and validation-only selection with one later test call.

All three tests passed before the real experiment. The selected epoch/F1 were
also independently checked against the saved training history. No additional
real test forward pass was performed for verification.

To run the tests in the existing WSL environment:

```bash
cd /mnt/d/Felix_proj/for_git/e-graphsage-nids
source /home/felix/.venvs/e-graphsage-nids/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest discover -s tests -p test_egraphsage.py -v
```

To reproduce the experiment (overwrites the E-GraphSAGE artifacts and evaluates
test once for that new run):

```bash
python -B src/train_egraphsage.py
```

Neither command regenerates splits, fits preprocessing, or retrains the MLP.
