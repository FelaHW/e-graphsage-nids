# Shuffled-neighborhood hybrid control

Executed on 2026-11-08 for seeds 42, 7, 21, 84, and 123. All 157 pre-existing
source, dataset, preprocessing, model/checkpoint, and historical-result files
retained their SHA256 hashes. New files are isolated under
`results/shuffled_hybrid_multiseed/`; no other experiment was rerun.

## Experimental change and architecture fidelity

The only modeling change is the assignment of valid LOEO context pairs to
target edges. `src/shuffled_hybrid_model.py` defines
`ShuffledHybridEGraphSAGE` as a subclass of the existing hybrid. Its constructor,
learned layers, and forward classifier are inherited unchanged:

- Shared node/context layer: Linear(16,32) + ReLU.
- Classifier input: 8 unchanged direct target features + 32 source context
  values + 32 destination context values.
- Classifier: Linear(72,32), ReLU, Linear(32,16), ReLU, Linear(16,2).
- **3,442 trainable parameters**, identical to the real hybrid.

First compute the ordinary LOEO source/destination embeddings using the
original graph and unchanged aggregation. Then edge i receives both context
embeddings belonging to permutation[i]. Its direct features and label remain
at row i. Source and destination contexts always move together as one pair.
Computing the shared row-wise transformation before indexing preserves the
real context vectors while reassigning their recipients. No edge_index,
initial node feature, normalized edge feature, or label tensor is shuffled.

`src/run_shuffled_hybrid_experiment.py` reuses
`run_hybrid_experiment.run_seed()` itself via temporary runtime settings for
the subclass and graph loader. Source files and historical outputs are never
modified. New control checkpoint/result metadata identify the shuffled variant.

## Exact deterministic assignment

For experiment seed s and split k in {train, validation, test}:

1. Encode the exact string `shuffled-context-v1|{s}|{k}` as UTF-8.
2. Compute SHA256.
3. Interpret the first eight digest bytes as an unsigned big-endian integer,
   then take modulo 2^63.
4. Seed a private CPU `torch.Generator` with that integer.
5. Draw `torch.randperm(E, generator=generator)`; reject a draw if any
   permutation[i] == i. Repeat with the same private generator until there are
   no fixed points.

For example, seed 42's test permutation seed is 8402392730569708536.
All exact derived seeds, permutation hashes, and donor-index lists are saved.
The installed PyTorch version is recorded in `protocol.json`.

The permutation function reads no labels, feature values, graph endpoints, or
epoch number. Its inputs are experiment seed, split identity, and the required
edge count. Assignments remain fixed across all epochs. The private generator
does not advance the global model RNG, preserving each corresponding real
hybrid's seed-based initial weights.

All 15 split/seed assignments are derangements: **zero fixed points**.
An empty graph returns an empty assignment; a one-edge graph is rejected
because a derangement is impossible. Real experiment graphs have 13,995/2,999/
2,999 edges.

## Fixed training and tests

The exact real-hybrid recipe was reused: CPU, one thread, deterministic
algorithms, seed Python/NumPy/PyTorch, Adam at 0.01, CrossEntropyLoss, and 500
full-graph epochs. Adam defaults remain betas (0.9,0.999), epsilon 1e-8, weight
decay 0, AMSGrad off. No schedule, early stopping, architecture changes, or
test-informed tuning were added.

Validation F1 after each epoch selects the checkpoint; strict improvement
keeps the earliest tied epoch. Test graphs load after selection. Each seed has
exactly one test evaluation/forward after restoring the selected weights.
Subgroup analysis reuses those predictions, and real-hybrid subgroup analysis
reads saved historical predictions without forwarding historical models again.

All **30 tests passed before training**. Eight new tests in
`tests/test_shuffled_hybrid.py` cover the requested behaviors:

- Unchanged target features, labels, graph tensors, and classifier input order.
- Actual reassignment of distinct tagged contexts with source/destination pairs
  kept together.
- Determinism, different seed/split assignments, bijectivity, and no fixed points.
- Label-independent assignment/predictions and no consumption of model RNG.
- Identical seed-based initial parameters, 3,442 parameter count, output shape,
  and finite forward/loss/backward values.
- Fixed assignments across forwards, context-availability migration diagnostics,
  and validation-only checkpoint selection with one later test forward.

## All five shuffled-control results

Metrics except validation F1 are test metrics. Confusion-matrix rows are true
labels and columns are predictions, both ordered [benign, attack].

| Seed | Best epoch | Best validation F1 | Accuracy | Precision | Recall | F1 | Confusion matrix | Fixed points train / val / test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 42 | 490 | 0.986446 | 0.982327 | 0.970091 | 0.995330 | 0.982549 | `[[1454,46],[7,1492]]` | 0 / 0 / 0 |
| 7 | 496 | 0.984453 | 0.981661 | 0.970665 | 0.993329 | 0.981866 | `[[1455,45],[10,1489]]` | 0 / 0 / 0 |
| 21 | 445 | 0.984483 | 0.981327 | 0.968811 | 0.994663 | 0.981567 | `[[1452,48],[8,1491]]` | 0 / 0 / 0 |
| 84 | 273 | 0.982084 | 0.981327 | 0.971877 | 0.991328 | 0.981506 | `[[1457,43],[13,1486]]` | 0 / 0 / 0 |
| 123 | 463 | 0.986437 | 0.981994 | 0.971298 | 0.993329 | 0.982190 | `[[1456,44],[10,1489]]` | 0 / 0 / 0 |

## Five-model comparison

Values are mean ± sample standard deviation across the five seeds, ddof=1.
Historical metrics were reused from their existing files.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.976926 ± 0.000683 | 0.962484 ± 0.001875 | 0.992528 ± 0.001446 | 0.977273 ± 0.000656 |
| Original E-GraphSAGE | 0.980527 ± 0.002281 | 0.964659 ± 0.003607 | 0.997598 ± 0.001116 | 0.980850 ± 0.002215 |
| LOEO neighborhood-only | 0.909303 ± 0.002820 | 0.936438 ± 0.005928 | 0.878185 ± 0.001012 | 0.906368 ± 0.002616 |
| Real-neighborhood hybrid | 0.973791 ± 0.004390 | 0.955296 ± 0.007281 | 0.994129 ± 0.001521 | 0.974317 ± 0.004208 |
| Shuffled-neighborhood hybrid | 0.981727 ± 0.000435 | 0.970548 ± 0.001180 | 0.993596 ± 0.001536 | 0.981935 ± 0.000438 |

Primary comparisons, using full-precision mean F1:

- Real hybrid - shuffled hybrid = **-0.007618819**
  (-0.762 percentage points).
- Shuffled hybrid - MLP = **+0.004662396**
  (+0.466 percentage points).
- Real hybrid - MLP = **-0.002956423**
  (-0.296 percentage points).

| Seed | Real - shuffled F1 | Shuffled - MLP F1 | Real - MLP F1 |
| --- | ---: | ---: | ---: |
| 42 | -0.008407815 | 0.005804382 | -0.002603433 |
| 7 | -0.008293545 | 0.005183052 | -0.003110493 |
| 21 | -0.013761942 | 0.004256725 | -0.009505217 |
| 84 | -0.003131108 | 0.004195849 | 0.001064741 |
| 123 | -0.004499685 | 0.003871971 | -0.000627714 |

The real hybrid scored lower in all five paired seeds (0 wins, 5 losses,
0 ties). The sample SD of real-minus-shuffled F1 differences is
0.004143799.

## Interpretation: Case C

**The current real-neighborhood representation is not useful in this
configuration and may introduce noise or instability.** The shuffled control
recorded higher F1 in every paired run. This does not mean random neighborhoods
are genuinely better relationships or contain meaningful intrusion information.
The result provides no positive evidence that the correct topology-specific
assignment helps this hybrid.

The real hybrid also remains below the MLP on average. The shuffled control's
higher observed score is not evidence of useful graph structure: its assignments
were deliberately randomized. The shared architecture can rely on direct flow
features and adapt to unrelated context inputs. This experiment does not
identify a unique optimization/regularization mechanism behind the difference.

The real and shuffled hybrids have matched architecture, initialization seeds,
optimizer, training budget, and selection code. The MLP comparison still uses
a different architecture and training recipe. Conclusions apply only to these
five seeds, the fixed dataset/split, and this representation; no broader
statistical or deployment generalization is claimed.

## Original-recipient subgroup comparison

Group membership is defined by the recipient edge's REAL LOEO neighborhood,
identically to the previous report. Both-empty has 800 edges (622 benign,
178 attack); non-empty has 2,199 edges (878 benign, 1,321 attack).

| Original-recipient group | Model | Edges | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Originally both endpoints empty | real | 800 | 0.987250 ± 0.005108 | 0.978329 ± 0.009439 | 0.964045 ± 0.016185 | 0.971101 ± 0.011652 |
| Originally both endpoints empty | shuffled | 800 | 0.990750 ± 0.002092 | 0.970249 ± 0.004810 | 0.988764 ± 0.007945 | 0.979402 ± 0.004706 |
| Originally at least one endpoint non-empty | real | 2199 | 0.968895 ± 0.006089 | 0.952397 ± 0.008842 | 0.998183 ± 0.001148 | 0.974737 ± 0.004819 |
| Originally at least one endpoint non-empty | shuffled | 2199 | 0.978445 ± 0.001047 | 0.970592 ± 0.001508 | 0.994247 ± 0.002429 | 0.982274 ± 0.000878 |

These are mean ± sample SD across seeds, using saved predictions. Mean
real-minus-shuffled F1 is
-0.008301415
in the original both-empty group and
-0.007537835
in the original non-empty group.

Prediction-disagreement details:

| Seed | Original-recipient group | Predictions differ | Real correct / shuffled wrong | Shuffled correct / real wrong |
| --- | --- | ---: | ---: | ---: |
| 42 | Originally both endpoints empty | 14 | 3 | 11 |
| 42 | Originally at least one endpoint non-empty | 36 | 9 | 27 |
| 7 | Originally both endpoints empty | 11 | 6 | 5 |
| 7 | Originally at least one endpoint non-empty | 45 | 9 | 36 |
| 21 | Originally both endpoints empty | 8 | 3 | 5 |
| 21 | Originally at least one endpoint non-empty | 69 | 14 | 55 |
| 84 | Originally both endpoints empty | 7 | 4 | 3 |
| 84 | Originally at least one endpoint non-empty | 41 | 15 | 26 |
| 123 | Originally both endpoints empty | 12 | 3 | 9 |
| 123 | Originally at least one endpoint non-empty | 32 | 12 | 20 |

### Context-availability ambiguity

Shuffling preserves the total number of empty context pairs but changes which
flows receive them. It does not preserve neighborhood availability per edge.
Test assignment transitions are:

| Seed | Empty -> empty | Empty -> non-empty | Non-empty -> empty | Non-empty -> non-empty | Recipient shares endpoint with donor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 203 | 597 | 597 | 1602 | 23 |
| 7 | 217 | 583 | 583 | 1616 | 17 |
| 21 | 216 | 584 | 584 | 1615 | 19 |
| 84 | 228 | 572 | 572 | 1627 | 19 |
| 123 | 212 | 588 | 588 | 1611 | 19 |

On average, 584.8 of the original 800 empty-context flows
(73.10%) receive a non-empty pair, with the same number
of original non-empty flows receiving an empty pair. Therefore the subgroup
comparison does not isolate an equal-availability topology effect. No
complicated matching or availability-preserving control was introduced.
The two groups also have different class balances.

### Chance overlap and identical vectors

A donor's valid LOEO context excludes the DONOR edge, not necessarily the
RECIPIENT edge. When donor and recipient share an endpoint, the recipient's
features can occur among the donor's remaining neighbors. This happened for
17-23 of 2,999 test recipients (about 0.57%-0.77%), as reported above.
The experiment follows the requested assignment-only shuffle and does not add
a second recipient-exclusion operation.

Zero fixed points means different donor indices. It does not guarantee
different numerical vectors: different empty neighborhoods produce identical
constant context embeddings, and other coincident context means are possible.
These are limitations of the shuffled control, not label use.

## Remaining limitations and artifacts

Graphs and permutations remain split-local. Validation/test inference still
uses other flows in its own split, preserving the batch/transductive-style
limitation rather than strict real-time causal IDS evaluation. Shared
endpoints/related flows across splits and the balanced development sample also
remain limitations.

New artifacts under `results/shuffled_hybrid_multiseed/`:

- `summary.json`: five new run records, five-model mean/std, paired comparisons,
  and original-recipient subgroup/prediction comparisons.
- `protocol.json`: exact seed derivation, unchanged recipe, and protected hashes.
- `seed_<seed>/checkpoint.pt`: selected weights tagged as the shuffled class,
  dimensions, selected epoch/F1, seed, and companion permutation metadata.
- `seed_<seed>/permutations.json`: complete train/validation/test donor indices.
- `seed_<seed>/run_summary.json`: metrics, confusion matrix, fixed points,
  permutation hashes/seeds, migration counts, and overlap diagnostics.
- `seed_<seed>/training_history.json`: all 500 training/validation records.
- `seed_<seed>/test_predictions.json`: predictions/labels plus original and
  assigned empty-context membership.
- `seed_<seed>.log`: reused trainer output with a shuffled-control header and
  final control metadata.

Independent verification checked all 15 exported permutations, regenerated
them exactly, reconciled checkpoint weight hashes, selected epochs, overall and
subgroup prediction metrics, CSV-based topology masks, and all five models'
mean/sample-SD calculations. No additional real test forward pass was used.
All historical files retained their original bytes.

Reproduction in the existing WSL environment:

```bash
cd /mnt/d/Felix_proj/for_git/e-graphsage-nids
source /home/felix/.venvs/e-graphsage-nids/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B src/run_shuffled_hybrid_experiment.py
```

The runner refuses to overwrite an existing output folder. No further
architectures or hyperparameter searches were started.

