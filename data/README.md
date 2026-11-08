# Data inputs and generated files

NF-UNSW-NB15 is not committed. Raw inputs and generated processed datasets are
ignored by Git. [`toy_flows.csv`](toy_flows.csv) is intentionally tracked as a tiny
educational fixture used by the [toy examples](../examples/README.md).

## Raw dataset

Supply exactly one file named `NF-UNSW-NB15.csv` beneath:

```text
data/raw/nf_unsw_nb15/
└── NF-UNSW-NB15.csv
```

Nested extraction directories are supported by `src/create_nf_unsw_sample.py`.
The supplied archive used
`extracted/88695f0f620eb568_MOHANAD_A4706/data/NF-UNSW-NB15.csv`
within that raw directory.

The bundled `FurtherInformation.txt` identifies this
[UQ eSpace dataset record](https://espace.library.uq.edu.au/view/UQ:5d0832d)
as the source of dataset metadata and licensing information. This URL is copied
from the supplied metadata.

Checksums for the supplied `NF-UNSW-NB15.csv`:

- **Bundled BagIt manifest SHA-1**, verified against the local CSV:
  `b38025603498ebeb3fb5732681a42ee5b23281db`.
- **Locally computed SHA-256**:
  `26e29152ead98fc27c012155cbcb0842f39ad41d64acf8b274a7e68ea8c71642`.

The SHA-256 is a local reproducibility record, not an upstream-published checksum.

## Generated processed files

The existing preparation scripts generate:

```text
data/processed/
├── nf_unsw_nb15_balanced_20k.csv
└── nf_unsw_nb15_splits/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    ├── train_normalized.csv
    ├── validation_normalized.csv
    ├── test_normalized.csv
    └── normalization_stats.json
```

`data/raw/` and `data/processed/` remain ignored, including all generated splits
and normalization statistics. Follow the [reproduction instructions](../README.md#reproduction)
for preparation order. Keep an existing verified split fixed when comparing
models or training seeds.
