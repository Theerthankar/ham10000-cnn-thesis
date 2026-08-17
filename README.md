# Lightweight vs. heavyweight CNNs for skin lesion classification

Twelve controlled experiments on HAM10000, comparing MobileNetV2, ResNet-50 and
EfficientNet-B3 across three class-imbalance strategies.

**All twelve experiments are complete.** Every result in the write-up comes from running
the notebooks in this repository in order. Nothing is copied in from elsewhere and no
number is typed by hand: the tables and figures are generated from the same JSON files
the training loop writes.

The thesis is in `thesis/` and builds with `latexmk -pdf main.tex`. Regenerate its tables
and figures from the run artefacts with:

```
python3 scripts/build_thesis_tables.py
python3 scripts/make_figures.py
```

Defence preparation notes, including the questions this design is most open to, are in
`DEFENCE_NOTES.md`.

## The design

|  | Augmentation only | Weighted CE | Random oversampling |
|---|---|---|---|
| MobileNetV2 @224 | E1 | E3 | E5 |
| ResNet-50 @224 | E2 | E4 | E6 |
| EfficientNet-B3 @224 | E7 | E8 | E9 |
| EfficientNet-B3 @300 | E10 | E11 | E12 |

E1 through E6 are the confirmatory 2x3 factorial, registered before any of it ran.
E7 through E12 are an ablation added later, and are kept in a separate statistical
family for that reason.

EfficientNet-B3 appears twice on purpose. At 224 it matches the input size of the other
two backbones, so a difference there is attributable to the architecture. At 300, its
native pretraining resolution, it gets its best shot. Running both separates "B3 is
better" from "B3 was handed more pixels", which one run alone cannot do.

Held identical across all twelve: the lesion-level split (seed 42), the augmentation
pipeline, Adam at lr 1e-3, batch size 32, 50 epochs with early stopping at patience 10,
and 16 dataloader workers. Only the architecture, the imbalance strategy and (for B3)
the input size vary.

The worker count belongs in that list rather than in a footnote. PyTorch seeds each
dataloader worker from `base_seed + worker_id`, and the augmentations draw from that
per-worker RNG, so changing the number of workers changes which random flip and rotation
each image receives. It does not bias any condition, but it does have to be the same
across all twelve runs for the comparison to stay controlled.

## Running it

Provision a GPU pod with at least 16 GB of VRAM. The 300px B3 runs are the memory
ceiling, peaking at 10.2 GB for a real training step at batch size 32 (measured, see
`RUNPOD.md`). Batch size stays at 32 because it is part of the fixed protocol, so give
the job a bigger GPU rather than shrinking the batch.

```
pip install -r requirements.txt
jupyter lab
```

Then work through `notebooks/` in numerical order:

| Notebook | What it does |
|---|---|
| `00_environment_check` | GPU, package versions, all three backbones build and run |
| `01_data_and_split` | Fetch HAM10000, create the shared lesion-level split |
| `02_eda` | Class distribution, leakage check, sample images, class weights |
| `03` to `14` | The twelve training runs, one notebook each |
| `20_efficiency_benchmark` | RQ3: parameters, size, CPU latency |
| `21_statistical_tests` | McNemar's, both families, Holm-corrected |
| `22_figures_and_tables` | Every figure and table for the paper |

Run `00` and `01` before anything else. The twelve training notebooks are independent
after that and can go in any order, though starting with `03` (E1) is sensible since it
is the fastest and it creates the split file the others load.

Budget 10 to 16 hours on an A40 for all twelve, and treat that as an estimate rather
than a measurement. The original six took around six hours between them; the three B3
runs at 224 cost roughly what ResNet-50 does, and the three at 300 push about 1.8x the
pixels per epoch on top of that. Time E1 first to calibrate against your actual pod
before leaving the rest unattended.

If the pod drops mid-run, just re-run the training cell. Checkpoints are written every
epoch and training resumes from the last completed one.

## What each run leaves behind

```
runs/E7/
  config.json            resolved config, split hash, GPU name, torch version
  history.csv            per-epoch train and validation curves
  test_metrics.json      accuracy, balanced accuracy, macro F1, AUC, per-class, confusion matrix
  test_predictions.csv   per-image prediction, needed for the paired tests
  checkpoint_best.pt     weights at lowest validation loss (gitignored)
```

The first four are small and belong in version control. They are the evidence that the
reported numbers came from a real run.

## Reproducibility

The split is created once, hashed, and written to
`runs/splits/seed42_lesion_stratified.json`. That file ships with the repository, so
every run loads the same partitions rather than re-deriving them. Every training
notebook prints the hash, and it should always read:

```
f35ac1de18678182
```

with these partition sizes:

| Partition | Images | Lesions |
|---|---|---|
| train | 8,012 | 5,976 |
| val | 979 | 747 |
| test | 1,024 | 747 |

Test-set support per class: nv 690, mel 115, bkl 105, bcc 53, akiec 33, vasc 15, df 13.

If the hash ever differs, something regenerated the split and the runs on either side of
that change are not comparable. Delete `runs/splits/`, re-run notebook `01`, and retrain
anything that carried the old hash.

Seeds are fixed at 42 and cuDNN runs in deterministic mode. Bit-identical results still
require the same GPU model, since kernel selection differs across architectures; a run
on different hardware should land very close but not exactly.

## Things worth knowing before defending this

The three backbones do not carry the same dropout. MobileNetV2 ships with p=0.2,
EfficientNet-B3 with p=0.3, ResNet-50 with none. They are left as published, because
the comparison is meant to be between the networks anyone would actually deploy. It is
still an asymmetry between the arms, and it slightly favours the two that have dropout.

There is no weight decay, no learning-rate schedule and no gradient clipping. Uniform
across all twelve, so no comparison is confounded, but it is a real limitation and it
is likely why ResNet-50's optimisation looks as unstable as it does.

`macro_auc_ovr` can come back null if a class is missing from a batch of predictions.
It never should on the full test set, but the field is nullable rather than crashing.

## Layout

```
src/ham10000/     the pipeline: split, dataset, transforms, models, imbalance, metrics, train
scripts/          CLI entry points used by the analysis notebooks
notebooks/        run these, in order
runs/             outputs, one directory per experiment
figures/          generated by notebook 22
tables/           generated by notebook 22
```

## Data

Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of
multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5,
180161 (2018). Distributed under CC BY-NC. Not redistributed here; notebook `01` fetches
it.
