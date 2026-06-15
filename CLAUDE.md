You are acting as a senior machine learning engineer, EEG/BCI researcher,
reproducibility engineer, and scientific Python developer.

Build a complete, reproducible research project for comparing a traditional
motor-imagery EEG pipeline against two deep-learning models on the Cho et al.
2017 GigaDB motor-imagery dataset.

Do not merely give me code snippets or an implementation plan. Create the actual
project structure, source files, configuration files, Jupyter notebooks, tests,
README, dataset download scripts, preprocessing pipeline, split manifests, and
result directories.

IMPORTANT EXECUTION POLICY
==========================

1. Download the dataset into the project if network access is available.
2. You may execute lightweight setup, download, integrity-check, and shape-check
   scripts.
3. DO NOT execute the training notebooks.
4. DO NOT run CSP, EEGNet, or ATCNet training.
5. DO NOT fabricate metrics, plots, checkpoints, predictions, or result files.
6. The experiment notebooks will be executed manually by me later.
7. Result directories must be created in advance, but generated result files
   must only appear when the relevant notebook is run.
8. Add README or .gitkeep files to empty result directories when necessary.
9. At the end, report:
   - which files and folders were created,
   - whether all dataset files were downloaded successfully,
   - which lightweight validations were executed,
   - which notebooks were intentionally not run,
   - any unresolved dataset-structure assumptions.

PROJECT TITLE
=============

Cho2017 Motor Imagery EEG Benchmark:
CSP + LDA vs EEGNet vs ATCNet

GENERAL RESEARCH QUESTION
=========================

Can compact end-to-end deep-learning models classify left-hand versus right-hand
motor imagery more accurately and robustly than a traditional CSP + LDA
pipeline, when all methods are evaluated using:

- the same participants,
- the same accepted trials,
- the same train/validation/test split manifest,
- the same primary EEG time interval,
- comparable preprocessing,
- untouched test sets,
- subject-wise reporting,
- reproducible random seeds?

The project must not assume that deep learning will necessarily outperform the
traditional method. CSP + LDA is a strong BCI baseline. The purpose is to
measure the trade-off between:

- predictive accuracy,
- inter-subject variability,
- training cost,
- inference latency,
- model size,
- feature engineering,
- interpretability.

DATASET
=======

Primary dataset page:
https://gigadb.org/dataset/100295

Paper:
https://doi.org/10.1093/gigascience/gix034

Data DOI:
https://doi.org/10.5524/100295

MOABB dataset class:
moabb.datasets.Cho2017

MOABB source currently downloads subject files using URLs in the form:

https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/s01.mat
...
https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/s52.mat

Expected dataset properties, which must be verified programmatically instead of
blindly trusted:

- 52 healthy participants.
- Subject IDs s01 through s52.
- 64 EEG channels.
- 4 simultaneous EMG channels.
- 68 signal channels in total.
- Sampling frequency: 512 Hz.
- Two motor-imagery classes:
  - left hand
  - right hand
- One recording session per participant.
- Five or six motor-imagery runs.
- 100 or 120 trials per class.
- Three-second motor-imagery instruction interval.
- Original MATLAB structure fields may include:
  - imagery_left
  - imagery_right
  - imagery_event
  - n_imagery_trials
  - movement_left
  - movement_right
  - movement_event
  - n_movement_trials
  - rest
  - noise
  - frame
  - srate
  - senloc
  - psenloc
  - subject
  - comment
  - bad_trial_indices

The first 64 channels are EEG and the following 4 channels are EMG.

Subjects s29 and s34 were excluded in the original paper because more than 90%
of their motor-imagery trials were correlated with EMG activity. The default
main experiment must exclude these two subjects, but EDA must still inspect and
report them.

Do not remove any of the other low-performing participants. The main benchmark
must use all 50 remaining subjects, including participants with weak
discriminative EEG patterns. Do not select only the 38 high-performing subjects.

SCIENTIFIC OBJECTIVES
=====================

Experiment 1:
Traditional CSP + LDA

Purpose:
Measure performance of a conventional, interpretable, low-compute BCI pipeline
based on manually selected frequency bands and spatial variance features.

Experiment 2:
EEGNet

Purpose:
Measure whether a compact EEG-specific CNN can learn temporal and spatial EEG
features end-to-end with limited parameter count.

Experiment 3:
ATCNet

Purpose:
Measure whether adding attention and temporal convolution to a convolutional
EEG front-end improves classification relative to EEGNet and CSP + LDA.

The final comparison must answer:

1. Which model has the highest mean subject-wise test accuracy?
2. Which model has the highest median accuracy?
3. Which model is most consistent across subjects?
4. How many participants achieve accuracy above 50%, 60%, 70%, and 80%?
5. Which participants remain difficult for all three models?
6. Does ATCNet provide a statistically supported improvement over EEGNet?
7. Does either deep model provide a statistically supported improvement over
   CSP + LDA?
8. What are the costs in parameters, training time, and inference latency?
9. Are model improvements broad across participants or driven by a small subset?
10. Do results support the claim that AI is superior, or only that it offers
    another accuracy/complexity trade-off?

NON-NEGOTIABLE DATA-LEAKAGE RULES
=================================

1. Create one shared split manifest before implementing model training.
2. Every experiment must use the exact same train, validation, and test trial
   assignments.
3. Test data must never be used for:
   - CSP fitting,
   - filter selection,
   - normalization statistics,
   - hyperparameter selection,
   - early stopping,
   - threshold selection,
   - model selection,
   - data-quality threshold selection.
4. Validation data may be used for:
   - hyperparameter selection,
   - early stopping,
   - learning-rate scheduling,
   - selection of the best epoch.
5. Fit all learned preprocessing only on the training set.
6. For neural networks, calculate normalization statistics from training trials
   only and apply the frozen statistics to validation and test trials.
7. Do not create overlapping windows from the same original trial across
   different splits.
8. Do not randomly split individual samples from the same trial.
9. Do not silently remove failed subjects or failed model runs from aggregates.
10. All exclusions and failures must be written to machine-readable reports.

PROJECT STRUCTURE
=================

Create a project with at least the following structure:

cho2017-motor-imagery-benchmark/
│
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Makefile
│
├── configs/
│   ├── base.yaml
│   ├── csp_lda.yaml
│   ├── eegnet.yaml
│   └── atcnet.yaml
│
├── data/
│   ├── raw/
│   │   └── cho2017/
│   │       ├── s01.mat
│   │       ├── ...
│   │       └── s52.mat
│   ├── interim/
│   ├── processed/
│   │   └── common/
│   ├── manifests/
│   │   ├── file_manifest.csv
│   │   ├── trial_manifest.csv
│   │   ├── split_manifest.csv
│   │   └── excluded_trials.csv
│   └── metadata/
│       ├── channel_names.json
│       ├── dataset_summary.json
│       └── preprocessing_summary.json
│
├── notebooks/
│   ├── 00_exploratory_data_analysis.ipynb
│   ├── 01_traditional_csp_lda.ipynb
│   ├── 02_eegnet.ipynb
│   ├── 03_atcnet.ipynb
│   └── 04_final_model_comparison.ipynb
│
├── src/
│   └── cho2017_benchmark/
│       ├── __init__.py
│       ├── paths.py
│       ├── config.py
│       ├── reproducibility.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py
│       │   ├── inspect_mat.py
│       │   ├── loader.py
│       │   ├── epoching.py
│       │   ├── artifacts.py
│       │   ├── preprocessing.py
│       │   ├── prepare.py
│       │   ├── splits.py
│       │   └── datasets.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── eegnet.py
│       │   ├── atcnet.py
│       │   └── factory.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── callbacks.py
│       │   └── checkpoints.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── predictions.py
│       │   ├── latency.py
│       │   ├── statistics.py
│       │   └── plots.py
│       │
│       └── reporting/
│           ├── __init__.py
│           ├── result_writer.py
│           └── comparison.py
│
├── results/
│   ├── eda/
│   │   ├── figures/
│   │   ├── tables/
│   │   └── reports/
│   │
│   ├── csp_lda/
│   │   ├── configs/
│   │   ├── figures/
│   │   ├── tables/
│   │   ├── predictions/
│   │   ├── models/
│   │   ├── logs/
│   │   └── metrics/
│   │
│   ├── eegnet/
│   │   ├── configs/
│   │   ├── figures/
│   │   ├── tables/
│   │   ├── predictions/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── histories/
│   │   └── metrics/
│   │
│   ├── atcnet/
│   │   ├── configs/
│   │   ├── figures/
│   │   ├── tables/
│   │   ├── predictions/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── histories/
│   │   └── metrics/
│   │
│   └── comparison/
│       ├── figures/
│       ├── tables/
│       └── reports/
│
├── scripts/
│   ├── download_dataset.py
│   ├── validate_dataset.py
│   ├── prepare_dataset.py
│   ├── create_splits.py
│   └── print_project_status.py
│
└── tests/
    ├── test_download_manifest.py
    ├── test_mat_loader.py
    ├── test_epoching.py
    ├── test_bad_trial_mapping.py
    ├── test_split_leakage.py
    ├── test_preprocessing.py
    ├── test_eegnet_shape.py
    ├── test_atcnet_shape.py
    └── test_metric_outputs.py

You may add files when they improve maintainability. Do not collapse the whole
project into notebooks.

LANGUAGE AND STYLE
==================

- Use English for:
  - source-code identifiers,
  - filenames,
  - classes,
  - functions,
  - variable names,
  - configuration keys,
  - docstrings.
- Use clear Turkish explanations in:
  - README,
  - notebook Markdown cells,
  - scientific interpretations,
  - final comparison report.
- Standard technical terms such as EEG, CSP, LDA, EEGNet, ATCNet, epoch,
  validation, inference, and subject-dependent may remain in English.
- Use type hints.
- Use pathlib instead of hard-coded path strings.
- Avoid duplicated logic between notebooks.
- Notebook cells must call functions from src rather than reimplementing the
  whole pipeline.
- Include useful error messages and assertions.

DEPENDENCIES
============

Use a PyTorch-based stack.

Expected libraries include:

- Python 3.11 or a well-supported compatible version
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- mne
- moabb, primarily for dataset metadata/download fallback
- torch
- braindecode
- pyyaml
- tqdm
- joblib
- pyarrow
- jupyter
- ipykernel
- pytest
- requests
- seaborn only if genuinely needed, but matplotlib should be sufficient

Pin mutually compatible versions in requirements.txt and environment.yml.
Document the selected versions.

Prefer a current stable Braindecode implementation for:

- EEGNetv4
- ATCNet

Wrap external models behind local factory functions so model construction is
centralized and testable.

If the installed stable Braindecode version does not expose a working ATCNet:

1. Do not silently substitute another model.
2. Use the official ATCNet repository/paper as the implementation reference.
3. Implement ATCNet locally in PyTorch.
4. Document all deviations from the reference architecture.
5. Add input/output shape tests.
6. Ensure the model supports:
   - 64 EEG channels,
   - 2 output classes,
   - the selected number of time samples.

DATASET DOWNLOAD
================

Create an idempotent downloader.

Required behavior:

1. Download s01.mat through s52.mat into:
   data/raw/cho2017/

2. Support:
   - resuming partial downloads,
   - retries with exponential backoff,
   - configurable timeout,
   - skipping valid existing files,
   - force-redownload option,
   - progress reporting.

3. Create:
   data/manifests/file_manifest.csv

Required columns:

- subject_id
- filename
- source_url
- local_path
- size_bytes
- download_status
- validation_status
- downloaded_at
- error_message

4. Validate every downloaded file by:
   - confirming the file exists,
   - confirming it is non-empty,
   - attempting scipy.io.loadmat,
   - confirming an `eeg` structure is present,
   - checking expected major fields,
   - checking the sampling rate,
   - checking major array dimensions.

5. Do not use the MOABB RawArray conversion as the canonical data loader.
   MOABB concatenates class recordings into a pseudo-continuous signal and may
   introduce edge effects around artificial boundaries.

6. It is acceptable to use MOABB only:
   - as a download fallback,
   - to verify channel naming,
   - to compare metadata.

7. Store the original .mat files unchanged.

8. Do not commit raw dataset files to Git. Add them to .gitignore.

MATLAB DATA INSPECTION
======================

Before writing assumptions into the loader, inspect at least several files,
including:

- s01
- s20
- s29
- s33
- s34
- s52

Create a script that prints and saves:

- top-level MATLAB keys,
- fields under the eeg struct,
- shape and dtype of each relevant field,
- sampling rate,
- number of event onsets,
- signal duration,
- number of channels,
- number of trials,
- structure and contents of bad_trial_indices,
- whether trial numbering is zero-based or one-based,
- whether bad trial indices are class-specific or global,
- whether explicit run identifiers exist,
- whether pre-stimulus samples are available.

Save this inspection report to:

results/eda/reports/mat_structure_report.md

Never guess how bad_trial_indices maps to trials. Determine it from the file
contents and validate it with assertions.

CANONICAL EPOCH EXTRACTION
==========================

Parse the original .mat files directly.

Do not blindly reshape signal arrays into trial tensors.

For each subject and class:

1. Read the relevant continuous class-specific array:
   - imagery_left
   - imagery_right

2. Use imagery_event onset markers to locate trial onset samples.

3. Extract each original trial using event onsets and the verified frame/sampling
   information.

4. Preserve:
   - subject_id,
   - class_label,
   - original_class_trial_index,
   - original_event_sample,
   - run_id,
   - trial_id,
   - bad-amplitude flag,
   - bad-EMG flag,
   - included/excluded status,
   - exclusion reason.

5. Use labels:
   - 0 = left_hand
   - 1 = right_hand

6. Keep EEG and EMG separate:
   - EEG shape: trials × 64 × time
   - EMG shape: trials × 4 × time

7. EMG must never be used as a predictive input in the three classification
   experiments.

8. EMG may only be used for:
   - artifact analysis,
   - validating provided bad-trial flags,
   - EDA figures,
   - demonstrating why muscle leakage is dangerous.

9. Preserve the full available trial interval in interim data. Do not discard
   potentially useful pre-onset samples before EDA has inspected them.

BAD TRIAL AND SUBJECT POLICY
============================

Implement explicit artifact bookkeeping.

Default main-analysis policy:

- Exclude s29 and s34 from model comparison.
- Retain them in EDA.
- For the remaining 50 participants, remove all trials identified by the
  dataset-provided bad_trial_indices.
- Preserve separate reasons where possible:
  - amplitude threshold
  - EMG correlation
  - other provided flags

Create:

data/manifests/excluded_trials.csv

Required columns:

- subject_id
- class_label
- original_trial_index
- run_id
- exclusion_reason
- source
- notes

EDA must show:

- trial count before exclusions,
- trial count after exclusions,
- exclusions by participant,
- exclusions by class,
- exclusions by reason,
- percentage removed,
- special plots for s29 and s34.

Do not independently reimplement the expensive permutation-based EMG rejection
algorithm for the main pipeline unless necessary. Prefer the provided bad-trial
indices. It is acceptable to calculate simpler EMG summaries in EDA for
illustration, but do not use EDA-derived thresholds to modify the test set.

RUN IDENTIFICATION
==================

The paper describes five or six runs and 100 or 120 trials per class.

First check whether explicit run information exists in the .mat files.

If explicit run information exists:
- use it.

If explicit run information does not exist:
- determine whether trial order and counts reliably support reconstructing runs
  as contiguous blocks of 20 trials per class,
- verify that each class contains exactly 100 or 120 trials,
- verify that the inferred number of runs is exactly 5 or 6,
- set:
  run_id_source = "inferred_from_order"
- document that this run assignment is inferred from the experimental protocol,
  not directly supplied.

If any file violates those assumptions:
- do not silently infer run IDs,
- record the issue,
- use the documented stratified fallback split only for that subject.

SHARED TRAIN/VALIDATION/TEST SPLIT
==================================

The primary benchmark is subject-dependent.

A separate model is trained and evaluated for each participant.

Preferred run-aware split:

For a participant with 5 runs:
- train: runs 1, 2, 3
- validation: run 4
- test: run 5

For a participant with 6 runs:
- train: runs 1, 2, 3, 4
- validation: run 5
- test: run 6

This is a chronological, later-run holdout and must be the default because it
reduces same-run leakage and provides a more realistic estimate than a purely
random trial split.

Assign split membership using the original trial/run identity before removing
bad trials. Then remove excluded trials without moving replacement trials from
another split.

If reliable run IDs cannot be obtained for a participant, use a deterministic
fallback:

- 70% train
- 15% validation
- 15% test
- stratified by class
- random_state = 42

Record:

- split_method = "run_holdout"
or
- split_method = "stratified_fallback"

Create:

data/manifests/split_manifest.csv

Required columns:

- subject_id
- trial_id
- class_label
- original_trial_index
- run_id
- run_id_source
- included
- exclusion_reason
- split
- split_method
- random_seed

Validate with automated tests:

- no trial_id appears in more than one split,
- no test trial appears in train or validation,
- both classes appear in every usable split,
- excluded trials are not included,
- s29 and s34 are not in the primary model-training manifest,
- all three experiments consume the same manifest,
- each subject has enough usable trials.

Create a human-readable split summary table containing, per participant:

- train left/right counts,
- validation left/right counts,
- test left/right counts,
- total excluded counts,
- split method.

PRIMARY PREPROCESSING
=====================

For the main controlled comparison, use the same core EEG input for all three
methods.

Canonical primary input:

- EEG channels only: 64
- common average reference
- band-pass: 8–30 Hz
- motor-imagery interval: 0.5–2.5 seconds after stimulus onset
- resample from 512 Hz to 256 Hz after filtering
- expected final shape:
  trials × 64 × 512

Filtering requirements:

- use a well-documented zero-phase filter suitable for offline analysis,
- avoid filtering each short epoch in a way that produces severe edge artifacts,
- preferably filter sufficiently padded continuous/class-specific data before
  final epoch cropping,
- document filter method, transition bandwidth, and padding behavior,
- do not apply a 50 Hz notch merely because the code runs in Türkiye,
- the source data were recorded in Korea and metadata indicate 60 Hz line
  frequency,
- a separate notch is unnecessary for the primary 8–30 Hz band-pass input.

Do not perform ICA automatically in the main benchmark. The dataset already
provides bad-trial information, and arbitrary ICA decisions would complicate
reproducibility.

NEURAL-NETWORK NORMALIZATION
============================

For EEGNet and ATCNet:

1. Use only the training trials of the current participant to calculate
   normalization parameters.

2. Normalize per EEG channel using one clearly documented strategy, preferably:
   - mean and standard deviation over all training trials and time samples,
     separately for each channel.

3. Apply the frozen training statistics to:
   - training,
   - validation,
   - test.

4. Add epsilon for numerical stability.

5. Save normalization statistics with the checkpoint.

6. Do not normalize each complete dataset before splitting.

7. Do not calculate normalization statistics independently on validation or
   test data.

CLASS BALANCE
=============

The dataset is approximately balanced.

- Do not use class weighting by default.
- Calculate class counts after bad-trial removal.
- If a training split has more than 10% class imbalance, allow optional
  train-only class weights.
- Never derive weights from validation or test labels.
- Report when weights are activated.

PROCESSED DATA CACHE
====================

Create one processed file per participant, for example:

data/processed/common/s01.npz

Each file should contain:

- X_eeg
- X_emg, where useful for EDA only
- y
- trial_ids
- run_ids
- original_trial_indices
- class_names
- sampling_rate
- channel_names
- preprocessing metadata

Use float32 for model inputs.

Do not create one enormous in-memory array containing all subjects if it is not
necessary.

Create an idempotent preparation script. It must skip valid processed files
unless force=True.

EXPLORATORY DATA ANALYSIS NOTEBOOK
==================================

Create:

notebooks/00_exploratory_data_analysis.ipynb

This notebook must be detailed and presentation-quality.

It must not train the three classification models.

The notebook must contain Turkish Markdown explanations for:

1. Dataset motivation
2. Motor imagery concept
3. Left/right hand classification problem
4. EEG versus EMG
5. Dataset experimental protocol
6. Channel layout
7. Trial structure
8. Artifact risks
9. Why subject variability matters
10. What the following experiments will test

EDA sections must include:

A. Dataset integrity
- number of downloaded files,
- missing/corrupt files,
- fields present in each file,
- sampling-rate consistency,
- channel-count consistency,
- NaN and infinite-value checks.

B. Participant and trial overview
- total participants,
- included participants,
- excluded participants,
- trials per participant and class,
- before/after artifact removal,
- five-run versus six-run participants.

C. Bad-trial analysis
- bad-trial counts by subject,
- bad-trial percentage by subject,
- bad-trial reason distribution,
- detailed comparison of s29 and s34 with typical subjects,
- EMG contamination summaries.

D. Raw signal examples
- representative EEG channels,
- representative EMG channels,
- left and right trials,
- clean versus bad trial examples,
- use C3, C4, Cz and nearby sensorimotor channels where available.

E. Amplitude and variance analysis
- channel-wise amplitude distribution,
- trial-wise peak-to-peak amplitude,
- channel variance,
- outlier participants,
- missing or flat channels if any.

F. Frequency-domain analysis
- Welch PSD,
- average PSD by class,
- mu band approximately 8–13/14 Hz,
- beta band approximately 13/14–30 Hz,
- C3 versus C4 comparison,
- subject-wise and grand-average plots.

G. Spatial analysis
- electrode layout,
- sensor positions,
- topographic maps of class-wise band power,
- left-minus-right and right-minus-left differences,
- clear explanation of contralateral sensorimotor patterns.

H. ERD/ERS analysis
- determine whether sufficient pre-stimulus baseline samples are available,
- if available, calculate ERD/ERS with a documented baseline such as
  -0.5 to 0 seconds,
- if unavailable, do not fabricate a baseline,
- instead show task-period relative band-power differences and clearly state
  the limitation.

I. Class separability
- simple feature-space visualization using train-independent descriptive
  features,
- do not tune a classifier in EDA,
- examples may include C3/C4 mu and beta power distributions.

J. Split inspection
- train/validation/test counts,
- run-aware split visualization,
- class balance per split,
- verification that no trial overlaps.

K. EDA conclusions
- what appears learnable,
- which participants are difficult,
- where artifact leakage may occur,
- what performance variation should be expected,
- implications for the three models.

EDA outputs must automatically save to:

results/eda/figures/
results/eda/tables/
results/eda/reports/

At minimum save:

- dataset_integrity.csv
- subject_trial_counts.csv
- bad_trial_summary.csv
- channel_quality_summary.csv
- split_summary.csv
- dataset_summary.md
- all major plots as PNG with at least 150 or 200 DPI

The notebook must display figures inline and save them.

EXPERIMENT 1: CSP + LDA
=======================

Create:

notebooks/01_traditional_csp_lda.ipynb

Purpose:
Implement a strong conventional motor-imagery baseline.

Pipeline:

1. Load the shared processed EEG and split manifest.
2. For each included participant:
   - train on that participant's training trials,
   - tune using only validation trials,
   - evaluate once on the untouched test trials.

3. Use:
   - MNE CSP
   - scikit-learn LDA

4. Default literature-like candidate:
   - 4 CSP components total, corresponding approximately to two spatial filters
     per class,
   - log average-power CSP features,
   - LDA.

5. Validation hyperparameter candidates may include:
   - CSP n_components: [4, 6, 8]
   - CSP regularization:
     - None
     - Ledoit-Wolf or an equivalent supported shrinkage option
   - LDA candidates:
     - standard SVD LDA
     - shrinkage LDA with an appropriate solver

6. Keep the search small and scientifically interpretable.

7. Fit CSP only on training trials.

8. Select hyperparameters by validation balanced accuracy.

9. Refit the selected pipeline using train + validation only after
   hyperparameter selection, then evaluate once on test.

10. Save the final CSP patterns, filters, selected hyperparameters, and LDA
    model for each participant.

Required interpretability outputs:

- CSP spatial patterns for representative participants,
- topomaps of the first and last discriminative CSP components,
- distribution of log-variance features,
- 2D feature plot where meaningful,
- explanation of how CSP relates to sensorimotor rhythms.

Required outputs:

results/csp_lda/tables/subject_metrics.csv
results/csp_lda/tables/selected_hyperparameters.csv
results/csp_lda/predictions/test_predictions.parquet
results/csp_lda/metrics/aggregate_metrics.json
results/csp_lda/figures/aggregate_confusion_matrix.png
results/csp_lda/figures/subject_accuracy_distribution.png
results/csp_lda/figures/subject_accuracy_sorted.png
results/csp_lda/models/<subject_id>.joblib
results/csp_lda/configs/resolved_config.yaml
results/csp_lda/logs/experiment.log

EXPERIMENT 2: EEGNET
====================

Create:

notebooks/02_eegnet.ipynb

Use PyTorch and preferably Braindecode EEGNetv4 behind a local wrapper.

Required model properties:

- input shape: batch × 64 × 512
- output: 2 class logits
- compact parameter count
- temporal convolution
- depthwise spatial convolution
- separable convolution
- dropout
- no softmax inside the model when using CrossEntropyLoss

Train a separate EEGNet model for each included participant.

Default training configuration:

- loss: CrossEntropyLoss
- optimizer: AdamW
- initial learning rate: approximately 1e-3
- weight decay: approximately 1e-4
- batch size: 32, configurable
- maximum epochs: 200
- early-stopping patience: 25
- monitor: validation balanced accuracy or validation loss, choose one and
  document it consistently
- learning-rate scheduler: ReduceLROnPlateau or another simple documented
  scheduler
- gradient clipping: optional but configurable
- deterministic seed: 42
- CUDA support
- CPU fallback
- mixed precision: optional and enabled only when CUDA is available
- save best validation checkpoint only

Do not use test performance to choose the best epoch.

After selecting the best checkpoint:

- optionally refit is not required for the neural models,
- evaluate the best validation checkpoint once on test,
- record the best epoch,
- record training time,
- record inference latency,
- record parameter count.

Do not add complex augmentation to the primary experiment.

Allow augmentation only as a disabled configuration option for future work:

- small Gaussian noise,
- small temporal shift,
- training trials only.

Required outputs:

results/eegnet/tables/subject_metrics.csv
results/eegnet/tables/training_summary.csv
results/eegnet/predictions/test_predictions.parquet
results/eegnet/checkpoints/<subject_id>/best.pt
results/eegnet/histories/<subject_id>.csv
results/eegnet/metrics/aggregate_metrics.json
results/eegnet/figures/aggregate_confusion_matrix.png
results/eegnet/figures/subject_accuracy_distribution.png
results/eegnet/figures/subject_accuracy_sorted.png
results/eegnet/figures/representative_training_curves.png
results/eegnet/configs/resolved_config.yaml
results/eegnet/logs/experiment.log

Checkpoint contents must include:

- model state_dict,
- resolved model configuration,
- normalization statistics,
- class mapping,
- sampling frequency,
- channel names,
- input time window,
- best epoch,
- validation metric,
- random seed,
- subject ID.

EXPERIMENT 3: ATCNET
====================

Create:

notebooks/03_atcnet.ipynb

Use a validated PyTorch ATCNet implementation.

Prefer the current stable Braindecode implementation. If it is unavailable or
incompatible, implement a local PyTorch version based on the official ATCNet
reference.

Required conceptual blocks:

- convolutional temporal-spatial feature extractor,
- overlapping temporal windows,
- attention mechanism,
- temporal convolutional network,
- classification head.

Required model properties:

- input shape: batch × 64 × 512
- output: 2 class logits
- configurable number of windows
- configurable attention settings
- configurable TCN depth and kernel size
- dropout
- no softmax inside model with CrossEntropyLoss

Before notebook completion:

- add a unit test using random data,
- verify forward output shape,
- verify backward pass,
- verify parameter count is finite and non-zero,
- verify the model handles batch sizes 1 and greater than 1.

Training protocol must be as similar as practical to EEGNet:

- same participants,
- same train/validation/test splits,
- same normalized inputs,
- same random seed policy,
- same loss,
- AdamW,
- comparable batch size,
- early stopping,
- validation-only model selection,
- untouched test evaluation.

Suggested configurable defaults:

- learning rate: 1e-3 or a documented architecture-appropriate value
- weight decay: 1e-4
- maximum epochs: 250
- early-stopping patience: 30
- batch size: 32
- seed: 42

Keep the hyperparameter search limited.

Do not run a large architecture search. The point is a clear model comparison,
not hyperparameter mining.

Required outputs:

results/atcnet/tables/subject_metrics.csv
results/atcnet/tables/training_summary.csv
results/atcnet/predictions/test_predictions.parquet
results/atcnet/checkpoints/<subject_id>/best.pt
results/atcnet/histories/<subject_id>.csv
results/atcnet/metrics/aggregate_metrics.json
results/atcnet/figures/aggregate_confusion_matrix.png
results/atcnet/figures/subject_accuracy_distribution.png
results/atcnet/figures/subject_accuracy_sorted.png
results/atcnet/figures/representative_training_curves.png
results/atcnet/configs/resolved_config.yaml
results/atcnet/logs/experiment.log

COMMON METRICS
==============

For every participant and model, calculate:

- accuracy
- balanced accuracy
- macro F1
- Cohen's kappa
- ROC AUC using positive-class probabilities
- precision for each class
- recall for each class
- F1 for each class
- confusion-matrix values
- number of train trials
- number of validation trials
- number of test trials
- selected/best epoch where applicable
- training duration
- mean inference latency per trial
- median inference latency per trial
- p95 inference latency per trial
- trainable parameter count
- model/checkpoint size where applicable

Use probabilities, not hard labels, for ROC AUC.

Handle cases where a metric is undefined. Record NaN and a reason instead of
crashing or silently substituting zero.

LATENCY MEASUREMENT
===================

Implement consistent inference timing:

- warm up the model before measurement,
- use batch size 1 for per-trial latency,
- synchronize CUDA before and after timed sections,
- exclude data loading and preprocessing unless separately reported,
- repeat enough times for a stable estimate,
- report CPU/GPU device information,
- report preprocessing latency separately for CSP if measured,
- do not compare GPU neural inference directly against CPU CSP without clearly
  reporting the device difference.

AGGREGATE REPORTING
===================

For each model, calculate:

- mean
- standard deviation
- median
- interquartile range
- minimum
- maximum
- 95% bootstrap confidence interval

Aggregate metrics must be based on participant-level scores, not only on pooled
trial predictions.

Also provide pooled test metrics, but label them explicitly as pooled metrics.

Required plots per model:

- aggregate confusion matrix,
- normalized confusion matrix,
- sorted participant accuracy,
- box plot or violin plot of participant accuracy,
- accuracy histogram,
- participant-wise macro-F1,
- accuracy versus number of clean training trials,
- training time by participant,
- inference latency summary.

FINAL COMPARISON NOTEBOOK
=========================

Create:

notebooks/04_final_model_comparison.ipynb

This notebook must not retrain models.

It must load saved outputs from:

- CSP + LDA
- EEGNet
- ATCNet

It must first validate that:

- all methods used the same split-manifest hash,
- all methods used the same included participant list,
- all methods evaluated the same test trial IDs,
- class mappings are identical,
- preprocessing version is compatible.

If the checks fail, stop with a clear error.

Comparison sections:

1. Project and research question
2. Evaluation protocol
3. Dataset exclusions
4. Per-model aggregate metrics
5. Participant-wise paired comparison
6. Statistical tests
7. Computational-cost comparison
8. Failure analysis
9. Interpretation
10. Limitations
11. Final conclusion

Required comparison tables:

- model summary table
- subject-wise accuracy table
- subject-wise macro-F1 table
- parameter/training/inference-cost table
- pairwise statistical-test table
- participants where model rankings differ
- common hard-participant table

Required comparison plots:

- side-by-side subject accuracy distributions
- paired participant accuracy lines
- subject × model accuracy heatmap
- sorted per-subject accuracy for all models
- mean metric comparison with 95% CI
- training-time comparison
- inference-latency comparison
- parameter-count comparison
- pairwise accuracy-difference plots
- confusion matrices in a consistent layout
- model rank per participant

STATISTICAL ANALYSIS
====================

Because all models are evaluated on the same participants, use paired
participant-level statistical analysis.

For each pair:

- CSP + LDA vs EEGNet
- CSP + LDA vs ATCNet
- EEGNet vs ATCNet

Perform:

- Wilcoxon signed-rank test on participant test accuracy
- Wilcoxon signed-rank test on participant macro-F1
- bootstrap 95% confidence interval of the mean paired difference
- an appropriate paired effect size, such as rank-biserial correlation or
  Cohen's dz, with the choice documented

Apply Holm correction across the three model-pair comparisons for each metric.

Report:

- raw p-value
- corrected p-value
- effect size
- mean difference
- median difference
- 95% confidence interval
- number of paired participants
- whether the result is statistically significant at alpha = 0.05

Do not claim practical superiority from statistical significance alone.
Interpret effect size and computational cost.

RESULT FILE SCHEMAS
===================

subject_metrics.csv must contain at least:

- model_name
- subject_id
- split_method
- n_train
- n_validation
- n_test
- accuracy
- balanced_accuracy
- macro_f1
- cohen_kappa
- roc_auc
- left_precision
- left_recall
- left_f1
- right_precision
- right_recall
- right_f1
- tn
- fp
- fn
- tp
- best_epoch
- train_time_seconds
- inference_mean_ms
- inference_median_ms
- inference_p95_ms
- trainable_parameters
- status
- error_message
- seed
- split_manifest_hash
- preprocessing_hash

test_predictions.parquet must contain:

- model_name
- subject_id
- trial_id
- original_trial_index
- run_id
- true_label
- predicted_label
- probability_left
- probability_right
- split
- seed
- checkpoint_path

Every experiment must save a resolved copy of its configuration.

REPRODUCIBILITY
===============

Implement one central seed function that configures:

- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA
- deterministic settings where possible

Document the performance consequences of strict determinism.

Default seed:
42

Create hashes for:

- split manifest
- preprocessing configuration
- model configuration

Save:

- Python version
- package versions
- operating system
- CPU
- GPU
- CUDA version
- cuDNN version
- execution timestamp
- Git commit hash when available

Save environment information under each result folder.

CONFIGURATION
=============

Do not hard-code experiment parameters throughout notebooks.

Use YAML configurations.

base.yaml should define:

- project paths
- subject list
- excluded subjects
- seed
- class mapping
- sampling rates
- time window
- filter band
- reference method
- artifact policy
- split policy
- device policy
- quick mode

Model-specific YAML files should override:

- model parameters
- optimizer
- scheduler
- batch size
- epochs
- patience
- model-specific hyperparameters

Implement:

QUICK_MODE = true/false

Quick mode may use a small explicitly listed participant subset, for example:

[1, 2, 3, 4, 5]

Quick mode is only for pipeline debugging.

All notebooks must prominently state:

"QUICK_MODE sonuçları nihai bilimsel sonuç olarak kullanılamaz."

Full mode must use all 50 eligible participants.

NOTEBOOK QUALITY
================

Every notebook must:

- start with title, purpose, research question, and expected outputs,
- load configuration from YAML,
- display the active configuration,
- use project-root discovery robustly,
- verify required data files,
- verify split manifest,
- avoid duplicated source implementation,
- include progress reporting,
- save all outputs,
- include a final result summary,
- include a section named "Limitations",
- include a section named "Generated Files",
- be restartable from a clean kernel,
- avoid hidden state dependencies,
- avoid absolute user-specific paths.

Use small notebook cells with clear responsibilities.

Add a final assertion in each experiment notebook verifying that all expected
result files were produced.

TESTS
=====

Create tests for at least:

1. MAT loader returns expected fields.
2. EEG/EMG channel separation is correct.
3. Event onset count matches trial count.
4. Trial epoch length is correct.
5. Bad-trial indices map to valid trials.
6. Excluded trials never enter a split.
7. A trial never appears in multiple splits.
8. Train/validation/test contain both classes.
9. Neural normalization uses training statistics only.
10. EEGNet forward output is batch × 2.
11. ATCNet forward output is batch × 2.
12. Neural models support backward propagation.
13. Metrics function returns the defined schema.
14. Result writer creates required columns.
15. Split and preprocessing hashes are deterministic.

Do not require the full dataset for every unit test. Create tiny synthetic
fixtures for most tests, and mark integration tests separately.

README
======

Write a detailed Turkish README containing:

1. Project purpose
2. Scientific motivation
3. Dataset description
4. Dataset citation
5. Model descriptions
6. Directory structure
7. Installation
8. Dataset download
9. Data validation
10. Data preparation
11. Notebook execution order
12. Quick mode
13. Full mode
14. GPU/CPU usage
15. Output locations
16. Reproducibility
17. Leakage-prevention rules
18. Expected limitations
19. How to cite the original dataset and model papers
20. Troubleshooting

Required execution order:

1. Install environment
2. Download dataset
3. Validate dataset
4. Prepare epochs and manifests
5. Run 00_exploratory_data_analysis.ipynb
6. Run 01_traditional_csp_lda.ipynb
7. Run 02_eegnet.ipynb
8. Run 03_atcnet.ipynb
9. Run 04_final_model_comparison.ipynb

Add command examples such as:

python scripts/download_dataset.py
python scripts/validate_dataset.py
python scripts/prepare_dataset.py
python scripts/create_splits.py
pytest -q

Makefile targets should include:

- install
- download
- validate-data
- prepare-data
- test
- clean-cache
- status

GITIGNORE
=========

Ignore:

- raw dataset files
- processed data
- model checkpoints
- notebook checkpoints
- temporary logs
- local environments
- large generated predictions
- generated plots where appropriate

Do not ignore lightweight configuration and schema files.

LIMITATIONS TO STATE EXPLICITLY
===============================

The project must explicitly state that:

1. The dataset has only one recording session per participant.
2. This is not a true cross-day or cross-session generalization study.
3. The main experiment is subject-dependent.
4. Inferred run identifiers may depend on trial ordering if explicit run labels
   are absent.
5. EEG motor-imagery performance varies strongly across participants.
6. Excluding only high-EMG subjects does not eliminate all possible artifact
   leakage.
7. Offline test accuracy does not equal online BCI usability.
8. Statistical significance does not guarantee clinical or operational value.
9. A deep model may overfit due to limited per-participant training data.
10. CSP + LDA may remain competitive or outperform neural models for some
    participants.
11. No result should be claimed until the notebooks have actually been run.
12. Empty results directories do not constitute experiment results.

FINAL ACCEPTANCE CRITERIA
=========================

The task is complete only when:

- the complete project structure exists,
- dataset download is implemented and attempted,
- all successfully downloaded files pass lightweight integrity checks,
- direct .mat parsing is implemented,
- bad-trial mapping is explicit and tested,
- the shared split manifest generator exists,
- the EDA notebook is detailed,
- all three model experiments are separate notebooks,
- the final comparison notebook exists,
- result directories and schemas are prepared,
- no training notebook has been executed,
- no fake metrics or plots exist,
- README explains the full workflow,
- tests exist and lightweight tests pass,
- model input/output shape tests pass,
- the final terminal response clearly summarizes what was created.

Before finishing, inspect the whole repository for:

- duplicated preprocessing,
- accidental test leakage,
- hard-coded absolute paths,
- inconsistent class mappings,
- inconsistent channel order,
- inconsistent sampling rates,
- result paths that do not match README,
- notebooks that depend on variables from previous notebooks,
- model code that incorrectly applies softmax before CrossEntropyLoss,
- checkpoint files accidentally created without training.

Do not leave placeholder pseudocode such as "TODO: train model" for core
functionality. The notebooks may remain unexecuted, but they must contain
complete executable implementations.