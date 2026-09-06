# Million Song Dataset analysis with PySpark

[![CI](https://github.com/Kenchch/Million-Song-Dataset-Analysis-with-Spark/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenchch/Million-Song-Dataset-Analysis-with-Spark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Distributed feature engineering, genre modelling, and implicit-feedback recommendations over the Million Song Dataset. The project combines the original analysis notebooks with reusable Spark command-line pipelines and dependency-free ranking-metric tests.

## At a glance

The source study used Taste Profile listening triplets and Million Song audio features.
The reusable workflows implement genre classification and implicit-feedback ALS.
The recorded binary GBT result was 0.8882 AUROC; see [provenance and limitations](docs/results.md).
That AUROC is not comparable to a majority-class accuracy baseline, and does not certify the current CLI.

## Workflows

```text
Million Song data / course extracts
          |
          +--> audio feature families --> schema-driven joins --> feature reduction
          |                                      |
          |                                      +--> LR / Random Forest / GBT genre models
          |
          +--> Taste Profile triplets --> activity filters --> per-user 80/20 split
                                                     |
                                                     +--> implicit-feedback Spark ALS
                                                              |
                                                              +--> Precision@10 / NDCG@10 / MAP@10
```

The reusable implementation separates two workflows:

- audio-feature genre classification with logistic regression, random forest, or gradient-boosted trees;
- implicit-feedback song recommendations using Spark ALS.

## Selected report visuals

![Audio-feature correlation heatmap](assets/audio-feature-correlation-heatmap.png)

*Correlation heatmap used to identify redundant audio features before modelling.*

![MAGD genre distribution](assets/genre-distribution.png)

*The genre imbalance that motivates balancing before binary classification.*

## Layout

```text
notebooks/             Sanitized original processing, modelling, and recommendation notebooks
src/msd_pipeline.py    Reusable Spark command-line workflows
src/metrics.py         Pure-Python ranking metrics
docs/results.md        Recorded results, provenance, and interpretation limits
data/README.md         Expected inputs and responsible data handling
tests/                 Ranking-metric and notebook-safety tests
assets/                Selected visuals from the submitted analysis
```

## Setup

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the Million Song Dataset / course-provided extracts separately, then configure Spark to reach your local, HDFS, or cloud paths.

The notebooks document the original Azure-backed cluster analysis. Their saved outputs were removed before publication because Spark configuration output contained expired access parameters and a cluster username. The source now filters security-sensitive Spark configuration keys; see [notebooks/README.md](notebooks/README.md).

## Commands

```bash
# Inspect audio schema constructed from its attributes file
spark-submit src/msd_pipeline.py audio-schema \
  --attributes /data/audio/attributes/msd-jmir-mfcc-all-v1.0.attributes \
  --features /data/audio/features/msd-jmir-mfcc-all-v1.0

# Train and evaluate a binary genre classifier
spark-submit src/msd_pipeline.py train-genre \
  --audio /data/audio/features/combined.parquet \
  --genres /data/genre/msd-MAGD-genreAssignment.tsv \
  --positive-genre Pop_Rock --model gbt --output output/genre_gbt

# Filter interactions, make a per-user 80/20 split, and train implicit ALS
spark-submit src/msd_pipeline.py train-als \
  --triplets /data/tasteprofile/triplets.tsv --output output/als

# Evaluate every held-out user, including users with no recommendations
spark-submit src/msd_pipeline.py evaluate-als \
  --recommendations output/als/recommendations \
  --test-interactions output/als/test_interactions \
  --k 10 --output output/als_evaluation
```

## Reproducibility choices

### ALS input and export contract

Taste Profile input must contain one row per `(user_id, song_id)`, nonempty IDs,
and finite positive play counts. The loader rejects invalid or duplicate pairs
before filtering or splitting. It does not sum duplicates: an overlapping input
file must not silently double implicit-feedback confidence. These checks add
an input scan and a pair-count shuffle to ingestion.

The user-wise split retains a training item for every user and a held-out item
for users with at least two retained songs; single-song users remain training-only.
The activity filters are a single pass, so filtering songs can leave a user with
fewer songs than the initial user threshold. This is not iterative k-core filtering.

The ALS output includes `indexers/` (the fitted Spark indexing pipeline),
`user_mapping/` and `song_mapping/` Parquet tables alongside `model/`,
`recommendations/` and `test_interactions/`. Join recommendation indices to these
mapping tables to recover source IDs; do not refit indexers to decode an old run.
The files are written sequentially, so use a fresh output directory for each run;
this export is not an atomic publication mechanism.

### Evaluation and CI

- The attributes CSV drives explicit Spark schemas; feature files are not inferred.
- Binary classifiers use a deterministic random seed and save their fitted Spark pipeline.
- Genre data is split before balancing; only the training partition is downsampled and used to select features. The held-out partition retains its original class mix. Logistic regression uses its deterministic optimizer; the split and stochastic tree models use explicit seeds.
- ALS splitting is performed within each user, so every test user also appears in training. The split orders each user's rows by a hash of the row rather than by `rand()`, which seeds a generator per partition: the same rows laid out over a different number of partitions draw different numbers and land on different sides of the boundary. That is measured, not assumed — `test_split_does_not_depend_on_input_partitioning` fails when the `rand()` version is put back, and the failure is what the hash is there to prevent.
- Recommendations exclude each user's training interactions. ALS scores every item, so the raw top-k is mostly songs the split already moved into training, which can never match a held-out row and merely displace candidates that could.
- The recommendation workflow removes sparse users and songs before factorisation. Tune the thresholds to your experiment rather than treating the report's values as universal.
- Inputs, model checkpoints, and output directories are gitignored.
- CI validates the ranking metrics, Notebook JSON, output stripping, and absence of known credential remnants without requiring the 48.4-million-row dataset.
- A separate CI job runs real Spark training on synthetic data: all three genre models, and an ALS fit asserting no recommendation is an item the user was trained on. It also asserts the unfiltered call does return such items, so the reason for the exclusion stays visible.

The standalone ranking helpers require a positive integer cutoff, credit each
relevant item once at its original rank, and retain duplicate slots as missed
opportunities. Precision divides by the requested cutoff even for shorter lists.
`evaluate-als` computes macro Precision@K, NDCG@K and MAP@K with Spark's
RankingEvaluator, including held-out users with zero predictions. It saves a small
`metrics/` CSV and per-user `users/` Parquet with labels, predictions and hit counts.
Duplicate recommendation ranks or songs are rejected. The real-Spark fixture
compares all three metrics with the standalone helpers, including missing users.
This validates the implementation; historical notebook scores have not been regenerated.

ALS defaults are rank 20, regularization 0.05, alpha 20 and 15 iterations;
`--rank`, `--reg-param`, `--alpha` and `--max-iter` expose these choices.
Genre `--balance-ratio` defaults to 1.0; only training negatives are sampled.
Genre evaluation also reports PR-AUC, held-out row count and positive prevalence.
Rows with missing numeric features are removed before splitting and counted.
Install `requirements-notebooks.txt` separately for historical notebook dependencies.

The code provides the pipeline; it does not claim that results will exactly reproduce the report without the same source snapshot, cluster configuration, preprocessing, and random seed.
