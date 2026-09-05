# Million Song Dataset analysis with PySpark

[![CI](https://github.com/Kenchch/Million-Song-Dataset-Analysis-with-Spark/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenchch/Million-Song-Dataset-Analysis-with-Spark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Distributed feature engineering, genre modelling, and implicit-feedback recommendations over the Million Song Dataset. The project combines the original analysis notebooks with reusable Spark command-line pipelines and dependency-free ranking-metric tests.

## At a glance

| Data scale | Audio modelling | Recommendations |
| --- | --- | --- |
| **48,373,586** listening events, **1,019,318** users, **384,546** songs | **12.2 GB** of audio feature files; best recorded binary result: **0.8882 AUROC** | ALS trained on **33,197,154** interactions, holding out **8,622,941** |

The recorded ALS ranking numbers do **not** describe that held-out set. They were
produced by a notebook cell that scored three sampled users, and they are
reported as such in [the results and limitations](docs/results.md), which also
covers two defects found in that evaluation while preparing this repository.

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
```

## Reproducibility choices

- The attributes CSV drives explicit Spark schemas; feature files are not inferred.
- Binary classifiers use a deterministic random seed and save their fitted Spark pipeline.
- Genre data is split before balancing; only the training partition is downsampled and used to select features. The held-out partition retains its original class mix. Logistic regression uses its deterministic optimizer; the split and stochastic tree models use explicit seeds.
- ALS splitting is performed within each user, so every test user also appears in training. The split orders each user's rows by a hash rather than `rand()`: the train and test frames are two filters over one plan, evaluated by separate actions, and a nondeterministic key generated twice can put a row in both halves or in neither.
- Recommendations exclude each user's training interactions. ALS scores every item, so the raw top-k is mostly songs the split already moved into training, which can never match a held-out row and merely displace candidates that could.
- The recommendation workflow removes sparse users and songs before factorisation. Tune the thresholds to your experiment rather than treating the report's values as universal.
- Inputs, model checkpoints, and output directories are gitignored.
- CI validates the ranking metrics, Notebook JSON, output stripping, and absence of known credential remnants without requiring the 48.4-million-row dataset.
- A separate CI job runs real Spark training on synthetic data: all three genre models, and an ALS fit asserting no recommendation is an item the user was trained on. It also asserts the unfiltered call does return such items, so the reason for the exclusion stays visible.

The standalone ranking helpers require a positive integer cutoff, credit each
relevant item once at its original rank, and retain duplicate slots as missed
opportunities. Precision divides by the requested cutoff even for shorter lists.
The ALS command currently exports recommendations and held-out interactions;
it does not execute these helpers or regenerate the historical notebook scores.

The code provides the pipeline; it does not claim that results will exactly reproduce the report without the same source snapshot, cluster configuration, preprocessing, and random seed.
