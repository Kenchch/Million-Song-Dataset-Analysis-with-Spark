# Recorded results and interpretation

The figures below were extracted from the original executed notebooks before their outputs were removed for credential hygiene.

## Data scale

| Stage | Recorded value |
| --- | ---: |
| Track metadata rows | 1,000,000 |
| Unique songs in metadata | 998,963 |
| Audio feature storage | 12.2 GB |
| Taste Profile interactions | 48,373,586 |
| Taste Profile users | 1,019,318 |
| Taste Profile songs | 384,546 |
| Users after minimum-activity filtering | 647,808 |
| Songs after minimum-popularity filtering | 161,173 |
| ALS training interactions | 33,197,154 |
| ALS test interactions | 8,622,941 |

## Genre modelling

| Model | Recorded result |
| --- | ---: |
| Logistic regression | accuracy 0.9143; AUROC 0.8730 |
| Random forest | accuracy 0.9173; precision 0.6899; AUROC 0.8472 |
| Gradient-boosted trees | accuracy 0.9143; AUROC 0.8882 |
| Multiclass model | accuracy 0.4089; F1 0.4717 |

Accuracy should be interpreted alongside AUROC and class balance because the binary genre target is imbalanced.

## ALS recommendation experiment

The recorded notebook used implicit feedback with rank 20, regularization 0.05, alpha 20, 15 iterations, and seed 25.

| Metric | Recorded value |
| --- | ---: |
| Precision@10 | 0.1333 |
| NDCG@10 | 0.1428 |
| MAP@10 | 0.0035 |

These results depend on the exact data snapshot, filtering order, random split, Spark version, cluster layout, and evaluation implementation. They should be treated as provenance-backed project results rather than as a comparison with current recommender-system research.
