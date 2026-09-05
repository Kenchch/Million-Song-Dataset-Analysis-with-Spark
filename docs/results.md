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

| Metric | Recorded value | What it actually measures |
| --- | ---: | --- |
| Precision@10 | 0.1333 | 4 hits across 30 slots |
| NDCG@10 | 0.1428 | same 3 users |
| "MAP@10" | 0.0035 | `meanAveragePrecision`, which ignores `k` |

**These three numbers were computed on three users, not on the held-out set.**
The cell that built the comparison frame took
`test_df.select("user_idx").distinct().limit(3)` to show a few example users
side by side, and the evaluation cell further down scored that same frame
rather than rebuilding it for the full population. Precision@10 of 0.1333 is
therefore 4 hits out of 3 users x 10 slots, and the figure carries no useful
confidence at that sample size. The 8,622,941 held-out interactions in the
table above were produced by the split; they were never scored.

**The third metric is not MAP@10.** The evaluator was constructed as
`RankingEvaluator(metricName="meanAveragePrecision", k=10)`. Spark exposes
`meanAveragePrecision` and `meanAveragePrecisionAtK` as separate metrics, and
the former ignores the `k` it is given: it averages precision over the whole
prediction list and normalises by the number of relevant items, not by
`min(k, relevant)`. That different denominator, not a modelling failure, is why
it sits roughly forty times below Precision@10. `src/metrics.py` implements
MAP@10 proper, so it would not reproduce 0.0035 from the same inputs.

Both problems are in the recorded evaluation, not in the model: the ALS fit
itself used the full training set. Nothing here is restated as a corrected
number, because recomputing one needs the cluster and the data snapshot, and
neither is available.

The current CLI evaluates differently in one further respect: it excludes each
user's training interactions from their recommendations, which the notebook did
not do. `recommendForAllUsers` scores every item, so the raw top-10 is
dominated by songs the split had already moved into training and which
therefore cannot match a held-out row. Any comparison against the numbers above
is a comparison across three separate changes.

These results depend on the exact data snapshot, filtering order, random split, Spark version, cluster layout, and evaluation implementation. They should be treated as provenance-backed project results rather than as a comparison with current recommender-system research.
