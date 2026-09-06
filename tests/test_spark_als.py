"""Real-Spark checks on the ALS split and on what the model is allowed to recommend."""

import pytest

pytest.importorskip("pyspark")
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.msd_pipeline import (
    SEED,
    clean_triplets,
    evaluate_als,
    recommend_unseen,
    train_als,
    userwise_split,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("als-regression-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def indexed(spark):
    rows = [
        (f"u{user}", f"s{(user * 7 + item) % 40}", float(1 + item % 3))
        for user in range(30)
        for item in range(12)
    ]
    frame = spark.createDataFrame(rows, ["user_id", "song_id", "play_count"]).distinct()
    indexers = [
        StringIndexer(inputCol="user_id", outputCol="user_idx"),
        StringIndexer(inputCol="song_id", outputCol="song_idx"),
    ]
    return (
        Pipeline(stages=indexers)
        .fit(frame)
        .transform(frame)
        .select(
            "user_id",
            "song_id",
            F.col("user_idx").cast("int"),
            F.col("song_idx").cast("int"),
            "play_count",
        )
        .cache()
    )


def _pairs(frame):
    return {(row.user_idx, row.song_idx) for row in frame.select("user_idx", "song_idx").collect()}


def test_ranking_evaluation_includes_users_without_recommendations(spark, tmp_path):
    from src.metrics import average_precision_at_k, ndcg_at_k, precision_at_k

    recs, heldout, output = [str(tmp_path / name) for name in ("recs", "test", "evaluation")]
    spark.createDataFrame(
        [(1, 10, 1), (1, 20, 2), (3, 99, 1)], ["user_idx", "song_idx", "rank"]
    ).write.parquet(recs)
    spark.createDataFrame(
        [(1, 10), (1, 30), (2, 40), (3, 99)], ["user_idx", "song_idx"]
    ).write.parquet(heldout)
    metrics = evaluate_als(spark, recs, heldout, output, k=2)
    population = [([10, 20], {10, 30}), ([], {40}), ([99], {99})]
    assert metrics["users"] == 3
    assert metrics["hits"] == 2
    assert metrics["users_below_k"] == 2
    for name, function in [
        ("precision_at_k", precision_at_k),
        ("ndcg_at_k", ndcg_at_k),
        ("map_at_k", average_precision_at_k),
    ]:
        assert metrics[name] == pytest.approx(sum(function(r, t, 2) for r, t in population) / 3)
    assert spark.read.parquet(f"{output}/users").count() == 3


@pytest.mark.parametrize(
    "payload",
    [
        "u1\ts1\t1\nu1\ts1\t2\n",
        "u1\ts1\tNaN\n",
        "u1\ts1\tInfinity\n",
        "u1\ts1\t-1\n",
        "u1\ts1\t0\n",
        "u1\ts1\tbad\n",
        "\ts1\t1\n",
    ],
)
def test_invalid_triplets_are_rejected(spark, tmp_path, payload):
    path = tmp_path / "invalid.tsv"
    path.write_text(payload)
    with pytest.raises(ValueError):
        clean_triplets(spark, str(path), 1, 1)


def test_small_histories_keep_a_holdout(spark):
    frame = spark.createDataFrame(
        [("u1", "s1"), ("u1", "s2"), ("u2", "s3")], ["user_id", "song_id"]
    )
    train, test = userwise_split(frame)
    assert train.count() == 2
    assert [(r.user_id) for r in test.collect()] == ["u1"]
    for fraction in [0, 1, -0.1, float("nan")]:
        with pytest.raises(ValueError):
            userwise_split(frame, fraction)


def test_exported_recommendations_resolve_to_source_ids(spark, tmp_path):
    path = tmp_path / "triplets.tsv"
    path.write_text(
        "".join(f"u{u}\ts{(u + i) % 12}\t{i + 1}\n" for u in range(8) for i in range(6))
    )
    output = str(tmp_path / "output")
    train_als(spark, str(path), output, 1, 1)
    from pyspark.ml import PipelineModel

    restored = PipelineModel.load(f"{output}/indexers")
    users = spark.read.parquet(f"{output}/user_mapping")
    songs = spark.read.parquet(f"{output}/song_mapping")
    recommendations = spark.read.parquet(f"{output}/recommendations")
    assert users.count() == 8
    assert songs.count() == 12
    decoded = recommendations.join(users, "user_idx").join(songs, "song_idx")
    assert decoded.count() == recommendations.count() > 0
    reindexed = restored.transform(decoded.select("user_id", "song_id"))
    assert (
        reindexed.select("user_idx", "song_idx")
        .exceptAll(recommendations.select("user_idx", "song_idx"))
        .count()
        == 0
    )


class TestUserwiseSplit:
    def test_split_is_disjoint_and_loses_nothing(self, indexed):
        train, test = userwise_split(indexed)

        train_pairs, test_pairs = _pairs(train), _pairs(test)
        assert train_pairs & test_pairs == set(), "a row must not be in both halves"
        assert train_pairs | test_pairs == _pairs(indexed), "every input row must land somewhere"

    def test_every_test_user_also_appears_in_training(self, indexed):
        """ALS cannot score a user it never saw, so a user-wise split is only
        useful if the held-out users are all represented in training."""
        train, test = userwise_split(indexed)

        train_users = {r.user_idx for r in train.select("user_idx").distinct().collect()}
        test_users = {r.user_idx for r in test.select("user_idx").distinct().collect()}
        assert test_users <= train_users

    def test_split_does_not_depend_on_input_partitioning(self, indexed):
        """The ordering key has to be a function of the row, not of where the
        row happens to sit.

        `rand(seed)` seeds a generator per partition, so the same rows laid out
        over a different number of partitions draw different numbers, order
        differently within each user, and land on different sides of the
        boundary. A row hash cannot do that.

        Calling the split twice on one frame does not test this: a cached
        parent hands back a single fixed partitioning, so `rand(seed)` is
        stable across those two calls and the assertion passes whichever
        implementation is underneath. Repartitioning is what separates them.
        """
        one_partition = userwise_split(indexed.repartition(1))
        many_partitions = userwise_split(indexed.repartition(7, "user_id"))

        assert _pairs(one_partition[0]) == _pairs(many_partitions[0])
        assert _pairs(one_partition[1]) == _pairs(many_partitions[1])


class TestRecommendUnseen:
    @pytest.fixture(scope="class")
    def trained(self, indexed):
        train, _ = userwise_split(indexed)
        train = train.cache()
        model = ALS(
            userCol="user_idx",
            itemCol="song_idx",
            ratingCol="play_count",
            implicitPrefs=True,
            rank=8,
            regParam=0.08,
            alpha=20.0,
            maxIter=5,
            seed=SEED,
            coldStartStrategy="drop",
        ).fit(train)
        return model, train

    def test_no_recommendation_is_an_item_the_user_already_played(self, trained):
        model, train = trained

        recommendations = recommend_unseen(model, train, k=5)

        leaked = recommendations.join(train, ["user_idx", "song_idx"], "inner").count()
        assert leaked == 0

    def test_the_unfiltered_call_does_return_training_items(self, trained):
        """Why the exclusion exists. Implicit ALS scores observed interactions
        highest, so the raw top-k is dominated by items the split already moved
        into training -- items that can never match a held-out row."""
        model, train = trained

        raw = (
            model.recommendForAllUsers(5)
            .select("user_idx", F.explode("recommendations").alias("rec"))
            .select("user_idx", F.col("rec.song_idx").alias("song_idx"))
        )

        assert raw.join(train, ["user_idx", "song_idx"], "inner").count() > 0

    def test_each_user_gets_at_most_k_ranked_rows(self, trained):
        model, train = trained

        recommendations = recommend_unseen(model, train, k=5)

        per_user = recommendations.groupBy("user_idx").count().collect()
        assert per_user, "expected recommendations for at least one user"
        assert all(row["count"] <= 5 for row in per_user)
        assert {r.rank for r in recommendations.select("rank").distinct().collect()} <= {
            1,
            2,
            3,
            4,
            5,
        }
