"""Real-Spark checks on the ALS split and on what the model is allowed to recommend."""
import pytest

pytest.importorskip("pyspark")
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F

from src.msd_pipeline import SEED, recommend_unseen, userwise_split


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]")
               .appName("als-regression-test")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "2").getOrCreate())
    yield session
    session.stop()


@pytest.fixture(scope="module")
def indexed(spark):
    rows = [(f"u{user}", f"s{(user * 7 + item) % 40}", float(1 + item % 3))
            for user in range(30) for item in range(12)]
    frame = spark.createDataFrame(rows, ["user_id", "song_id", "play_count"]).distinct()
    indexers = [StringIndexer(inputCol="user_id", outputCol="user_idx"),
                StringIndexer(inputCol="song_id", outputCol="song_idx")]
    return Pipeline(stages=indexers).fit(frame).transform(frame).select(
        "user_id", "song_id", F.col("user_idx").cast("int"), F.col("song_idx").cast("int"), "play_count"
    ).cache()


def _pairs(frame):
    return {(row.user_idx, row.song_idx) for row in frame.select("user_idx", "song_idx").collect()}


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

    def test_split_is_stable_across_separate_evaluations(self, indexed):
        """The train and test frames are separate filters over one plan, each
        evaluated by its own action. With a nondeterministic ordering key the
        two evaluations can disagree and rows leak across the boundary."""
        first_train, first_test = userwise_split(indexed)
        first = (_pairs(first_train), _pairs(first_test))

        second_train, second_test = userwise_split(indexed)

        assert (_pairs(second_train), _pairs(second_test)) == first


class TestRecommendUnseen:
    @pytest.fixture(scope="class")
    def trained(self, indexed):
        train, _ = userwise_split(indexed)
        train = train.cache()
        model = ALS(userCol="user_idx", itemCol="song_idx", ratingCol="play_count",
                    implicitPrefs=True, rank=8, regParam=0.08, alpha=20.0, maxIter=5,
                    seed=SEED, coldStartStrategy="drop").fit(train)
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

        raw = model.recommendForAllUsers(5).select(
            "user_idx", F.explode("recommendations").alias("rec")
        ).select("user_idx", F.col("rec.song_idx").alias("song_idx"))

        assert raw.join(train, ["user_idx", "song_idx"], "inner").count() > 0

    def test_each_user_gets_at_most_k_ranked_rows(self, trained):
        model, train = trained

        recommendations = recommend_unseen(model, train, k=5)

        per_user = recommendations.groupBy("user_idx").count().collect()
        assert per_user, "expected recommendations for at least one user"
        assert all(row["count"] <= 5 for row in per_user)
        assert {r.rank for r in recommendations.select("rank").distinct().collect()} <= {1, 2, 3, 4, 5}
