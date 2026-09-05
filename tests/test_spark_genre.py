"""Exercise real Spark estimators and the held-out population on a tiny fixture."""
import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from src.msd_pipeline import SEED, train_genre_model


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]")
               .appName("genre-regression-test")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "2").getOrCreate())
    yield session
    session.stop()


@pytest.mark.parametrize("model_name", ["lr", "rf", "gbt"])
def test_train_genre_preserves_test_population(spark, tmp_path, model_name):
    audio_path = str(tmp_path / "audio")
    genre_path = str(tmp_path / "genres.tsv")
    output = str(tmp_path / model_name)
    rows = [(f"track-{i}", float(i % 7), float(i % 11)) for i in range(240)]
    spark.createDataFrame(rows, ["track_id", "feature_a", "feature_b"]).write.parquet(audio_path)
    genres = [(f"track-{i}", "Pop_Rock" if i % 4 == 0 else "Other") for i in range(240)]
    (tmp_path / "genres.tsv").write_text("".join(f"{track}\t{genre}\n" for track, genre in genres))
    labelled = spark.read.parquet(audio_path).join(
        spark.read.option("sep", "\t").csv(genre_path).toDF("track_id", "genre"), "track_id"
    )
    _, expected_test = labelled.randomSplit([0.8, 0.2], seed=SEED)
    expected_ids = {r.track_id for r in expected_test.select("track_id").collect()}
    train_genre_model(spark, audio_path, genre_path, "Pop_Rock", model_name, output)
    actual = spark.read.parquet(f"{output}/predictions")
    assert {r.track_id for r in actual.select("track_id").collect()} == expected_ids
    fitted = PipelineModel.load(f"{output}/model")
    assert fitted.stages[0].getInputCols() == ["feature_a", "feature_b"]
    metrics = spark.read.option("header", True).csv(f"{output}/metrics").first()
    assert 0 <= float(metrics.roc_auc) <= 1
