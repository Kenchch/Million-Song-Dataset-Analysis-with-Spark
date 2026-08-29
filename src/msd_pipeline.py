"""Spark pipelines for Million Song Dataset genre classification and ALS recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession, Window, functions as F, types as T


SEED = 420


def get_spark() -> SparkSession:
    return SparkSession.builder.appName("million-song-analysis").getOrCreate()


def schema_from_attributes(spark: SparkSession, attributes_path: str) -> T.StructType:
    """Construct a feature-file schema from the accompanying attributes CSV."""
    mapping = {"string": T.StringType(), "real": T.DoubleType(), "numeric": T.DoubleType(), "float": T.DoubleType()}
    attributes = spark.read.option("header", "false").csv(attributes_path).select(
        F.col("_c0").alias("name"), F.lower(F.trim(F.col("_c1"))).alias("source_type")
    )
    fields = []
    for row in attributes.collect():
        if row.source_type not in mapping:
            raise ValueError(f"Unsupported attribute type {row.source_type!r} for {row.name!r}")
        fields.append(T.StructField(row.name, mapping[row.source_type], True))
    return T.StructType(fields)


def load_audio(spark: SparkSession, attributes_path: str, features_path: str) -> DataFrame:
    return spark.read.schema(schema_from_attributes(spark, attributes_path)).option("header", "false").csv(features_path)


def nonzero_numeric_columns(frame: DataFrame) -> list[str]:
    numeric = [field.name for field in frame.schema.fields if isinstance(field.dataType, T.NumericType)]
    maxima = frame.agg(*[F.max(F.abs(F.col(column))).alias(column) for column in numeric]).first().asDict()
    return [column for column in numeric if maxima[column] not in (None, 0)]


def train_genre_model(spark: SparkSession, audio_path: str, genre_path: str, positive_genre: str, model_name: str, output: str) -> None:
    audio = spark.read.parquet(audio_path)
    genre = spark.read.option("sep", "\t").option("header", "false").csv(genre_path).select(
        F.col("_c0").alias("track_id"), F.col("_c1").alias("genre")
    )
    labelled = audio.join(genre, "track_id", "inner").withColumn(
        "label", F.when(F.col("genre") == positive_genre, F.lit(1.0)).otherwise(F.lit(0.0))
    )
    positives = labelled.filter("label = 1")
    negatives = labelled.filter("label = 0")
    negative_fraction = min(1.0, positives.count() / max(negatives.count(), 1))
    balanced = positives.unionByName(negatives.sample(False, negative_fraction, SEED))
    features = nonzero_numeric_columns(balanced)
    if model_name == "lr":
        assembler = VectorAssembler(inputCols=features, outputCol="features_raw", handleInvalid="skip")
        stages = [assembler, StandardScaler(inputCol="features_raw", outputCol="features", withMean=False), LogisticRegression(maxIter=100, seed=SEED)]
    elif model_name == "rf":
        assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
        stages = [assembler, RandomForestClassifier(numTrees=200, maxDepth=12, seed=SEED)]
    elif model_name == "gbt":
        assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
        stages = [assembler, GBTClassifier(maxIter=100, maxDepth=6, seed=SEED)]
    else:
        raise ValueError("model must be one of: lr, rf, gbt")
    train, test = balanced.randomSplit([0.8, 0.2], seed=SEED)
    fitted = Pipeline(stages=stages).fit(train)
    predictions = fitted.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)
    Path(output).mkdir(parents=True, exist_ok=True)
    fitted.write().overwrite().save(f"{output}/model")
    spark.createDataFrame([(model_name, positive_genre, len(features), auc)], ["model", "positive_genre", "feature_count", "roc_auc"]).write.mode("overwrite").option("header", True).csv(f"{output}/metrics")
    predictions.select("track_id", "genre", "label", "prediction", "probability").write.mode("overwrite").parquet(f"{output}/predictions")


def clean_triplets(spark: SparkSession, triplets_path: str, min_user_items: int, min_song_users: int) -> DataFrame:
    interactions = spark.read.option("sep", "\t").option("header", "false").csv(triplets_path).select(
        F.col("_c0").alias("user_id"), F.col("_c1").alias("song_id"), F.col("_c2").cast("float").alias("play_count")
    )
    active_users = interactions.groupBy("user_id").agg(F.countDistinct("song_id").alias("item_count")).filter(F.col("item_count") >= min_user_items)
    active_songs = interactions.groupBy("song_id").agg(F.countDistinct("user_id").alias("user_count")).filter(F.col("user_count") >= min_song_users)
    return interactions.join(active_users.select("user_id"), "user_id").join(active_songs.select("song_id"), "song_id")


def userwise_split(interactions: DataFrame, train_fraction: float = 0.8) -> tuple[DataFrame, DataFrame]:
    ranked = interactions.withColumn("_random", F.rand(SEED)).withColumn(
        "_rank", F.row_number().over(Window.partitionBy("user_id").orderBy("_random"))
    ).withColumn("_count", F.count("*").over(Window.partitionBy("user_id")))
    cutoff = F.ceil(F.col("_count") * train_fraction)
    return ranked.filter(F.col("_rank") <= cutoff).drop("_random", "_rank", "_count"), ranked.filter(F.col("_rank") > cutoff).drop("_random", "_rank", "_count")


def train_als(spark: SparkSession, triplets_path: str, output: str, min_user_items: int, min_song_users: int) -> None:
    interactions = clean_triplets(spark, triplets_path, min_user_items, min_song_users)
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
    song_indexer = StringIndexer(inputCol="song_id", outputCol="song_idx", handleInvalid="skip")
    indexed = Pipeline(stages=[user_indexer, song_indexer]).fit(interactions).transform(interactions).select(
        F.col("user_id"), F.col("song_id"), F.col("user_idx").cast("int"), F.col("song_idx").cast("int"), "play_count"
    )
    train, test = userwise_split(indexed)
    model = ALS(userCol="user_idx", itemCol="song_idx", ratingCol="play_count", implicitPrefs=True, rank=64, regParam=0.08, alpha=20.0, maxIter=15, seed=SEED, coldStartStrategy="drop").fit(train)
    Path(output).mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(f"{output}/model")
    model.recommendForAllUsers(10).write.mode("overwrite").parquet(f"{output}/recommendations")
    test.write.mode("overwrite").parquet(f"{output}/test_interactions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("audio-schema")
    schema.add_argument("--attributes", required=True)
    schema.add_argument("--features", required=True)
    genre = commands.add_parser("train-genre")
    genre.add_argument("--audio", required=True)
    genre.add_argument("--genres", required=True)
    genre.add_argument("--positive-genre", required=True)
    genre.add_argument("--model", choices=("lr", "rf", "gbt"), default="gbt")
    genre.add_argument("--output", required=True)
    als = commands.add_parser("train-als")
    als.add_argument("--triplets", required=True)
    als.add_argument("--output", required=True)
    als.add_argument("--min-user-items", type=int, default=20)
    als.add_argument("--min-song-users", type=int, default=20)
    args = parser.parse_args()
    spark = get_spark()
    try:
        if args.command == "audio-schema":
            frame = load_audio(spark, args.attributes, args.features)
            frame.printSchema()
            frame.show(5, truncate=False)
        elif args.command == "train-genre":
            train_genre_model(spark, args.audio, args.genres, args.positive_genre, args.model, args.output)
        else:
            train_als(spark, args.triplets, args.output, args.min_user_items, args.min_song_users)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
