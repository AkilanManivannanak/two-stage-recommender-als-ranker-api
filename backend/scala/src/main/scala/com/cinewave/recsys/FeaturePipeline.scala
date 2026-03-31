/**
 * CineWave RecSys — Scala Feature Engineering
 * ============================================
 * Implements ALS (Alternating Least Squares) collaborative filtering and
 * item co-occurrence computation using Spark MLlib.
 *
 * Why Scala here (not Python):
 *   1. MLlib's ALS is natively implemented in Scala — calling it from
 *      PySpark adds a JVM↔Python serialization round-trip for every
 *      iteration.  Running natively in Scala eliminates that overhead.
 *   2. The co-occurrence computation (cartesian join on user events) is
 *      O(n²) in the number of ratings — Scala's Dataset API leverages
 *      Tungsten's off-heap binary encoding, which is 2-4× faster than
 *      the Python UDF path for this shape of computation.
 *   3. The output (ALS factors, co-occurrence scores) is written to
 *      Parquet — Python consumers (Metaflow, the FastAPI serving layer)
 *      read the Parquet files with pyarrow, so the interop is clean.
 *
 * Build & run:
 *   sbt assembly
 *   spark-submit --class com.cinewave.recsys.FeaturePipeline \
 *       target/scala-2.12/cinewave-recsys-assembly-1.0.jar \
 *       --ratings /app/data/ratings.parquet \
 *       --items   /app/data/items.parquet \
 *       --out     /app/artifacts/scala_features
 *
 * Outputs (Parquet):
 *   <out>/user_factors/       — ALS user embeddings  (user_id, features: Array[Float])
 *   <out>/item_factors/       — ALS item embeddings  (item_id, features: Array[Float])
 *   <out>/als_predictions/    — Top-K ALS recs per user (user_id, item_id, rating)
 *   <out>/cooccurrence/       — Item co-occurrence counts (item_a, item_b, count, score)
 *   <out>/item_popularity/    — Per-item popularity stats (item_id, n_ratings, avg_rating, pop_score)
 */

package com.cinewave.recsys

import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel

object FeaturePipeline {

  // ── Hyper-parameters (tunable via --conf spark.cinewave.*) ────────────────

  val ALS_RANK       = 64     // embedding dimension
  val ALS_MAX_ITER   = 20     // training iterations
  val ALS_REG_PARAM  = 0.1    // L2 regularisation
  val ALS_ALPHA      = 40.0   // confidence weight for implicit feedback
  val TOP_K          = 50     // top-K recs per user from ALS
  val COOC_MIN_COUNT = 2      // minimum co-views to keep a co-occurrence pair

  def main(args: Array[String]): Unit = {
    val params = parseArgs(args)

    val spark = SparkSession.builder()
      .appName("CineWave-FeaturePipeline")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.shuffle.partitions", "50")
      .config("spark.ml.recommendation.als.numUserBlocks", "10")
      .config("spark.ml.recommendation.als.numItemBlocks", "10")
      .getOrCreate()

    import spark.implicits._

    println("[CineWave] Loading ratings...")
    val ratings = loadRatings(spark, params("ratings"))
    ratings.persist(StorageLevel.MEMORY_AND_DISK)
    println(s"[CineWave] Loaded ${ratings.count()} ratings")

    println("[CineWave] Loading items...")
    val items = loadItems(spark, params("items"))

    val out = params("out")

    // 1. Item popularity features
    println("[CineWave] Computing item popularity...")
    val popularity = computeItemPopularity(ratings)
    popularity.write.mode("overwrite").parquet(s"$out/item_popularity")
    println(s"[CineWave] ✓ item_popularity written")

    // 2. ALS collaborative filtering
    println(s"[CineWave] Training ALS (rank=$ALS_RANK, iter=$ALS_MAX_ITER)...")
    val (alsModel, userFactors, itemFactors) = trainALS(spark, ratings)
    userFactors.write.mode("overwrite").parquet(s"$out/user_factors")
    itemFactors.write.mode("overwrite").parquet(s"$out/item_factors")
    println(s"[CineWave] ✓ ALS factors written")

    // 3. Top-K ALS recommendations for all users
    println(s"[CineWave] Generating top-$TOP_K ALS recommendations per user...")
    val userRecs = alsModel.recommendForAllUsers(TOP_K)
    // Explode the recommendations array into rows
    val alsPreds = userRecs
      .select($"id".alias("user_id"), explode($"recommendations").alias("rec"))
      .select($"user_id", $"rec.id".alias("item_id"), $"rec.rating".alias("als_score"))
      .orderBy($"user_id", desc("als_score"))
    alsPreds.write.mode("overwrite").parquet(s"$out/als_predictions")
    println(s"[CineWave] ✓ als_predictions written (${alsPreds.count()} rows)")

    // 4. Item co-occurrence (users who watched A also watched B)
    println("[CineWave] Computing item co-occurrence...")
    val cooc = computeCoOccurrence(spark, ratings)
    cooc.write.mode("overwrite").parquet(s"$out/cooccurrence")
    println(s"[CineWave] ✓ cooccurrence written (${cooc.count()} pairs)")

    ratings.unpersist()
    spark.stop()
    println("[CineWave] Pipeline complete.")
  }


  // ── Data loading ──────────────────────────────────────────────────────────

  /**
   * Load ratings.  Supports both Parquet and CSV.
   * Expected schema: user_id INT, item_id INT, rating FLOAT, timestamp LONG
   */
  def loadRatings(spark: SparkSession, path: String): DataFrame = {
    import spark.implicits._
    val raw = if (path.endsWith(".csv")) {
      spark.read.option("header", "true").csv(path)
        .select(
          col("user_id").cast(IntegerType),
          col("item_id").cast(IntegerType),
          col("rating").cast(FloatType),
          col("timestamp").cast(LongType),
        )
    } else {
      spark.read.parquet(path)
        .select(
          col("user_id").cast(IntegerType),
          col("item_id").cast(IntegerType),
          col("rating").cast(FloatType),
          col("timestamp").cast(LongType),
        )
    }
    // ALS requires columns named exactly "user", "item", "rating"
    raw.withColumnRenamed("user_id", "user")
       .withColumnRenamed("item_id", "item")
       .na.drop()
  }

  def loadItems(spark: SparkSession, path: String): DataFrame = {
    if (path.endsWith(".csv")) {
      spark.read.option("header", "true").csv(path)
    } else {
      spark.read.parquet(path)
    }
  }


  // ── ALS Training ─────────────────────────────────────────────────────────

  def trainALS(
      spark: SparkSession,
      ratings: DataFrame,
  ): (ALSModel, DataFrame, DataFrame) = {
    import spark.implicits._

    val als = new ALS()
      .setRank(ALS_RANK)
      .setMaxIter(ALS_MAX_ITER)
      .setRegParam(ALS_REG_PARAM)
      .setAlpha(ALS_ALPHA)
      .setImplicitPrefs(true)         // treat ratings as implicit confidence
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")
      .setColdStartStrategy("drop")   // drop NaN predictions for unseen users/items
      .setNonnegative(false)

    val model = als.fit(ratings)

    // Rename ALS output columns to user_id / item_id for downstream consumers
    val userFactors = model.userFactors
      .withColumnRenamed("id", "user_id")
      .withColumnRenamed("features", "als_vector")

    val itemFactors = model.itemFactors
      .withColumnRenamed("id", "item_id")
      .withColumnRenamed("features", "als_vector")

    (model, userFactors, itemFactors)
  }


  // ── Item co-occurrence ────────────────────────────────────────────────────

  /**
   * For each user, find all pairs of items they rated.
   * Score = Jaccard similarity: |users who rated both| / |users who rated either|.
   *
   * Uses a self-join on user — this is the O(n²) step where Scala/Tungsten
   * is significantly faster than Python UDFs.
   */
  def computeCoOccurrence(spark: SparkSession, ratings: DataFrame): DataFrame = {
    import spark.implicits._

    // Keep one row per (user, item) — deduplicate multiple ratings
    val userItems = ratings
      .select($"user", $"item")
      .distinct()
      .persist(StorageLevel.MEMORY_AND_DISK)

    // Self-join to get all pairs (item_a, item_b) per user
    val pairs = userItems.alias("a")
      .join(userItems.alias("b"), col("a.user") === col("b.user"))
      .where(col("a.item") < col("b.item"))   // canonical ordering, no duplicates
      .select(
        col("a.item").alias("item_a"),
        col("b.item").alias("item_b"),
      )

    // Count co-occurrences
    val coocCounts = pairs
      .groupBy("item_a", "item_b")
      .agg(count("*").alias("cooc_count"))
      .where(col("cooc_count") >= COOC_MIN_COUNT)

    // Item support (how many users rated each item) for Jaccard
    val itemSupport = userItems
      .groupBy("item")
      .agg(countDistinct("user").alias("n_users"))

    // Jaccard = cooc / (support_a + support_b - cooc)
    val withJaccard = coocCounts
      .join(itemSupport.withColumnRenamed("item", "item_a")
                       .withColumnRenamed("n_users", "support_a"), "item_a")
      .join(itemSupport.withColumnRenamed("item", "item_b")
                       .withColumnRenamed("n_users", "support_b"), "item_b")
      .select(
        col("item_a"),
        col("item_b"),
        col("cooc_count"),
        (col("cooc_count").cast(DoubleType) /
         (col("support_a") + col("support_b") - col("cooc_count"))
        ).alias("jaccard_score"),
      )
      .orderBy(desc("jaccard_score"))

    userItems.unpersist()
    withJaccard
  }


  // ── Item popularity ───────────────────────────────────────────────────────

  /**
   * Compute per-item popularity score combining rating count and average rating.
   * pop_score = log(1 + n_ratings) * avg_rating
   */
  def computeItemPopularity(ratings: DataFrame): DataFrame = {
    import org.apache.spark.sql.functions.{log, avg => sparkAvg}

    ratings
      .groupBy("item")
      .agg(
        count("*").alias("n_ratings"),
        sparkAvg("rating").alias("avg_rating"),
      )
      .withColumnRenamed("item", "item_id")
      .withColumn("pop_score",
        log(lit(1.0) + col("n_ratings")) * col("avg_rating"))
      .orderBy(desc("pop_score"))
  }


  // ── Argument parsing ──────────────────────────────────────────────────────

  def parseArgs(args: Array[String]): Map[String, String] = {
    val defaults = Map(
      "ratings" -> "/app/data/ratings.parquet",
      "items"   -> "/app/data/items.parquet",
      "out"     -> "/app/artifacts/scala_features",
    )
    val parsed = args.sliding(2, 2).collect {
      case Array(k, v) if k.startsWith("--") => k.stripPrefix("--") -> v
    }.toMap
    defaults ++ parsed
  }
}
