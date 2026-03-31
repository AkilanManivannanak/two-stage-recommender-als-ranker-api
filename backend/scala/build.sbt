// CineWave RecSys — Scala Feature Pipeline
// Build with: sbt assembly
// Then run: spark-submit --class com.cinewave.recsys.FeaturePipeline target/scala-2.12/cinewave-recsys-assembly-1.0.jar

name         := "cinewave-recsys"
version      := "1.0"
scalaVersion := "2.12.18"

// Spark 3.5 — matches the PySpark version used in spark_features.py
val sparkVersion = "3.5.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"   % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"    % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib"  % sparkVersion % "provided",
)

// sbt-assembly: fat jar for spark-submit
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x                             => MergeStrategy.first
}

assembly / mainClass := Some("com.cinewave.recsys.FeaturePipeline")

// Output jar name
assembly / assemblyJarName := "cinewave-recsys-assembly-1.0.jar"

// Compiler options
scalacOptions ++= Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-unchecked",
)
