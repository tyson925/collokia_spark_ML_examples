package uy.com.collokia.ml.classification.OneVsRest.naiveBayes

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.machineLearning.printMultiClassMetrics
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.classification.OneVsRest.*
import uy.com.collokia.util.LABELS
import uy.com.collokia.util.NaiveBayesProperties
import uy.com.collokia.util.featureCol
import uy.com.collokia.util.labelIndexCol
import java.io.File
import java.io.Serializable

class OneVsRestNaiveBayes : Serializable {

    companion object {


        @JvmStatic fun main(args: Array<String>) {
            val oneVsRest = OneVsRestNaiveBayes()
            oneVsRest.runOnSpark()
        }
    }

    fun evaluateOneVsRestNaiveBayes(dataset: Dataset<Row>): NaiveBayesProperties {
        val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))
        val indexer = StringIndexerModel.load(LABELS)

        val cachedTrain = train.cache()
        val cachedTest = test.cache()

        //, "bernoulli"
        val evaluations = listOf("multinomial").flatMap { modelType ->
            listOf(1.0, 2.0, 5.0).map { smoothing ->
                val oneVsRest = constructNaiveBayesClassifier(modelType, smoothing)
                val ovrModel = oneVsRest.fit(cachedTrain)

                val metrics = evaluateModel(ovrModel, cachedTest, indexer)
                val properties = NaiveBayesProperties(modelType, smoothing)
                println("${metrics.weightedFMeasure()}\t$properties")
                Tuple2(properties, metrics)

            }
        }

        val sortedEvaluations = evaluations.sortedBy({ metricsData -> metricsData._2.fMeasure(1.0) }).reversed().map { metricsData ->
            Tuple2(metricsData._1, printMultiClassMetrics(metricsData._2))
        }

        println(sortedEvaluations.joinToString("\n"))

        val bestNaiveBayesProperties = sortedEvaluations.first()._1
        val oneVsRest = constructNaiveBayesClassifier(bestNaiveBayesProperties.modelType, bestNaiveBayesProperties.smoothing)

        val ovrModel = oneVsRest.fit(cachedTrain)

        evaluateModelConfusionMTX(ovrModel, cachedTest)

        return bestNaiveBayesProperties
    }

    fun evaluate10Fold(bestNaiveBayesProperties: NaiveBayesProperties, corpus: Dataset<Row>) {

        val naiveBayesClassifier = constructNaiveBayesClassifier(bestNaiveBayesProperties.modelType, bestNaiveBayesProperties.smoothing)

        val pipeline = Pipeline().setStages(arrayOf(naiveBayesClassifier))

        evaluateModel10Fold(pipeline, corpus)
    }

    private fun constructNaiveBayesClassifier(modelType: String, smoothing: Double): OneVsRest {
        val naiveBayes = NaiveBayes().setSmoothing(smoothing).setModelType(modelType)
        val oneVsRest = OneVsRest().setClassifier(naiveBayes)
                .setFeaturesCol(featureCol)
                .setLabelCol(labelIndexCol)
        return oneVsRest
    }


    fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200")
                    .set("es.nodes.discovery", "true")
                    .set("es.nodes.wan.only", "false")

            val jsc = JavaSparkContext(sparkConf)
            val sparkSession = SparkSession.builder().master("local").appName("one vs rest example").orCreate

            val dataset = if (File(corpusFileName).exists()) {
                sparkSession.read().load(corpusFileName)
            } else {
                generateVtm(jsc, sparkSession)
            }
            val bestNaiveBayesProperties = evaluateOneVsRestNaiveBayes(dataset)
            evaluate10Fold(bestNaiveBayesProperties,dataset)

        }
        println("Execution time is ${time.second}")
    }

}