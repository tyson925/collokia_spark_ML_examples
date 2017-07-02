package uy.com.collokia.ml.classification.OneVsRest.logReg

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
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
import uy.com.collokia.util.LogisticRegressionProperties
import uy.com.collokia.util.featureCol
import uy.com.collokia.util.labelIndexCol
import java.io.File
import java.io.Serializable

class OneVsRestLogReg : Serializable {

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            val oneVsRest = OneVsRestLogReg()
            oneVsRest.runOnSpark()
        }
    }


    fun evaluateOneVsRestLogReg(dataset: Dataset<Row>): LogisticRegressionProperties {
        val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))
        val indexer = StringIndexerModel.load(LABELS)

        val cachedTrain = train.cache()
        val cachedTest = test.cache()

        val evaluations =
                //listOf(100, 200, 300, 600).flatMap { numIterations ->
                listOf(600).flatMap { numIterations ->
                    //listOf(1E-5, 1E-6, 1E-7).flatMap { stepSize ->
                    listOf(1E-5).flatMap { stepSize ->
                        listOf(0.01,0.1).flatMap { regressionParam ->
                            //listOf(0.01, 0.1, 0.3, 0.5, 0.8, 1.0).flatMap { elasticNetParam ->
                            listOf(0.0).flatMap { elasticNetParam ->
                            //listOf(true, false).flatMap { fitIntercept ->
                            listOf(true).flatMap { fitIntercept ->
                                listOf(true).map { standardization ->


                                        val oneVsRest = constructLogRegClassifier(numIterations, stepSize, fitIntercept, standardization,
                                                regressionParam, elasticNetParam)
                                        val ovrModel = oneVsRest.fit(cachedTrain)
                                        val metrics = evaluateModel(ovrModel, cachedTest, indexer)

                                        val properties = LogisticRegressionProperties(numIterations, stepSize, fitIntercept, standardization,
                                                regressionParam, elasticNetParam)
                                        println("${metrics.weightedFMeasure()}\t$properties")
                                        Tuple2(properties, metrics)
                                    }
                                }
                            }
                        }
                    }
                }

        val sortedEvaluations = evaluations.sortedBy({ metricsData -> metricsData._2.fMeasure(1.0) }).reversed().map { metricsData ->
            Tuple2(metricsData._1, printMultiClassMetrics(metricsData._2))
        }

        println(sortedEvaluations.joinToString("\n"))

        val bestLogRegProperties = sortedEvaluations.first()._1

        val oneVsRest = constructLogRegClassifier(bestLogRegProperties.numIterations,
                bestLogRegProperties.stepSize,
                bestLogRegProperties.fitIntercept,
                bestLogRegProperties.standardization,
                bestLogRegProperties.regParam)

        val ovrModel = oneVsRest.fit(cachedTrain)

        evaluateModelConfusionMTX(ovrModel, cachedTest)

        return bestLogRegProperties
    }

    fun evaluate10Fold(bestProperties: LogisticRegressionProperties, corpus: Dataset<Row>) {

        val logRegClassifier = constructLogRegClassifier(bestProperties.numIterations,
                bestProperties.stepSize,
                bestProperties.fitIntercept,
                bestProperties.standardization,
                bestProperties.regParam)

        val pipeline = Pipeline().setStages(arrayOf(logRegClassifier))

        evaluateModel10Fold(pipeline, corpus)

    }

    private fun constructLogRegClassifier(numIterations: Int,
                                          stepSize: Double,
                                          fitIntercept: Boolean,
                                          standardization: Boolean,
                                          regParam: Double,
                                          elasticNetParam: Double = 0.0): OneVsRest {

        val logisticRegression = LogisticRegression()
                .setMaxIter(numIterations)
                .setTol(stepSize)
                .setFitIntercept(fitIntercept)
                .setStandardization(standardization)
                .setRegParam(regParam)

        if (elasticNetParam != 0.0) {
            logisticRegression.elasticNetParam = elasticNetParam
        }

        val oneVsRest = OneVsRest().setClassifier(logisticRegression)
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
                val corpus = generateVtm(jsc, sparkSession)
                corpus.write().save(corpusFileName.substringBeforeLast("."))
                corpus
            }

            //val bestProperties = evaluateOneVsRestLogReg(dataset)
            val bestProperties = LogisticRegressionProperties(600, 1.0E-7, true, false, 0.01,0.1)
            dataset.show(10,false)
            evaluateOneVsRestLogReg(dataset)
            //evaluate10Fold(bestProperties, dataset)

        }
        println("Execution time is ${time.second}")
    }

}
