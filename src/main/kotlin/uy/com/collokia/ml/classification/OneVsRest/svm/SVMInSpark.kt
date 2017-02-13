package uy.com.collokia.ml.classification.OneVsRest.svm

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.machineLearning.evaluateAndPrintPrediction
import uy.com.collokia.common.utils.machineLearning.predicateMLModel
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.classification.OneVsRest.corpusFileName
import uy.com.collokia.ml.classification.OneVsRest.generateVtm
import uy.com.collokia.ml.classification.nlp.vtm.convertDataFrameToLabeledPoints
import uy.com.collokia.scala.ClassTagger
import java.io.File
import java.io.Serializable

class SVMInSpark() : Serializable {

    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val svm = SVMInSpark()
            svm.runOnSpark()
        }
    }


    fun evaluate10Fold(data: JavaRDD<LabeledPoint>): Double {
        val tenFolds = MLUtils.kFold(data.rdd(), 10, 10, ClassTagger.scalaClassTag(LabeledPoint::class.java))

        val resultsInFmeasure = tenFolds.mapIndexed { i, fold ->
            val (trainData, testData) = fold
            println("number of fold:\t$i")
            val Fmeasure = evaluateSVM(trainData.toJavaRDD(), testData.toJavaRDD(), 2)
            Fmeasure
        }
        return resultsInFmeasure.average()
    }

    fun buildSimpleSVM(trainData: JavaRDD<LabeledPoint>, numClasses: Int): SVMModel {
        println(trainData.take(3).joinToString("\n"))

// Run training algorithm to build the model
        val numIterations = 300
        println("Build SVM with $numClasses classes...")
        val model = SVMWithSGD.train(trainData.cache().rdd(), numIterations,numClasses.toDouble(),numClasses.toDouble())
        return model
    }

    fun evaluateSVM(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): Double {

        val model = buildSimpleSVM(trainData, numClasses)

        trainData.unpersist()

        println("evaulate decision tree model...")

        val testPrediction = predicateMLModel(model, cvData)
        val FMeasure = evaluateAndPrintPrediction(numClasses, testPrediction)
        return FMeasure
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

            val datasetRDD = convertDataFrameToLabeledPoints(dataset)
            evaluate10Fold(datasetRDD)


        }
        println("Execution time is ${time.second}")
    }


}