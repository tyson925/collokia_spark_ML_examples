@file:Suppress("UNUSED_VARIABLE")

package uy.com.collokia.ml.rdf

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.machineLearning.evaluateAndPrintPrediction
import uy.com.collokia.common.utils.machineLearning.predicateRandomForest
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.scala.ClassTagger


class RandomForestInSpark(){


    fun evaluate10Fold(data : JavaRDD<LabeledPoint>) : Double{
        val tenFolds = MLUtils.kFold(data.rdd(),10,10, ClassTagger.scalaClassTag(LabeledPoint::class.java))

        val resultsInFmeasure = tenFolds.mapIndexed { i, fold ->
            val (trainData,testData) = fold
            println("number of fold:\t$i")
            val Fmeasure = evaluateSimpleForest(trainData.toJavaRDD(),testData.toJavaRDD(),2)
            Fmeasure
        }

        return resultsInFmeasure.average()
    }

    fun evaluateSimpleForest(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): Double {
        val categoricalFeatureInfo = mapOf<Int, Int>()
        val featureSubsetStrategy = "auto"
        val impurity = "gini"
        val maxDepth = 10
        val maxBin = 300
        val numTree = 50

        val forestModel = buildSimpleForest(trainData,numClasses,categoricalFeatureInfo,featureSubsetStrategy,impurity,maxDepth,maxBin,numTree)

        trainData.unpersist()

        println("evaulate random forest model...")
        val testPrediction = predicateRandomForest(forestModel, cvData)

        return evaluateAndPrintPrediction(numClasses,testPrediction)
    }

    fun buildSimpleForest(trainData: JavaRDD<LabeledPoint>, numClasses: Int, categoricalFeatureInfo : Map<Int, Int>,
                                 featureSubsetStrategy : String, impurity : String, maxDepth : Int, maxBin : Int, numTree : Int): RandomForestModel {
        println("train a reandom forest with $numClasses classes and parameteres featureSubsetStrategy=$featureSubsetStrategy impurity=$impurity," +
                " depth=$maxDepth, bins=$maxBin, numTree=$numTree")
        return RandomForest.trainClassifier(trainData, numClasses, categoricalFeatureInfo, numTree, featureSubsetStrategy, impurity,
                maxDepth, maxBin, numTree)

    }

    fun evaluateForest(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int) {
        val categoricalFeatureInfo = mapOf<Int, Int>()
        val featureSubsetStrategy = "auto"
        val impurities = listOf("gini", "entropy")
        val maxDepths = listOf(10, 20, 30)
        val maxBins = listOf(40, 300)
        val numTrees = listOf(10, 50)

        val evaluations =
                impurities.flatMap { impurity ->
                    maxDepths.flatMap { depth ->
                        maxBins.flatMap { bins ->
                            numTrees.map { numTree ->
                                // Specify value count for categorical features 10, 11
                                val model = buildSimpleForest(trainData, numClasses, categoricalFeatureInfo,  featureSubsetStrategy, impurity, depth, bins, numTree)

                                val trainScores = trainData.map { point ->
                                    val prediction = model.predict(point.features())
                                    Tuple2(prediction as Any, point.label() as Any)
                                }
                                val testScores = cvData.map { point ->
                                    val prediction = model.predict(point.features())
                                    Tuple2(prediction as Any, point.label() as Any)
                                }
                                //val metricsTrain = MulticlassMetrics(trainScores.rdd())
                                val metricsTest = MulticlassMetrics(testScores.rdd())

                                // Return train and CV accuracy
                                Tuple2(Triple(impurity, depth, bins), Triple(metricsTest.weightedFMeasure(), metricsTest.weightedPrecision(), metricsTest.weightedRecall()))
                            }
                        }
                    }
                }

        println(evaluations.sortedBy({ evaluation -> evaluation._2.first }).reversed().joinToString("\n"))

    }


    fun runTenFoldOnSpark() {
        val time = measureTimeInMillis {


            val jsc = getLocalSparkContext("Random forest")

            val corpusInRaw = jsc.textFile("./testData/reuters/json/reuters.json").cache().repartition(8)
            val sparkSession = SparkSession.builder()
                    .master("local")
                    .appName("reuters classification")
                    .orCreate
            //val (trainDF, cvDF, testDF) = corpusInRaw.randomSplit(doubleArrayOf(0.8, 0.1, 0.1))
            //val (trainDF, testDF) = corpusInRaw.randomSplit(doubleArrayOf(0.9, 0.1))
            //val parsedCorpus = docClass.parseCorpus(sparkSession, corpusInRaw, "ship")
            //evaluate10Fold(parsedCorpus)

            closeSpark(jsc)
        }
        println("Execution time is ${time.second}")
    }

    companion object {
        @JvmStatic fun main(args : Array<String>){
            val randomForset = RandomForestInSpark()
            randomForset.runTenFoldOnSpark()
        }
    }
}
