package uy.com.collokia.ml.rdf

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.ml.classification.DocumentClassification
import uy.com.collokia.ml.util.predicateRandomForest
import uy.com.collokia.ml.util.printBinaryClassificationMetrics
import uy.com.collokia.ml.util.printMulticlassMetrics
import uy.com.collokia.scala.ClassTagger
import uy.com.collokia.util.component1
import uy.com.collokia.util.component2
import uy.com.collokia.util.formatterToTimePrint
import uy.com.collokia.util.measureTimeInMillis


public class RandomForestInSpark(){


    public fun evaulate10Fold(data : JavaRDD<LabeledPoint>) : Double{
        val tenFolds = MLUtils.kFold(data.rdd(),10,10, ClassTagger.scalaClassTag(LabeledPoint::class.java))

        val resultsInFmeasure = tenFolds.mapIndexed { i, fold ->
            val (trainData,testData) = fold
            println("number of fold:\t${i}")
            //val Fmeasure = buildDecisionTreeModel(trainData.toJavaRDD(),testData.toJavaRDD(),2)
            val Fmeasure = buildSimpleForest(trainData.toJavaRDD(),testData.toJavaRDD(),2)
            Fmeasure
        }
        return resultsInFmeasure.average()
    }

    public fun buildSimpleForest(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): Double {
        val categoricalFeatureInfo = mapOf<Int, Int>()
        val featureSubsetStrategy = "auto"
        val impurity = "entropy"
        val maxDepth = 10
        val maxBin = 32
        val numTree = 50

        //10 to 4, 11 to 40

        val forestModel = buildSimpleForest(trainData,numClasses,categoricalFeatureInfo,featureSubsetStrategy,impurity,maxDepth,maxBin,numTree)

        trainData.unpersist()

        println("evaulate decision tree model...")
        val FMeasure = if (numClasses == 2) {
            val evaulationBin = BinaryClassificationMetrics(predicateRandomForest(forestModel, cvData),100)
            val evaulation = MulticlassMetrics(predicateRandomForest(forestModel, cvData))
            println(printMulticlassMetrics(evaulation))
            println(printBinaryClassificationMetrics(evaulationBin))
            evaulation.fMeasure(1.0)
        } else {
            val evaulation = MulticlassMetrics(predicateRandomForest(forestModel, cvData))
            println(printMulticlassMetrics(evaulation))
            evaulation.fMeasure(1.0)
        }

        return FMeasure
    }

    public fun buildSimpleForest(trainData: JavaRDD<LabeledPoint>, numClasses: Int, categoricalFeatureInfo : Map<Int, Int>,
                                 featureSubsetStrategy : String, impurity : String, maxDepth : Int, maxBin : Int, numTree : Int): RandomForestModel {
        println("train a reandom forest with ${numClasses} classes and parameteres featureSubsetStrategy=${featureSubsetStrategy} impurity=${impurity}," +
                " depth=${maxDepth}, bins=${maxBin}, numTree=${numTree}")
        return RandomForest.trainClassifier(trainData, numClasses, categoricalFeatureInfo, numTree, featureSubsetStrategy, impurity,
                maxDepth, maxBin, numTree)

    }

    public fun evaluateForest(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int) {
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


    public fun runTenFold() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("RandomForest").setMaster("local[6]")

            val jsc = JavaSparkContext(sparkConf)

            val corpusInRaw = jsc.textFile("./data/reuters/json/reuters.json").cache().repartition(8)
            val sparkSession = SparkSession.builder()
                    .master("local")
                    .appName("reuters classification")
                    .getOrCreate()
            //val (trainDF, cvDF, testDF) = corpusInRaw.randomSplit(doubleArrayOf(0.8, 0.1, 0.1))
            //val (trainDF, testDF) = corpusInRaw.randomSplit(doubleArrayOf(0.9, 0.1))
            val docClass = DocumentClassification()
            val parsedCorpus = docClass.parseCorpus(sparkSession, corpusInRaw, "ship")
            evaulate10Fold(parsedCorpus)

        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }


}


fun main(args: Array<String>) {

}
