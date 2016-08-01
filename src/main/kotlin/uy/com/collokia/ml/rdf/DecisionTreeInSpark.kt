package uy.com.collokia.ml.rdf

import org.apache.log4j.BasicConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.model.RandomForestModel
import scala.Tuple2
import uy.com.collokia.ml.util.*
import java.io.Serializable


public class DecisionTreeInSpark() : Serializable {

    public fun buildDecisionTreeModel(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int)  : Double{

        val impurity = "gini"
        val depth = 10
        val bins = 300

        val model = buildDecisionTreeModel(trainData, numClasses, impurity, depth, bins)

        println("evaulate decision tree model...")
        val FMeasure = if (numClasses == 2) {
            val evaulationBin = getBinaryClassificationMetrics(model, cvData)
            val evaulation = getMulticlassMetrics(model, cvData)
            println(printMulticlassMetrics(evaulation))
            println(printBinaryClassificationMetrics(evaulationBin))
            evaulation.fMeasure()
        } else {
            val evaulation = getMulticlassMetrics(model, cvData)
            println(printMulticlassMetrics(evaulation))
            evaulation.fMeasure()
        }
        return FMeasure
    }

    public fun buildDecisionTreeModel(trainData: JavaRDD<LabeledPoint>, numClasses: Int, impurity: String, depth: Int, bins: Int): DecisionTreeModel {
        // Build a simple default DecisionTreeModel
        println("train a decision tree with classes ${numClasses} and parameteres impurity=${impurity}, depth=${depth}, bins=${bins}")
        val model = DecisionTree.trainClassifier(trainData, numClasses, mapOf<Int, Int>(), impurity, depth, bins)
        return model
    }



    public fun evaluate(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, testData: JavaRDD<LabeledPoint>, numClasses: Int) {
        val evaluations =
                listOf("gini", "entropy").flatMap { impurity ->
                    intArrayOf(10, 20, 30).flatMap { depth ->
                        intArrayOf(40, 300).map { bins ->
                            val model = buildDecisionTreeModel(trainData,numClasses,impurity,depth,bins)
                            val metrics = getMulticlassMetrics(model, cvData)
                            Tuple2(Triple(impurity, depth, bins), metrics)
                        }
                    }
                }

        val sortedEvaulations = evaluations.sortedBy({ metricsData -> metricsData._2.fMeasure() }).reversed().map { metricsData ->
            Tuple2(metricsData._1, printMulticlassMetrics(metricsData._2))
        }

        println(sortedEvaulations.joinToString("\n"))

        val bestTreePoperties = sortedEvaulations.first()._1

        val model = DecisionTree.trainClassifier(
                trainData.union(cvData), numClasses, mapOf<Int, Int>(), bestTreePoperties.first, bestTreePoperties.second, bestTreePoperties.third)
        println(getMulticlassMetrics(model, testData).precision())
        println(getMulticlassMetrics(model, trainData.union(cvData)).precision())
    }

    public fun unencodeOneHot(rawData: JavaRDD<String>): JavaRDD<LabeledPoint> {

        val res = rawData.map { line ->
            val values = line.split(',').map({ value -> value.toDouble() })
            // Which of 4 "wilderness" features is 1
            val wilderness = values.slice(IntRange(10, 14)).indexOf(1.0).toDouble()
            // Similarly for following 40 "soil" features
            val soil = values.slice(IntRange(14, 54)).indexOf(1.0).toDouble()
            // Add derived features back to first 10
            val features = values.slice(IntRange(0, 10)).toMutableList()
            features.add(wilderness)
            features.add(soil)
            val featureVector = Vectors.dense(features.toDoubleArray())
            val label = values.last() - 1
            LabeledPoint(label, featureVector)
        }
        return res
    }

    public fun evaluateCategorical(rawData: JavaRDD<String>) {
        val data = unencodeOneHot(rawData)

        val (trainData, cvData, testData) = data.randomSplit(doubleArrayOf(0.8, 0.1, 0.1))
        trainData.cache()
        cvData.cache()
        testData.cache()

        val evaluations =
                listOf("gini", "entropy").flatMap { impurity ->
                    intArrayOf(10, 20, 30).flatMap { depth ->
                        intArrayOf(40, 300).map { bins ->
                            // Specify value count for categorical features 10, 11
                            val model = DecisionTree.trainClassifier(
                                    trainData, 7, mapOf(10 to 4, 11 to 40), impurity, depth, bins)

                            val trainAccuracy = getMulticlassMetrics(model, trainData).precision()
                            val cvAccuracy = getMulticlassMetrics(model, cvData).precision()
                            // Return train and CV accuracy
                            Tuple2(Triple(impurity, depth, bins), Tuple2(trainAccuracy, cvAccuracy))
                        }
                    }
                }


        println(evaluations.sortedBy({ evaluation -> evaluation._2._2 }).reversed().joinToString("\n"))

        val model = DecisionTree.trainClassifier(
                trainData.union(cvData), 7, mapOf(10 to 4, 11 to 40), "entropy", 30, 300)
        println(getMulticlassMetrics(model, testData).precision())

        trainData.unpersist()
        cvData.unpersist()
        testData.unpersist()

    }

    public fun evaluateSimpleForest(rawData: JavaRDD<String>, numClasses: Int) {
        val data = unencodeOneHot(rawData)

        val (trainData, cvData) = data.randomSplit(doubleArrayOf(0.9, 0.1))
        trainData.cache()
        cvData.cache()
        val forest = evaluateSimpleForest(trainData, cvData, numClasses)
        val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
        val vector = Vectors.dense(input.split(',').map({ it -> it.toDouble() }).toDoubleArray())
        println(forest.predict(vector))
    }

    public fun evaluateSimpleForest(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): RandomForestModel {
        val categoricalFeatureInfo = mapOf<Int, Int>()
        val featureSubsetStrategy = "auto"
        val impurity = "entropy"
        val maxDepth = 10
        val maxBin = 32
        val numTree = 50

        //10 to 4, 11 to 40
        val forestModel = RandomForest.trainClassifier(trainData, numClasses, categoricalFeatureInfo, numTree, featureSubsetStrategy, impurity,
                maxDepth, maxBin, numTree)

        trainData.unpersist()

        val predictionsAndLabels = cvData.map({ example ->
            Tuple2(forestModel.predict(example.features()) as Any, example.label() as Any)
        })
        val evaulation = MulticlassMetrics(predictionsAndLabels.rdd())
        printMulticlassMetrics(evaulation)
        return forestModel
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
                                val model = RandomForest.trainClassifier(trainData,
                                        numClasses, categoricalFeatureInfo, numTree, featureSubsetStrategy, impurity, depth, bins, numTree)


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


    public fun runRDF() {
        val sparkConf = SparkConf().setAppName("DecisionTree").setMaster("local[6]")

        val jsc = JavaSparkContext(sparkConf)

        val rawData = jsc.textFile("./data/DT/covtype.data.gz")

        val data = rawData.map { line ->
            val values = line.split(',').map({ value -> value.toDouble() })
            val featureVector = Vectors.dense(values.toDoubleArray())
            val label = values.last() - 1
            LabeledPoint(label, featureVector)
        }

        val (trainData, cvData, testData) = data.randomSplit(doubleArrayOf(0.8, 0.1, 0.1))
        trainData.cache()
        cvData.cache()
        testData.cache()

        //buildDecisionTreeModel(trainData, cvData)
        randomClassifier(trainData, cvData)
        //evaluate(trainData, cvData, testData)
        evaluateCategorical(rawData)
        evaluateSimpleForest(rawData, 7)

        trainData.unpersist()
        cvData.unpersist()
        testData.unpersist()


    }

}

fun main(args: Array<String>) {
    BasicConfigurator.configure()
    val decisionTree = DecisionTreeInSpark()
    decisionTree.runRDF()
}

