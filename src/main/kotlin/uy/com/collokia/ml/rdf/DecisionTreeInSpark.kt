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
import java.io.Serializable


public class DecisionTreeInSpark() : Serializable {

    public fun simpleDecisionTree(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses : Int) {

        // Build a simple default DecisionTreeModel
        val model = DecisionTree.trainClassifier(trainData, numClasses, mapOf<Int, Int>(), "gini", 4, 100)

        val metrics = getMetrics(model, cvData)

        println(metrics.confusionMatrix())
        println(metrics.precision())

        val res = (0..numClasses-1).map({ category ->
            "${category}\tprecision:\t${metrics.precision(category.toDouble())}, recall:\t${metrics.recall(category.toDouble())}, F-measure:\t${metrics.fMeasure(category.toDouble())}"

        }).joinToString("\n")
        println(res)
        println("precision:${metrics.precision()}, recall:\t${metrics.recall()}, F-measure:\t${metrics.fMeasure()}")
    }

    public fun getMetrics(model: DecisionTreeModel, data: JavaRDD<LabeledPoint>): MulticlassMetrics {

        val predictionsAndLabels = data.map { instance ->

            Tuple2(model.predict(instance.features()) as Any, instance.label() as Any)
        }
        return MulticlassMetrics(predictionsAndLabels.rdd())
    }

    public fun randomClassifier(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>) {
        val trainPriorProbabilities = classProbabilities(trainData)
        val cvPriorProbabilities = classProbabilities(cvData)
        val accuracy = trainPriorProbabilities.zip(cvPriorProbabilities).map { probapilities ->
            val (trainProb, cvProb) = probapilities
            trainProb * cvProb
        }.sum()
        println(accuracy)
    }

    public fun classProbabilities(data: JavaRDD<LabeledPoint>): DoubleArray {
        // Count (category,count) in data
        val countsByCategory = data.map({ instance -> instance.label() }).countByValue()
        // order counts by category and extract counts
        val counts = countsByCategory.toSortedMap().map { it -> it.value }

        return counts.map({ it -> it.toDouble() / counts.sum() }).toDoubleArray()
    }

    public fun evaluate(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, testData: JavaRDD<LabeledPoint>,numClasses : Int) {
        val evaluations =
                listOf("gini", "entropy").flatMap { impurity ->
                    intArrayOf(1, 20).flatMap { depth ->
                        intArrayOf(10, 300).map { bins ->
                            val model = DecisionTree.trainClassifier(
                                    trainData, numClasses, mapOf<Int, Int>(), impurity, depth, bins)
                            val accuracy = getMetrics(model, cvData).precision()
                            Tuple2(Triple(impurity, depth, bins), accuracy)
                        }
                    }
                }

        println(evaluations.sortedBy({ it -> it._2 }).reversed().joinToString("\n"))

        val model = DecisionTree.trainClassifier(
                trainData.union(cvData), numClasses, mapOf<Int, Int>(), "entropy", 20, 300)
        println(getMetrics(model, testData).precision())
        println(getMetrics(model, trainData.union(cvData)).precision())
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
                            val trainAccuracy = getMetrics(model, trainData).precision()
                            val cvAccuracy = getMetrics(model, cvData).precision()
                            // Return train and CV accuracy
                            Tuple2(Triple(impurity, depth, bins), Tuple2(trainAccuracy, cvAccuracy))
                        }
                    }
                }


        println(evaluations.sortedBy({ evaluation -> evaluation._2._2 }).reversed().joinToString("\n"))

        val model = DecisionTree.trainClassifier(
                trainData.union(cvData), 7, mapOf(10 to 4, 11 to 40), "entropy", 30, 300)
        println(getMetrics(model, testData).precision())

        trainData.unpersist()
        cvData.unpersist()
        testData.unpersist()

    }

    public fun evaluateForest(rawData: JavaRDD<String>) {
        val data = unencodeOneHot(rawData)

        val forest = evaluateForest(data)
        val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
        val vector = Vectors.dense(input.split(',').map({ it -> it.toDouble() }).toDoubleArray())
        println(forest.predict(vector))
    }

    public fun evaluateForest(data: JavaRDD<LabeledPoint>): RandomForestModel {
        val (trainData, cvData) = data.randomSplit(doubleArrayOf(0.9, 0.1))
        trainData.cache()
        cvData.cache()

        val forestModel = RandomForest.trainClassifier(trainData, 7, mapOf(10 to 4, 11 to 40), 20, "auto", "entropy", 30, 300, 12)

        trainData.unpersist()

        val predictionsAndLabels = cvData.map({ example ->
            Tuple2(forestModel.predict(example.features()) as Any, example.label() as Any)
        })
        println(MulticlassMetrics(predictionsAndLabels.rdd()).precision())

        return forestModel
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

        //simpleDecisionTree(trainData, cvData)
        //randomClassifier(trainData, cvData)
        //evaluate(trainData, cvData, testData)
        evaluateCategorical(rawData)
        evaluateForest(rawData)

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

