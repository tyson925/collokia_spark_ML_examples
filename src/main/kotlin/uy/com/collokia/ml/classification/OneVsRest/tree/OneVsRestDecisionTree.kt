@file:Suppress("unused")

package uy.com.collokia.ml.classification.OneVsRest.tree

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.machineLearning.printMultiClassMetrics
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.classification.OneVsRest.*
import uy.com.collokia.util.DecisionTreeProperties
import uy.com.collokia.util.LABELS
import uy.com.collokia.util.featureCol
import uy.com.collokia.util.labelIndexCol
import java.io.File
import java.io.Serializable

class OneVsRestDecisionTree() : Serializable {

    companion object {


        @JvmStatic fun main(args : Array<String>){
            val oneVsRest = OneVsRestDecisionTree()
            oneVsRest.runOnSpark()
        }
    }

    fun evaluateOneVsRestDecisionTrees(dataset: Dataset<Row>) : DecisionTreeProperties{

        val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))
        val indexer = StringIndexerModel.load(LABELS)

        val cachedTrain = train.cache()
        val cachedTest = test.cache()
        val evaluations =
                //listOf("gini", "entropy").flatMap { impurity ->
                listOf("entropy").flatMap { impurity ->
                    intArrayOf(10, 20, 30).flatMap { depth ->
                        intArrayOf(40, 300).map { bins ->

                            val oneVsRest = constructDecisionTreeClassifier(impurity,depth,bins)
                            val ovrModel = oneVsRest.fit(cachedTrain)

                            val metrics = evaluateModel(ovrModel, cachedTest, indexer)
                            val properties = DecisionTreeProperties(impurity, depth, bins)
                            println("${metrics.weightedFMeasure()}\t$properties")
                            Tuple2(properties, metrics)
                        }
                    }
                }

        val sortedEvaluations = evaluations.sortedBy({ metricsData -> metricsData._2.fMeasure(1.0) }).reversed().map { metricsData ->
            Tuple2(metricsData._1, printMultiClassMetrics(metricsData._2))
        }

        println(sortedEvaluations.joinToString("\n"))

        val bestTreeProperties = sortedEvaluations.first()._1
        val oneVsRest = constructDecisionTreeClassifier(bestTreeProperties.impurity,
                bestTreeProperties.maxDepth,
                bestTreeProperties.bins)

        val ovrModel = oneVsRest.fit(cachedTrain)

        evaluateModelConfusionMTX(ovrModel, cachedTest)

        return bestTreeProperties

    }

    fun evaluate10Fold(bestProperties : DecisionTreeProperties, corpus : Dataset<Row>){

        val oneVsRestClassifier = constructDecisionTreeClassifier(bestProperties.impurity,
                bestProperties.maxDepth,bestProperties.bins)
        val pipeline = Pipeline().setStages(arrayOf(oneVsRestClassifier))

        evaluateModel10Fold(pipeline, corpus)
    }

    private fun constructDecisionTreeClassifier(impurity : String,depth : Int,bins : Int) : OneVsRest{

        val decisionTree = DecisionTreeClassifier().setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)

        val oneVsRest = OneVsRest().setClassifier(decisionTree)
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

            dataset.show(3,false)
            //val bestProperties = evaluateOneVsRestDecisionTrees(dataset)

            val bestProperties = DecisionTreeProperties("entropy",30,300)
            evaluate10Fold(bestProperties,dataset)

        }
        println("Execution time is ${time.second}")
    }
}
