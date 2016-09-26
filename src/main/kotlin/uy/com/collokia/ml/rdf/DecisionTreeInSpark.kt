@file:Suppress("UNUSED_VARIABLE")

package uy.com.collokia.ml.rdf

import org.apache.log4j.BasicConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.machineLearning.evaulateAndPrintPrediction
import uy.com.collokia.common.utils.machineLearning.predicateDecisionTree
import uy.com.collokia.common.utils.machineLearning.printMulticlassMetrics
import uy.com.collokia.common.utils.machineLearning.randomClassifier
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.scala.ClassTagger
import uy.com.collokia.util.REUTERS_DATA
import uy.com.collokia.util.featureCol
import uy.com.collokia.util.labelIndexCol
import java.io.Serializable
import uy.com.collokia.ml.classification.nlp.vtm.*
import uy.com.collokia.util.readData.parseCorpus
import uy.com.collokia.util.DecisionTreeProperties


class DecisionTreeInSpark() : Serializable {

    fun evaluate10FoldDF(sparkSession : SparkSession, dataInRaw: JavaRDD<String>, category : String) {

        val vtmPipeline = constructVTMPipeline(arrayOf(),2000)

        val data = parseCorpus(sparkSession,dataInRaw,category)
        val impurity = "gini"
        val depth = 10
        val bins = 300
        //println("train a decision tree with classes and parameteres impurity=${impurity}, depth=${depth}, bins=${bins}")
        //val decisionTree = DecisionTreeClassifier().setFeaturesCol(normalizer.setOutputCol).setLabelCol(indexer.setOutputCol).setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)

        val vtm = vtmPipeline.fit(data).transform(data)

        val decisionTree = DecisionTreeClassifier().setFeaturesCol(featureCol).setLabelCol(labelIndexCol).setImpurity(impurity)

        val pipeline = Pipeline().setStages(arrayOf(decisionTree))

        // ParamGridBuilder was applied to construct a grid of parameters to search over.
        val paramGrid = ParamGridBuilder()
                //.addGrid(decisionTree.impurity(), arrayOf("gini", "entropy").iterator() as Iterable<String>)
                //.addGrid(decisionTree.maxDepth(), intArrayOf(10, 20, 30))
                //.addGrid(decisionTree.maxBins(), intArrayOf(40, 300))
                .build()


        val evaluator = MulticlassClassificationEvaluator().setLabelCol(labelIndexCol).setPredictionCol("prediction")

        // "f1", "precision", "recall", "weightedPrecision", "weightedRecall"
        //evaulator.set("metricName", "f1(1.0)")
        evaluator.metricName = "f1"

        // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
// This will allow us to jointly choose parameters for all Pipeline stages.
// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
// Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
// is areaUnderROC.
        val cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(10)

        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(vtm)

        val model = cvModel.bestModel()

        val avgMetrics = cvModel.avgMetrics()

        val paramsToScore = cvModel.estimatorParamMaps.mapIndexed { i, paramMap ->
            Tuple2(paramMap, avgMetrics[i])
        }.sortedByDescending { stat -> stat._2 }

        println(paramsToScore.joinToString("\n"))


    }

    fun evaluate10Fold(data: JavaRDD<LabeledPoint>): Double {
        val tenFolds = MLUtils.kFold(data.rdd(), 10, 10, ClassTagger.scalaClassTag(LabeledPoint::class.java))

        val resultsInFmeasure = tenFolds.mapIndexed { i, fold ->
            val (trainData, testData) = fold
            println("number of fold:\t${i}")
            val Fmeasure = evaluateDecisionTreeModel(trainData.toJavaRDD(), testData.toJavaRDD(), 2)
            trainData.unpersist(false)
            testData.unpersist(false)
            Fmeasure
        }
        return resultsInFmeasure.average()
    }


    fun evaluateDecisionTreeModel(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): Double {

        val impurity = "gini"
        val depth = 10
        val bins = 300

        //println("train a decision tree with classes ${numClasses} and parameteres impurity=${impurity}, depth=${depth}, bins=${bins}")
        val model = buildDecisionTreeModel(trainData, numClasses, impurity, depth, bins)

        //val dt = DecisionTree()

        println("evaulate decision tree model...")
        val testPrediction = predicateDecisionTree(model, cvData)

        return evaulateAndPrintPrediction(numClasses,testPrediction)
    }

    fun buildDecisionTreeModel(trainData: JavaRDD<LabeledPoint>, numClasses: Int, impurity: String, depth: Int, bins: Int): DecisionTreeModel {
        // Build a simple default DecisionTreeModel
        println("train a decision tree with classes ${numClasses} and parameteres impurity=${impurity}, depth=${depth}, bins=${bins}")

        //DecisionTreeClassifier().
        val model = DecisionTree.trainClassifier(trainData, numClasses, mapOf<Int, Int>(), impurity, depth, bins)
        return model
    }


    fun evaluateDecisionTree(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, testData: JavaRDD<LabeledPoint>, numClasses: Int): Double {
        val evaluations =
                listOf("gini", "entropy").flatMap { impurity ->
                    intArrayOf(10, 20, 30).flatMap { depth ->
                        intArrayOf(40, 300).map { bins ->
                            val model = buildDecisionTreeModel(trainData, numClasses, impurity, depth, bins)
                            val metrics = MulticlassMetrics(predicateDecisionTree(model, cvData))
                            Tuple2(DecisionTreeProperties(impurity, depth, bins), metrics)
                        }
                    }
                }

        val sortedEvaluations = evaluations.sortedBy({ metricsData -> metricsData._2.fMeasure(1.0) }).reversed().map { metricsData ->
            Tuple2(metricsData._1, printMulticlassMetrics(metricsData._2))
        }

        println(sortedEvaluations.joinToString("\n"))

        val bestTreePoperties = sortedEvaluations.first()._1

        val model = DecisionTree.trainClassifier(
                trainData.union(cvData), numClasses, mapOf<Int, Int>(), bestTreePoperties.impurity, bestTreePoperties.maxDepth, bestTreePoperties.bins)
        val testEval = MulticlassMetrics(predicateDecisionTree(model, testData))
        println("test eval:\t${testEval.accuracy()}")
        println("train + testData:\t${MulticlassMetrics(predicateDecisionTree(model, trainData.union(cvData))).accuracy()}")
        return testEval.fMeasure(1.0)
    }

    fun unencodeOneHot(rawData: JavaRDD<String>): JavaRDD<LabeledPoint> {

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

    fun evaluateCategorical(rawData: JavaRDD<String>) {
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

                            val trainAccuracy = MulticlassMetrics(predicateDecisionTree(model, trainData)).accuracy()
                            val cvAccuracy = MulticlassMetrics(predicateDecisionTree(model, cvData)).accuracy()
                            // Return train and CV accuracy
                            Tuple2(Triple(impurity, depth, bins), Tuple2(trainAccuracy, cvAccuracy))
                        }
                    }
                }


        println(evaluations.sortedBy({ evaluation -> evaluation._2._2 }).reversed().joinToString("\n"))

        val model = DecisionTree.trainClassifier(
                trainData.union(cvData), 7, mapOf(10 to 4, 11 to 40), "entropy", 30, 300)
        println(MulticlassMetrics(predicateDecisionTree(model, testData)).weightedPrecision())
//BinaryClassificationMetrics(predictionsAndLabels.rdd(),100)
        trainData.unpersist()
        cvData.unpersist()
        testData.unpersist()

    }


    fun runTenFold() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("DecisionTree").setMaster("local[6]")

            val jsc = JavaSparkContext(sparkConf)

            val corpusInRaw = jsc.textFile(REUTERS_DATA).cache().repartition(8)
            val sparkSession = SparkSession.builder()
                    .master("local")
                    .appName("reuters classification")
                    .orCreate

            evaluate10FoldDF(sparkSession, corpusInRaw, "earn")

        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }


    fun runRDF() {
        val time = measureTimeInMillis {
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

            //buildDecisionTreeModel(trainData, testData)
            randomClassifier(trainData, cvData)
            //evaluate(trainData, testData, testData)
            evaluateCategorical(rawData)


            trainData.unpersist()
            cvData.unpersist()
            testData.unpersist()
        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }

    companion object {
        @JvmStatic fun main(args: Array<String>){
            BasicConfigurator.configure()
            val decisionTree = DecisionTreeInSpark()
            decisionTree.runRDF()
            //decisionTree.runTenFold()
        }
    }
}
