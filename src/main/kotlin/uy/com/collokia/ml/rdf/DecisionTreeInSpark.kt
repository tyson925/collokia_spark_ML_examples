package uy.com.collokia.ml.rdf

import org.apache.log4j.BasicConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import scala.Tuple2
import scala.collection.Iterable
import uy.com.collokia.ml.classification.DocumentClassification
import uy.com.collokia.ml.util.*
import java.io.Serializable


public class DecisionTreeInSpark() : Serializable {


    public fun evaulate10Fold(dataInRaw: DataFrame) {

        val indexer = StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(dataInRaw)
        println("labels:\t ${indexer.labels().joinToString("\t")}")

        val tokenizer = Tokenizer().setInputCol("rawText").setOutputCol("words")

        val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol("filteredWords")

        val ngramTransformer = NGram().setInputCol(remover.outputCol).setOutputCol("ngrams").setN(4)

        val hashingTf = HashingTF().setInputCol(ngramTransformer.outputCol).setOutputCol("hashedFeatures").setNumFeatures(2000)

        val idfModel = IDF().setInputCol(hashingTf.outputCol).setOutputCol("idfFeatures").setMinDocFreq(3)

        val normalizer = Normalizer().setInputCol(idfModel.outputCol).setOutputCol("normIdfFeatures").setP(1.0)

        val impurity = "gini"
        val depth = 10
        val bins = 300
        //println("train a decision tree with classes and parameteres impurity=${impurity}, depth=${depth}, bins=${bins}")
        //val decisionTree = DecisionTreeClassifier().setFeaturesCol(normalizer.outputCol).setLabelCol(indexer.outputCol).setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)

        val decisionTree = DecisionTreeClassifier().setFeaturesCol(normalizer.outputCol).setLabelCol(indexer.outputCol).setImpurity(impurity)

        val pipeline = Pipeline().setStages(arrayOf(indexer, tokenizer, remover, ngramTransformer, hashingTf, idfModel, normalizer, decisionTree))

        // ParamGridBuilder was applied to construct a grid of parameters to search over.
        val paramGrid = ParamGridBuilder()
                //.addGrid(decisionTree.impurity(), arrayOf("gini", "entropy").iterator() as Iterable<String>)
                .addGrid(decisionTree.maxDepth(), intArrayOf(10, 20, 30))
                .addGrid(decisionTree.maxBins(), intArrayOf(40, 300))
                .build()


        val evaulator = MulticlassClassificationEvaluator().setLabelCol("categoryIndex").setPredictionCol("prediction")
                // "f1", "precision", "recall", "weightedPrecision", "weightedRecall"
                .setMetricName("f1")

        // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
// This will allow us to jointly choose parameters for all Pipeline stages.
// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
// Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
// is areaUnderROC.
        val cv = CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaulator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10)

        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(dataInRaw)

        val model = cvModel.bestModel()

        val avgMetrics = cvModel.avgMetrics()
        val paramsToScore = cvModel.estimatorParamMaps.mapIndexed { i, paramMap ->
            Tuple2(paramMap, avgMetrics[i])
        }.sortedByDescending { stat -> stat._2 }

        println(paramsToScore.joinToString("\n"))


    }

    public fun buildDecisionTreeModel(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): Double {

        val impurity = "gini"
        val depth = 10
        val bins = 300

        println("train a decision tree with classes ${numClasses} and parameteres impurity=${impurity}, depth=${depth}, bins=${bins}")
        val model = buildDecisionTreeModel(trainData, numClasses, impurity, depth, bins)

        println("evaulate decision tree model...")
        val FMeasure = if (numClasses == 2) {
            val evaulationBin = getBinaryClassificationMetrics(model, cvData)
            val evaulation = getMulticlassMetrics(model, cvData)
            println(printMulticlassMetrics(evaulation))
            println(printBinaryClassificationMetrics(evaulationBin))
            evaulation.fMeasure(1.0)
        } else {
            val evaulation = getMulticlassMetrics(model, cvData)
            println(printMulticlassMetrics(evaulation))
            evaulation.fMeasure(1.0)
        }
        return FMeasure
    }

    public fun buildDecisionTreeModel(trainData: JavaRDD<LabeledPoint>, numClasses: Int, impurity: String, depth: Int, bins: Int): DecisionTreeModel {
        // Build a simple default DecisionTreeModel
        println("train a decision tree with classes ${numClasses} and parameteres impurity=${impurity}, depth=${depth}, bins=${bins}")
        val model = DecisionTree.trainClassifier(trainData, numClasses, mapOf<Int, Int>(), impurity, depth, bins)
        return model
    }


    public fun evaluate(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, testData: JavaRDD<LabeledPoint>, numClasses: Int): Double {
        val evaluations =
                listOf("gini", "entropy").flatMap { impurity ->
                    intArrayOf(10, 20, 30).flatMap { depth ->
                        intArrayOf(40, 300).map { bins ->
                            val model = buildDecisionTreeModel(trainData, numClasses, impurity, depth, bins)
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
        return getMulticlassMetrics(model, testData).fMeasure(1.0)
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

    public fun runTenFold() {
        val sparkConf = SparkConf().setAppName("DecisionTree").setMaster("local[6]")

        val jsc = JavaSparkContext(sparkConf)

        val corpusInRaw = jsc.textFile("./data/reuters/json/reuters.json").cache().repartition(8)
        val sqlContext = SQLContext(jsc)
        //val (trainDF, cvDF, testDF) = corpusInRaw.randomSplit(doubleArrayOf(0.8, 0.1, 0.1))
        //val (trainDF, testDF) = corpusInRaw.randomSplit(doubleArrayOf(0.9, 0.1))
val docClass = DocumentClassification()
        val parsedCorpus = docClass.parseCorpus(sqlContext, corpusInRaw, "acq")
        evaulate10Fold(parsedCorpus)

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
    //decisionTree.runRDF()
    decisionTree.runTenFold()
}

