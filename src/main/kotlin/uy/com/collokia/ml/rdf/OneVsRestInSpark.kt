@file:Suppress("UNUSED_VARIABLE")

package uy.com.collokia.ml.rdf

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.*
import org.apache.spark.ml.feature.*
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Serializable
import scala.Tuple2
import uy.com.collokia.common.utils.deleteIfExists
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.machineLearning.printMatrix
import uy.com.collokia.common.utils.machineLearning.printMulticlassMetrics
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.classification.nlp.vtm.constructTagVtmDataPipeline
import uy.com.collokia.ml.classification.nlp.vtm.constructTitleVtmDataPipeline
import uy.com.collokia.ml.classification.nlp.vtm.constructVTMPipeline
import uy.com.collokia.ml.classification.readData.readDzoneFromEs
import uy.com.collokia.util.*
import java.text.DecimalFormat


class OneVsRestInSpark() : Serializable{

    companion object {
        val formatter = DecimalFormat("#0.00")
        val CONTENT_VTM_VOC_SIZE = 1500
        val TITLE_VTM_VOC_SIZE = 500
        val TAG_VTM_VOC_SIZE = 200

        @JvmStatic fun main(args: Array<String>) {
            val ovr = OneVsRestInSpark()
            ovr.runOnSpark()
        }
    }

    fun generateVtm(jsc: JavaSparkContext): Dataset<Row> {
        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").orCreate

        val stopwords = jsc.broadcast(jsc.textFile("./data/stopwords.txt").collect().toTypedArray())

        val corpus = readDzoneFromEs(sparkSession, jsc)

        val vtmDataPipeline = constructVTMPipeline(stopwords.value, CONTENT_VTM_VOC_SIZE)

        println(corpus.count())

        val vtmPipelineModel = vtmDataPipeline.fit(corpus)

        val cvModel = vtmPipelineModel.stages()[4] as CountVectorizerModel
        println("cv model vocabulary: " + cvModel.vocabulary().toList())

        val indexer = vtmPipelineModel.stages()[0] as StringIndexerModel
        if (deleteIfExists(LABELS)) {
            indexer.save(LABELS)
        }

        val parsedCorpus = vtmPipelineModel.transform(corpus).drop("content", "words", "filteredWords", "ngrams", "tfFeatures")

        val vtmTitlePipeline = constructTitleVtmDataPipeline(stopwords.value, TITLE_VTM_VOC_SIZE)

        val vtmTitlePipelineModel = vtmTitlePipeline.fit(parsedCorpus)

        val parsedCorpusTitle = vtmTitlePipelineModel.transform(parsedCorpus).drop("title_words", "filtered_titleWords", "title_ngrams", "tf_titleFeatures")

        parsedCorpusTitle.show(10, false)

        val vtmTagPipeline = constructTagVtmDataPipeline(TAG_VTM_VOC_SIZE)

        val vtmTagPipelineModel = vtmTagPipeline.fit(parsedCorpusTitle)

        val fullParsedCorpus = vtmTagPipelineModel.transform(parsedCorpusTitle).drop("tag_words", "tag_ngrams", "tag_tfFeatures")

        val contentScaler = vtmPipelineModel.stages().last() as Normalizer

        val titleNormalizer = vtmTitlePipelineModel.stages().last() as Normalizer

        val tagNormalizer = vtmTagPipelineModel.stages().last() as Normalizer

        //VectorAssembler().
        val assembler = VectorAssembler().setInputCols(arrayOf(contentScaler.outputCol, titleNormalizer.outputCol, tagNormalizer.outputCol))
                .setOutputCol(featureCol)

        if (deleteIfExists(VTM_PIPELINE)) {
            vtmPipelineModel.save(VTM_PIPELINE)
        }

        val dataset = assembler.transform(fullParsedCorpus)
        return dataset
    }

    fun evaluateOneVsRest(dataset: Dataset<Row>) {

        val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))

        val impurity = "gini"
        val depth = 10
        val bins = 300
        val dt = DecisionTreeClassifier().setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)
        //dt.maxMemoryInMB = 512

        val lr = LogisticRegression().setMaxIter(300).setTol(1E-6).setFitIntercept(true)

        val nb = NaiveBayes()

        val oneVsRest = OneVsRest().setClassifier(dt)
                .setFeaturesCol(featureCol)
                .setLabelCol(labelIndexCol)

        train.show(3)

        val ovrModel = oneVsRest.fit(train)
        //val ovrModel = perceptron.fit(train)

        if (deleteIfExists(OVR_MODEL)) {
            ovrModel.save(OVR_MODEL)
        }
        evaluateModelConfusionMTX(ovrModel, test)
    }

    public fun evaluateModelConfusionMTX(ovrModel: OneVsRestModel, test: Dataset<Row>) {
        val indexer = StringIndexerModel.load(LABELS)

        val metrics = evaluateModel(ovrModel, test, indexer)
        val confusionMatrix = metrics.confusionMatrix()

// compute the false positive rate per label
//        val predictionColSchema = predictions.schema().fields()[0]
//        val numClasses = MetadataUtils.getNumClasses(predictionColSchema).get()

        val fprs = (0..indexer.labels().size - 1).map({ p ->
            val fMeasure = metrics.fMeasure(p.toDouble()) * 100
            val precision = metrics.precision(p.toDouble()) * 100
            val recall = metrics.recall(p.toDouble()) * 100

            Tuple2(indexer.labels()[p], EvaluationMetrics(indexer.labels()[p], fMeasure, precision, recall))
        })

        println(printMatrix(confusionMatrix, indexer.labels().toList()))
        println("overall results:")
        println("FMeasure:\t${formatter.format(metrics.weightedFMeasure() * 100)}\t" +
                "Precision:\t${formatter.format(metrics.weightedPrecision() * 100)}\t" +
                "Recall:\t${formatter.format(metrics.weightedRecall() * 100)}\t" +
                "TP:\t${formatter.format(metrics.weightedTruePositiveRate() * 100)}\n" +
                "Accuracy:\t${formatter.format(metrics.accuracy() * 100)}")


        println(fprs.joinToString("\n"))
    }

    public fun evaluateModel(ovrModel: OneVsRestModel, test: Dataset<Row>, indexer: StringIndexerModel): MulticlassMetrics {
        // Convert indexed labels back to original labels.
        val labelConverter = IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(indexer.labels())


        val predicatePipeline = Pipeline().setStages(arrayOf(ovrModel, labelConverter))

        val predictions = predicatePipeline.fit(test).transform(test)

        predictions.show(3)
        // evaluate the model
        val predictionsAndLabels = predictions.select("prediction", labelIndexCol).toJavaRDD().map({ row ->
            Tuple2(row.getDouble(0) as Any, row.getDouble(1) as Any)
        })

        val metrics = MulticlassMetrics(predictionsAndLabels.rdd())
        return metrics
    }

    fun evaluateOneVsRestDecisionTrees(dataset: Dataset<Row>) {

        val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))
        val indexer = StringIndexerModel.load(LABELS)

        val evaluations =
                listOf("gini", "entropy").flatMap { impurity ->
                    intArrayOf(10, 20, 30).flatMap { depth ->
                        intArrayOf(40, 300).map { bins ->
                            val dt = DecisionTreeClassifier().setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)

                            val oneVsRest = OneVsRest().setClassifier(dt)
                                    .setFeaturesCol(featureCol)
                                    .setLabelCol(labelIndexCol)
                            val ovrModel = oneVsRest.fit(train)

                            val metrics = evaluateModel(ovrModel, test, indexer)
                            val properties = DecisionTreeProperties(impurity, depth, bins)
                            println("${metrics.weightedFMeasure()}\t$properties")
                            Tuple2(properties, metrics)
                        }
                    }
                }

        val sortedEvaluations = evaluations.sortedBy({ metricsData -> metricsData._2.fMeasure(1.0) }).reversed().map { metricsData ->
            Tuple2(metricsData._1, printMulticlassMetrics(metricsData._2))
        }

        println(sortedEvaluations.joinToString("\n"))

        val bestTreeProperties = sortedEvaluations.first()._1
        val bestDecisionTree = DecisionTreeClassifier()
                .setImpurity(bestTreeProperties.impurity)
                .setMaxDepth(bestTreeProperties.maxDepth)
                .setMaxBins(bestTreeProperties.bins)

        val oneVsRest = OneVsRest().setClassifier(bestDecisionTree)
                .setFeaturesCol(featureCol)
                .setLabelCol(labelIndexCol)
        val ovrModel = oneVsRest.fit(train)

        evaluateModelConfusionMTX(ovrModel, test)


    }


    fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200")
                    .set("es.nodes.discovery", "true")
                    .set("es.nodes.wan.only", "false")

            val jsc = JavaSparkContext(sparkConf)
            val dataset = generateVtm(jsc)
            evaluateOneVsRestDecisionTrees(dataset)

        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }

}


