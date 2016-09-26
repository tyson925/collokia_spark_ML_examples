package uy.com.collokia.ml.classification

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.machineLearning.convertLabeledPointToArff
import uy.com.collokia.common.utils.machineLearning.saveArff
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.classification.nlp.vtm.constructVTMPipeline
import uy.com.collokia.ml.classification.nlp.vtm.convertDataFrameToLabeledPoints
import uy.com.collokia.ml.classification.nlp.vtm.extractFeaturesFromCorpus
import uy.com.collokia.ml.classification.nlp.vtm.setTfIdfModel
import uy.com.collokia.ml.logreg.LogisticRegressionInSpark
import uy.com.collokia.ml.rdf.DecisionTreeInSpark
import uy.com.collokia.ml.rdf.RandomForestInSpark
import uy.com.collokia.ml.svm.SVMSpark
import uy.com.collokia.util.ClassifierResults
import uy.com.collokia.util.REUTERS_DATA
import java.io.Serializable
import uy.com.collokia.ml.classification.readData.parseCorpus

@Suppress("UNUSED_VARIABLE") class DocumentClassification() : Serializable {

    companion object {
        val MAPPER = jacksonObjectMapper()
        val topCategories = listOf("ship", "grain", "money-fx", "corn", "trade", "crude", "earn", "wheat", "acq", "interest")
        //val topCategories = listOf("earn", "acq")
        val featureCol = "normIdfFeatures"
        val labelIndexCol = "categoryIndex"

        @JvmStatic fun main(args : Array<String>){
            val docClassifier = DocumentClassification()
            //docClassifier.readReutersJson()
            docClassifier.runOnSpark()
        }
    }




    fun tenFoldReutersDataEvaulationWithClassifiers(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile(REUTERS_DATA).cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").orCreate

        val decisionTree = DecisionTreeInSpark()
        val randomForest = RandomForestInSpark()
        val svm = SVMSpark()
        val logReg = LogisticRegressionInSpark()

        val results = topCategories.map { category ->

            val data = convertDataFrameToLabeledPoints(constructVTMData(sparkSession, corpusInRaw, category))

            val dtFMeasure = decisionTree.evaluate10Fold(data)
            val rfFMeasure = randomForest.evaluate10Fold(data)
            val svmFMeasure = svm.evaulate10Fold(data)
            val logRegFMeasure = logReg.evaluate10Fold(data)

            data.unpersist()
            ClassifierResults(category, dtFMeasure, rfFMeasure, svmFMeasure, logRegFMeasure)

        }

        println(results.joinToString("\n"))

    }

    fun tenFoldReutersDataEvaulation(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile(REUTERS_DATA).cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").orCreate

        val decisionTree = DecisionTreeInSpark()

        val logisticRegression = LogisticRegressionInSpark()

        val results = topCategories.map { category ->

            val data = convertDataFrameToLabeledPoints(constructVTMData(sparkSession, corpusInRaw, category))

            val arffData = convertLabeledPointToArff(data)
            saveArff(arffData, "./testData/reuters/arff/${category}.arff")

            //val fMeasure = decisionTree.evaluate10Fold(data)
            val fMeasure = logisticRegression.evaluate10Fold(data)

            //val Fmeasure = 1.0

            data.unpersist()
            Pair(category, fMeasure)

        }.joinToString("\n")

        println(results)
    }

    fun constructVTMData(sparkSession: SparkSession, corpusInRaw: JavaRDD<String>, category: String?): Dataset<Row> {
        val parsedCorpus = parseCorpus(sparkSession, corpusInRaw, category)

        corpusInRaw.unpersist()

        println("category:\t${category}")

        val vtmPipeline = constructVTMPipeline(arrayOf(),2000)
        val data = vtmPipeline.fit(parsedCorpus).transform(parsedCorpus)
        data.show(3)
        return data
    }

    fun reutersDataEvaulation(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile("./testData/reuters/json/reuters.json").cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").orCreate

        val results = topCategories.map { category ->
            val parsedCorpus = extractFeaturesFromCorpus(parseCorpus(sparkSession, corpusInRaw, category))

            val cvModel = CountVectorizer().setInputCol(featureCol).setOutputCol("tfFeatures").setVocabSize(2000).setMinDF(3.0)
                    .fit(parsedCorpus)

            val hashedCorpusDF = cvModel.transform(parsedCorpus)

            parsedCorpus.unpersist()

            val (hashedTrainDF, hashedTestDF) = hashedCorpusDF.randomSplit(doubleArrayOf(0.9, 0.1))

            val idfModel = setTfIdfModel(hashedTrainDF)

            val trainTfIdfDF = idfModel.transform(hashedTrainDF)

            hashedTrainDF.unpersist()

            val normalizer = Normalizer().setInputCol(idfModel.outputCol).setOutputCol("normIdfFeatures").setP(1.0)
            //val normalizer = Normalizer().setInputCol("features").setOutputCol("normIdfFeatures").setP(1.0)

            val normTrainTfIdfDF = normalizer.transform(trainTfIdfDF)

            trainTfIdfDF.unpersist()

            val trainData = convertDataFrameToLabeledPoints(normTrainTfIdfDF).cache()

            normTrainTfIdfDF.unpersist()

            val testTfIdfDF = idfModel.transform(hashedTestDF)
            val normTestTfIdfDF = normalizer.transform(testTfIdfDF)
            testTfIdfDF.unpersist()

            val testData = convertDataFrameToLabeledPoints(normTestTfIdfDF).cache()

            val dt = DecisionTreeInSpark()

            val Fmeasure = dt.evaluateDecisionTreeModel(trainData, testData, 2)

            trainData.unpersist()
            testData.unpersist()
            Pair(category, Fmeasure)

        }.joinToString("\n")

        println(results)
    }

    fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost").set("es.port", "9200").set("es.nodes.discovery", "false").set("es.nodes.wan.only", "true")

            val jsc = JavaSparkContext(sparkConf)

            val sparkSession = SparkSession.builder().master("local").appName("reuters classification").orCreate


            val test = JavaEsSpark.esRDD(jsc, "dzone_data/DocumentRow")
            //val test = readDzoneFromEs(sparkSession,jsc)

            //val test = sparkSession.read().format("org.elasticsearch.spark.sql").option("es.field.read.as.array.exclude","labels").load("dzone_data/DocumentRow")
// inspect the data
            println(test.count())

            //val testData = parseCorpus(jsc)

            //reutersDataEvaulation(jsc)
            //tenFoldReutersDataEvaulation(jsc)
            //tenFoldReutersDataEvaulationWithClassifiers(jsc)
            //val dt = DecisionTreeInSpark()
            //dt.evaluateSimpleForest(testData)
            //dt.evaluate10FoldDF(trainData, testData, testData, 10)
            //println(dt.classProbabilities(trainData).joinToString("\n"))
            //dt.buildDecisionTreeModel(trainData,testData,10)
        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }




}


