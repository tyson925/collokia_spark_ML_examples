package uy.com.collokia.ml.classification

//import org.apache.spark.mllib.linalg.SparseVector
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.ml.regression.LabeledPoint
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.ml.logreg.LinearRegressionInSpark
import uy.com.collokia.ml.rdf.DecisionTreeInSpark
import uy.com.collokia.ml.rdf.RandomForestInSpark
import uy.com.collokia.ml.svm.SVMSpark
import uy.com.collokia.ml.util.convertLabeledPointToArff
import uy.com.collokia.ml.util.saveArff
import uy.com.collokia.util.formatterToTimePrint
import uy.com.collokia.util.measureTimeInMillis
import java.io.File
import java.io.Serializable

public data class ReutersDocument(val title: String?, var body: String?, val date: String,
                                  val topics: List<String>?, val places: List<String>?, val organisations: List<String>?, val id: Int) : Serializable

//required "var" according to `Encoders.bean`
public data class ReutersRow(var category: String, var content: String) : Serializable

public data class ClassifierResults(val category : String,val decisiontTree : Double, val randomForest : Double, val svm : Double,val logReg : Double) : Serializable

public class DocumentClassification() : Serializable {

    companion object {
        val MAPPER = jacksonObjectMapper()
        val topCategories = listOf("ship", "grain", "money-fx", "corn", "trade", "crude", "earn", "wheat", "acq", "interest")
        //val topCategories = listOf("earn", "acq")
        val featureOutput = "filteredWords"
    }

    public fun parseCorpus(sparkSession: SparkSession, corpusInRaw: JavaRDD<String>, subTopic: String?): Dataset<ReutersRow> {

        val corpusRow = corpusInRaw.map { line ->
            val doc = MAPPER.readValue(line, ReutersDocument::class.java)
            val topics = doc.topics?.intersect(topCategories) ?: listOf<String>()
            val content = doc.body + (doc.title ?: "")

            val row = if (topics.contains(subTopic)) {
                ReutersRow(subTopic!!, content)
            } else {
                ReutersRow("other", content)
            }
            row
        }

        corpusInRaw.unpersist()

        println("corpus size: " + corpusRow.count())

        val reutersEncoder = Encoders.bean(ReutersRow::class.java)

        val textDataFrame = sparkSession.createDataset(corpusRow.rdd(), reutersEncoder)

        corpusRow.unpersist()

        return textDataFrame
    }


    public fun exractFeaturesFromCorpus(textDataFrame: Dataset<ReutersRow>): Dataset<Row> {

        val indexer = StringIndexer().setInputCol(ReutersRow::category.name).setOutputCol("categoryIndex").fit(textDataFrame)
        println(indexer.labels().joinToString("\t"))

        val indexedTextDataFrame = indexer.transform(textDataFrame)

        val tokenizer = Tokenizer().setInputCol(ReutersRow::content.name).setOutputCol("words")
        val wordsDataFrame = tokenizer.transform(indexedTextDataFrame)

        val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(featureOutput)

        val filteredWordsDataFrame = remover.transform(wordsDataFrame)

        //val ngramTransformer = NGram().setInputCol("filteredWords").setOutputCol("ngrams").setN(4)
        val ngramTransformer = NGram().setInputCol("words").setOutputCol("ngrams").setN(4)

        //       val ngramsDataFrame = ngramTransformer.transform(filteredWordsDataFrame)
        val ngramsDataFrame = ngramTransformer.transform(wordsDataFrame)

        //return ngramsDataFrame
        return filteredWordsDataFrame
    }

    public fun reutersDataEvaulation(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile("./testData/reuters/json/reuters.json").cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val results = topCategories.map { category ->
            val parsedCorpus = exractFeaturesFromCorpus(parseCorpus(sparkSession, corpusInRaw, category))

            val cvModel = CountVectorizer().setInputCol(featureOutput).setOutputCol("tfFeatures").setVocabSize(2000).setMinDF(3.0)
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

            val Fmeasure = dt.evaulateDecisionTreeModel(trainData, testData, 2)

            trainData.unpersist()
            testData.unpersist()
            Pair(category, Fmeasure)

        }.joinToString("\n")

        println(results)
    }

    public fun tenFoldReutersDataEvaulationWithClassifiers(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile("./data/reuters/json/reuters.json").cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val decisionTree = DecisionTreeInSpark()
        val randomForest = RandomForestInSpark()
        val svm = SVMSpark()
        val logReg = LinearRegressionInSpark()

        val results = topCategories.map { category ->

            val data = constructVTMData(sparkSession,corpusInRaw,category)

            val dtFMeasure = decisionTree.evaulate10Fold(data)
            val rfFMeasure = randomForest.evaulate10Fold(data)
            val svmFMeasure = svm.evaulate10Fold(data)
            val logRegFMeasure = logReg.evaulate10Fold(data)

            data.unpersist()
            ClassifierResults(category, dtFMeasure,rfFMeasure,svmFMeasure,logRegFMeasure)

        }

        println(results.joinToString("\n"))

    }

    public fun tenFoldReutersDataEvaulation(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile("./data/reuters/json/reuters.json").cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val decisionTree = DecisionTreeInSpark()

        val results = topCategories.map { category ->

            val data = constructVTMData(sparkSession,corpusInRaw,category)

            val arffData = convertLabeledPointToArff(data)
            saveArff(arffData, "./testData/reuters/arff/${category}.arff")

            val fMeasure = decisionTree.evaulate10Fold(data)

            //val Fmeasure = 1.0

            data.unpersist()
            Pair(category, fMeasure)

        }.joinToString("\n")

        println(results)
    }

    public fun constructVTMData(sparkSession : SparkSession,corpusInRaw : JavaRDD<String>,category : String) : JavaRDD<LabeledPoint>{
        val parsedCorpus = exractFeaturesFromCorpus(parseCorpus(sparkSession, corpusInRaw, category))

        corpusInRaw.unpersist()

        println("category:\t${category}")

        val cvModel = CountVectorizer().setInputCol(featureOutput).setOutputCol("tfFeatures").setVocabSize(2000).setMinDF(2.0)
                .fit(parsedCorpus)

        val hashedCorpusDF = cvModel.transform(parsedCorpus)

        parsedCorpus.unpersist()

        val idfModel = setTfIdfModel(hashedCorpusDF)

        val trainTfIdfDF = idfModel.transform(hashedCorpusDF)

        hashedCorpusDF.unpersist()

        val normalizer = Normalizer().setInputCol(idfModel.outputCol).setOutputCol("normIdfFeatures").setP(1.0)

        val normTrainTfIdfDF = normalizer.transform(trainTfIdfDF)

        trainTfIdfDF.unpersist()

        val data = convertDataFrameToLabeledPoints(normTrainTfIdfDF).cache()

        normTrainTfIdfDF.unpersist()

        return data
    }


    public fun convertDataFrameToLabeledPoints(data: Dataset<Row>): JavaRDD<org.apache.spark.mllib.regression.LabeledPoint> {
        val converter = IndexToString()
                .setInputCol("categoryIndex")
                .setOutputCol("originalCategory")
        val converted = converter.transform(data)


        val featureData = converted.select("normIdfFeatures", "categoryIndex", "originalCategory")

        val labeledDataPoints = featureData.toJavaRDD().map({ feature ->
            val features = feature.getAs<SparseVector>(0)
            val label = feature.getDouble(1)
//            println(label)
            org.apache.spark.mllib.regression.LabeledPoint(label, org.apache.spark.mllib.linalg.SparseVector(features.size(), features.indices(), features.values()))
        })

        println("number of testData: " + labeledDataPoints.count())

        val labelStat = featureData.select("originalCategory").javaRDD().mapToPair { label ->
            Tuple2(label.getString(0), 1L)
        }.reduceByKey { a, b -> a + b }

        println(labelStat.collectAsMap())

        return labeledDataPoints
    }

    public fun setTfIdfModel(corpus: Dataset<Row>): IDFModel {
        val idf = IDF().setInputCol("tfFeatures").setOutputCol("idfFeatures").setMinDocFreq(3)

        val idfModel = idf.fit(corpus)

        return idfModel
    }


    public fun readJson() {
        var index = 0
        File("./testData/reuters/json/reuters.json").bufferedWriter().use { writer ->
            val time = File("./testData/reuters/").listFiles().filter { file -> file.name.endsWith(".json") }.forEach { file ->
                val jsons = file.readLines().joinToString("\n").split("},").toMutableList()
                jsons[0] = jsons[0].substring(1)
                jsons[jsons.lastIndex] = jsons[jsons.lastIndex].substringBeforeLast("]")

                jsons.forEach { json ->
                    //println(cleanJson(json))
                    val reutersDoc = MAPPER.readValue(cleanJson(json), ReutersDocument::class.java)

                    reutersDoc.body?.let { body ->
                        println(index++)
                        reutersDoc.body = reutersDoc.body?.replace("\n", "")
                        if (reutersDoc.topics != null && reutersDoc.topics.intersect(topCategories).isNotEmpty()) {
                            writer.write(MAPPER.writeValueAsString(reutersDoc) + "\n")
                        }
                    }
                }
            }
        }
    }


    private fun cleanJson(json: String): String {
        return json.replace("\n", " ").replace("\\\n", " ").replace("\u0003", "") + "}"
    }

    public fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")

            val jsc = JavaSparkContext(sparkConf)

            //val testData = parseCorpus(jsc)

            //reutersDataEvaulation(jsc)
            tenFoldReutersDataEvaulation(jsc)
            //val dt = DecisionTreeInSpark()
            //dt.evaulateSimpleForest(testData)
            //dt.evaluate(trainData, testData, testData, 10)
            //println(dt.classProbabilities(trainData).joinToString("\n"))
            //dt.buildDecisionTreeModel(trainData,testData,10)
        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }

}

fun main(args: Array<String>) {

    val docClassifier = DocumentClassification()
    //docClassifier.readJson()
    docClassifier.runOnSpark()

}

