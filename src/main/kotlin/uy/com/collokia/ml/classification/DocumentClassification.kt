package uy.com.collokia.ml.classification

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.feature.*
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import scala.Tuple2
import uy.com.collokia.ml.rdf.DecisionTreeInSpark
import java.io.File
import java.io.Serializable
import javax.xml.crypto.Data

public data class ReutersDocument(val title: String?, var body: String?, val date: String,
                                  val topics: List<String>?, val places: List<String>?, val organisations: List<String>?, val id: Int) : Serializable

public class DocumentClassification() : Serializable {

    var hashingTF: HashingTF? = null
    var idfModel: IDFModel? = null

    companion object {
        val MAPPER = jacksonObjectMapper()
        val topCategories = listOf("ship", "grain", "money-fx", "corn", "trade", "crude", "earn", "wheat", "acq", "interest")
        //val topCategories = listOf("earn", "acq")
    }


    public fun parseCorpus(sqlContext: SQLContext, corpusInRaw: JavaRDD<String>): DataFrame {

        val corpus = corpusInRaw.map { line ->
            val doc = MAPPER.readValue(line, ReutersDocument::class.java)
            doc
        }/*.filter { doc ->
            doc.topics != null || doc.topics?.intersect(topCategories)?.isNotEmpty() ?: false
        }*/

        corpusInRaw.unpersist()

        println("corpus size: " + corpus.count())

        val corpusRow = corpus.flatMap { doc ->
            val topics = doc.topics?.intersect(topCategories) ?: listOf<String>()
            topics.map { topic ->
                RowFactory.create(topic, doc.body!!)
            }
        }

        corpus.unpersist()

        val schema = StructType(arrayOf<StructField>(StructField("category", DataTypes.StringType, true, Metadata.empty()), StructField(
                "rawText", DataTypes.StringType, true, Metadata.empty())))

        val textDataFrame = sqlContext.createDataFrame(corpusRow, schema)

        corpusRow.unpersist()

        val indexer = StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(textDataFrame)
        println(indexer.labels().joinToString("\t"))

        val indexedTextDataFrame = indexer.transform(textDataFrame)

        val tokenizer = Tokenizer().setInputCol("rawText").setOutputCol("words")
        val wordsDataFrame = tokenizer.transform(indexedTextDataFrame)

        val remover = StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")

        val filteredWordsDataFrame = remover.transform(wordsDataFrame)

        val ngramTransformer = NGram().setInputCol("filteredWords").setOutputCol("ngrams").setN(4)

        val ngramsDataFrame = ngramTransformer.transform(filteredWordsDataFrame)

        return ngramsDataFrame
    }

    public fun createTfCorpus(ngramsDataFrame: DataFrame): DataFrame {
        val cvModel = CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(3)
                .setMinDF(2.0)
                .fit(ngramsDataFrame)

        val cvDataDrame = cvModel.transform(ngramsDataFrame)
        return cvDataDrame
    }

    public fun createTfIdfCorpus(jsc : JavaSparkContext) {
        val corpusInRaw = jsc.textFile("./data/reuters/json/reuters.json").cache().repartition(8)
        val sqlContext = SQLContext(jsc)
        val (trainDF, cvDF, testDF) = corpusInRaw.randomSplit(doubleArrayOf(0.8, 0.1, 0.1))

        val parsedtrainDF = parseCorpus(sqlContext,trainDF)
        val parsedCvDF = parseCorpus(sqlContext,cvDF)
        val parsedTestDF = parseCorpus(sqlContext,testDF)

        val hashingTF = hasingTf()
        val hashedTrainDF = hashingTF.transform(parsedtrainDF)
        val idfModel = setTfIdfModel(hashedTrainDF)

        val trainTfIdfDF = idfModel.transform(hashedTrainDF).cache()
        val trainData = convertDataFrameToLabeledPoints(trainTfIdfDF)

        val hashedCVDF = hashingTF.transform(parsedCvDF)
        val hashedTestDF = hasingTf().transform(parsedTestDF)

        val cvTfIdfDF = idfModel.transform(hashedCVDF).cache()
        val testTfIdfDF = idfModel.transform(hashedTestDF).cache()
        val cvData = convertDataFrameToLabeledPoints(cvTfIdfDF)
        val testData = convertDataFrameToLabeledPoints(testTfIdfDF)

        val dt = DecisionTreeInSpark()
        //dt.evaluateSimpleForest(trainData,cvData,10)
        //dt.evaluate(trainData, cvData, testData, 10)
        dt.evaluateForest(trainData,cvData,10)

    }


    public fun convertDataFrameToLabeledPoints(data: DataFrame): JavaRDD<LabeledPoint> {
        val converter = IndexToString()
                .setInputCol("categoryIndex")
                .setOutputCol("originalCategory")
        val converted = converter.transform(data)


        val featureData = converted.select("idfFeatures", "categoryIndex", "originalCategory")

        for (r in featureData.take(3)) {

            val features = r.getAs<SparseVector>(0)
            val label = r.getDouble(1)
            val original = r.getString(2)
            System.out.println(features)
            println(label)
            println(original)
        }

        val labeledDataPoints = featureData.toJavaRDD().map({ feature ->
            val features = feature.getAs<SparseVector>(0)
            val label = feature.getDouble(1)
//            println(label)
            LabeledPoint(label, features)
        })

        println("number of data: " + labeledDataPoints.count())

        val labelStat = featureData.select("originalCategory").javaRDD().mapToPair { label ->
            Tuple2(label.getString(0), 1L)
        }.reduceByKey { a, b -> a + b }

        println(labelStat.collectAsMap())

        return labeledDataPoints
    }

    public fun hasingTf(): HashingTF {
        val numFeatures = 2000

        val hashingTF = HashingTF().setInputCol("ngrams")
                .setOutputCol("tfFeatures")
                .setNumFeatures(numFeatures)
        return hashingTF
    }

    public fun setTfIdfModel(corpus: DataFrame): IDFModel {
        val idf = IDF().setInputCol("tfFeatures").setOutputCol("idfFeatures").setMinDocFreq(3)

        val idfModel = idf.fit(corpus)

        return idfModel
    }


    public fun readJson() {
        var index = 0
        File("./data/reuters/json/reuters.json").bufferedWriter().use { writer ->
            val time = File("./data/reuters/").listFiles().filter { file -> file.name.endsWith(".json") }.forEach { file ->
                val jsons = file.readLines().joinToString("\n").split("},").toMutableList()
                jsons[0] = jsons[0].substring(1)
                jsons[jsons.lastIndex] = jsons[jsons.lastIndex].substringBeforeLast("]")

                jsons.forEach { json ->
                    //println(cleanJson(json))
                    val reutersDoc = MAPPER.readValue(cleanJson(json), ReutersDocument::class.java)

                    reutersDoc.body?.let { body ->
                        println(index++)
                        reutersDoc.body = reutersDoc.body?.replace("\n", "")
                        if (reutersDoc.topics != null || reutersDoc.topics?.intersect(topCategories)?.isNotEmpty() ?: false) {
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
        val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")

        val jsc = JavaSparkContext(sparkConf)

        //val data = parseCorpus(jsc)

        createTfIdfCorpus(jsc)
        //val dt = DecisionTreeInSpark()
        //dt.evaluateSimpleForest(data)
        //dt.evaluate(trainData, cvData, testData, 10)
        //println(dt.classProbabilities(trainData).joinToString("\n"))
        //dt.simpleDecisionTree(trainData,testData,10)

    }

}

fun main(args: Array<String>) {

    val docClassifier = DocumentClassification()
    //docClassifier.readJson()
    docClassifier.runOnSpark()
}

