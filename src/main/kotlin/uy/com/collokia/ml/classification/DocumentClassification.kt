package uy.com.collokia.ml.classification

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.feature.*
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import java.io.File
import java.io.Serializable

public data class ReutersDocument(val title: String?, var body: String?, val date: String,
                                  val topics: List<String>?, val places: List<String>?, val organisations: List<String>?, val id: Int) : Serializable

public class DocumentClassification() : Serializable {

    companion object {
        val MAPPER = jacksonObjectMapper()
    }


    public fun createCorpus(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile("./data/reuters/json/reuters.json")

        val sqlContext = SQLContext(jsc)

        val corpus = corpusInRaw.map { line ->
            val doc = MAPPER.readValue(line, ReutersDocument::class.java)
            doc
        }.filter { doc ->
            doc.topics != null
        }.map { doc ->
            RowFactory.create(doc.topics!!, doc.body!!)
        }

        val schema = StructType(arrayOf<StructField>(StructField("labels", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty()), StructField(
                "rawText", DataTypes.StringType, false, Metadata.empty())))

        val textDataFrame = sqlContext.createDataFrame(corpus, schema)
        val tokenizer = Tokenizer().setInputCol("rawText").setOutputCol("words")
        val wordsDataFrame = tokenizer.transform(textDataFrame)

        val remover = StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")

        val filteredWordsDataFrame = remover.transform(wordsDataFrame)

        val ngramTransformer = NGram().setInputCol("filteredWords").setOutputCol("ngrams")

        val ngramsDataFrame = ngramTransformer.transform(filteredWordsDataFrame)

        val numFeatures = 100000

        val hashingTF = HashingTF().setInputCol("ngrams")
                .setOutputCol("tfFeatures")
                .setNumFeatures(numFeatures)

        val tfFeatures = hashingTF.transform(ngramsDataFrame)

        val idf = IDF().setInputCol("tfFeatures").setOutputCol("idfFeatures")

        val idfModel = idf.fit(tfFeatures)

        val rescaledData = idfModel.transform(tfFeatures)

        for (r in rescaledData.select("idfFeatures", "labels").take(3)) {
            val features = r.getAs<SparseVector>(0)
            val label = r.getList<String>(1)
            System.out.println(features)
            println(label)
        }
    }


    public fun readJson() {

        val time = File("./data/reuters/").listFiles().filter { file -> file.name.endsWith(".json") }.forEach { file ->
            val jsons = file.readLines().joinToString("\n").split("},").toMutableList()
            jsons[0] = jsons[0].substring(1)
            jsons[jsons.lastIndex] = jsons[jsons.lastIndex].substringBeforeLast("]")

            File("./data/reuters/json/reuters.json").bufferedWriter().use { writer ->
                jsons.forEach { json ->
                    //println(cleanJson(json))
                    val reutersDoc = MAPPER.readValue(cleanJson(json), ReutersDocument::class.java)

                    reutersDoc.body?.let { body ->
                        reutersDoc.body = reutersDoc.body?.replace("\n", "")
                        writer.write(MAPPER.writeValueAsString(reutersDoc) + "\n")
                    }

                }

            }

        }
    }


    private fun cleanJson(json: String): String {
        return json.replace("\n", " ").replace("\\\n", " ").replace("\u0003", "") + "}"
    }

    public fun runOnSpark() {
        val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[6]")

        val jsc = JavaSparkContext(sparkConf)

        createCorpus(jsc)

    }

}

fun main(args: Array<String>) {

    val docClassifier = DocumentClassification()
    //docClassifier.readJson()
    docClassifier.runOnSpark()
}

