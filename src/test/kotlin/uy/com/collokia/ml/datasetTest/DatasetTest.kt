package uy.com.collokia.ml.datasetTest

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.feature.*
import org.apache.spark.sql.SparkSession
import uy.com.collokia.common.utils.deleteIfExists
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.classification.nlp.vtm.constructTagVtmDataPipeline
import uy.com.collokia.ml.classification.nlp.vtm.constructTitleVtmDataPipeline
import uy.com.collokia.ml.classification.nlp.vtm.constructVTMPipeline
import uy.com.collokia.util.DocumentRow
import uy.com.collokia.util.LABELS
import uy.com.collokia.util.featureCol
import java.io.Serializable

class DatasetTest() : Serializable {

    fun datasetAssamblerTest(jsc: JavaSparkContext) {
        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").orCreate

        val stopwords = jsc.broadcast(jsc.textFile("./data/stopwords.txt").collect().toTypedArray())

        val corpus = sparkSession.createDataFrame(listOf(DocumentRow("bigdata", "Hi I heard about Spark", "big data with spark", "spark big data"),

                DocumentRow("java", "I wish Java could use case classes", "love to code in Java", "java"),
                DocumentRow("machinelearning", "Logistic regression models are neat", "Logistic regression", "machine learning")
        ), DocumentRow::class.java).toDF("category", "content", "labels","title")


        //val ngrams = OwnNGram().setInputCol("content").setOutputCol("ngrams")

        //ngrams.transform(corpus).show(3,false)

        //corpus.select(functions.regexp_replace(functions.split(corpus.col("content")," ")," ","_")).show(3,false)
        //corpus.select(functions.array("content","title")).show(3,false)
        //corpus.select(corpus.col("*"), functions.split(functions.concat_ws(" ",corpus.col("content"),corpus.col("title"))," ").`as`("content")).show(3,false)

        val vtmDataPipeline = constructVTMPipeline(stopwords.value,2000)

        println(corpus.count())

        val vtmPipelineModel = vtmDataPipeline.fit(corpus)


        val cvModel = vtmPipelineModel.stages()[4] as CountVectorizerModel

        println("cv model vocabulary: " + cvModel.vocabulary().toList())
        val indexer = vtmPipelineModel.stages()[0] as StringIndexerModel
        if (deleteIfExists(LABELS)) {
            indexer.save(LABELS)
        }

        val parsedCorpus = vtmPipelineModel.transform(corpus).drop("content", "words", "filteredWords", "tfFeatures")

        val vtmTitlePipeline = constructTitleVtmDataPipeline(stopwords.value,500)

        val vtmTitlePipelineModel = vtmTitlePipeline.fit(parsedCorpus)

        val parsedCorpusTitle = vtmTitlePipelineModel.transform(parsedCorpus).drop("title_words", "filtered_titleWords", "tf_titleFeatures")

        parsedCorpusTitle.show(10, false)

        val vtmTagPipeline = constructTagVtmDataPipeline(200)

        val vtmTagPipelineModel = vtmTagPipeline.fit(parsedCorpusTitle)

        val fullParsedCorpus = vtmTagPipelineModel.transform(parsedCorpusTitle).drop("tag_words", "tag_ngrams", "tag_tfFeatures")

        val contentScaler = vtmPipelineModel.stages().last() as StandardScalerModel

        val titleNormalizer = vtmTitlePipelineModel.stages().last() as Normalizer

        val tagNormalizer = vtmTagPipelineModel.stages().last() as Normalizer

        //VectorAssembler().
        val assembler = VectorAssembler().setInputCols(arrayOf(contentScaler.outputCol, titleNormalizer.outputCol, tagNormalizer.outputCol))
                .setOutputCol(featureCol)

        assembler.transform(fullParsedCorpus).show(3, false)
    }

    fun runSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200")
                    .set("es.nodes.discovery", "true")
                    .set("es.nodes.wan.only", "false")

            val jsc = JavaSparkContext(sparkConf)
            datasetAssamblerTest(jsc)

        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")

    }

    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val datasetTest = DatasetTest()
            datasetTest.runSpark()
        }
    }

}
