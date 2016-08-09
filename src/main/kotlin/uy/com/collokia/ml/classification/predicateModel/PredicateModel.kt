package uy.com.collokia.ml.classification.predicateModel

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.classification.OneVsRestModel
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import uy.com.collokia.ml.classification.DocumentClassification
import uy.com.collokia.ml.classification.DocumentRow
import uy.com.collokia.ml.classification.VTM_PIPELINE
import uy.com.collokia.ml.rdf.LABELS
import uy.com.collokia.ml.rdf.OVR_MODEL
import uy.com.collokia.ml.util.extractContentBoiler
import uy.com.collokia.util.formatterToTimePrint
import uy.com.collokia.util.measureTimeInMillis
import java.io.Serializable
import java.io.File
import java.net.URL
import java.util.*

public class PredicateModel() : Serializable {

    companion object {
        val LOG = LoggerFactory.getLogger(PredicateModel::class.java)
        val MAPPER = jacksonObjectMapper()

    }

    public fun predicateModel(jsc : JavaSparkContext) {
        val vtmPipeline = PipelineModel.load(VTM_PIPELINE)
        val ovrModel = OneVsRestModel.load(OVR_MODEL)

        val sparkSession = SparkSession.builder().master("local").appName("prediction").getOrCreate()

        val urls = loadUrls()

        val urlContents = urls.map { url ->
            val content = extractContentBoiler(URL(url),LOG)
            println("content: ${content}")
            DocumentRow("bigData",content)
        }
        val urlContentsRDD = jsc.parallelize(urlContents)

        val documentClassification = DocumentClassification()

        val test = vtmPipeline.transform(documentClassification.documentRddToDF(sparkSession,urlContentsRDD))

        val indexer = StringIndexerModel.load(LABELS)

        val labelConverter = IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(indexer.labels())


        val predicatePipeline = Pipeline().setStages(arrayOf(ovrModel,labelConverter))



        val predictions = predicatePipeline.fit(test).transform(test)
        //ovrModel.transform(test).show()
        //val predictions = labelConverter.transform()
        predictions.show()
    }


    private fun loadUrls(): List<String> {
        val res = LinkedList<String>()
        File("./data/urls/urls").forEachLine { line ->
            res.add(line)
        }
        return res
    }

    public fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200").set("es.nodes.discovery", "false")

            val jsc = JavaSparkContext(sparkConf)
predicateModel(jsc)

        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }

}


fun main(args: Array<String>) {
val predicateModel = PredicateModel()
    predicateModel.runOnSpark()
    //extractContentBoiler(URL("https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/spark-mllib-models.html"), PredicateModel.LOG)
}
