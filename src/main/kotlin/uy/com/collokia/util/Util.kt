package uy.com.collokia.util

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SparkSession
import java.io.Serializable
import java.text.DecimalFormat

val REUTERS_DATA = "./data/reuters/json/reuters.json"
val VTM_PIPELINE = "./data/model/vtmPipeLine"

data class ReutersDocument(val title: String?, var body: String?, val date: String,
                           val topics: List<String>?, val places: List<String>?, val organisations: List<String>?, val id: Int) : Serializable

//required "var" according to `Encoders.bean`
data class DocumentRow(var category: String, var content: String, var title: String, var labels: String) : Serializable

data class ClassifierResults(val category: String, val decisionTree: Double, val randomForest: Double, val svm: Double,
                             val logReg: Double) : Serializable

val MAPPER = jacksonObjectMapper()

val featureCol = "normIdfFeatures"
val labelIndexCol = "categoryIndex"


val OVR_MODEL = "./data/model/ovrDectisonTree"
val LABELS = "./data/model/labelIndexer"

data class EvaluationMetrics(val category: String, val fMeasure: Double, val precision: Double, val recall: Double) : scala.Serializable {
    companion object {
        val formatter = DecimalFormat("#0.00")
    }

    override fun toString(): String {
        return "evaluation metrics for $category:\t" +
                "FMeasure:\t${formatter.format(fMeasure)}\t" +
                "Precision:\t${formatter.format(precision)}\t" +
                "Recall:${formatter.format(recall)}"
    }
}
