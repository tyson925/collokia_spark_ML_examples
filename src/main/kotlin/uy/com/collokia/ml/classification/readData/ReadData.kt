package uy.com.collokia.ml.classification.readData

import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.ml.classification.DocumentRow

public fun readDzoneFromEs(sparkSession: SparkSession, jsc: JavaSparkContext) : Dataset<DocumentRow> {

    val corpusRow = JavaEsSpark.esRDD(jsc, "dzone_data/SOThreadExtractValues").map { line ->
        val (id, map) = line
        val category = map.getOrElse("category") { "other" } as String
        val content = map.getOrElse("parsedContent") { "other" } as String
        val title = map.getOrElse("title") { "other" } as String
        //val taggedTitle = title.split(Regex("W")).map { titleToken -> "title:${titleToken}"}.joinToString(" ")
        val labels = map.getOrElse("tags") { listOf<String>() } as List<String>
        val taggedLabels = labels.map { label -> "label:${label}" }.joinToString(" ")
        DocumentRow(category, content, title,labels.joinToString(" "))
    }
    return corpusRow.convertRDDToDF(sparkSession)
}

