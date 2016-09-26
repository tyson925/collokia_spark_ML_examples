@file:Suppress("UNUSED_VARIABLE")

package uy.com.collokia.util.readData

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.util.DocumentRow
import uy.com.collokia.util.MAPPER
import uy.com.collokia.util.ReutersDocument
import java.io.File
import uy.com.collokia.ml.classification.ReutersDocumentClassification

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

fun readReutersJson() {
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
                    if (reutersDoc.topics != null && reutersDoc.topics.intersect(ReutersDocumentClassification.topCategories).isNotEmpty()) {
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

fun parseCorpus(sparkSession: SparkSession, corpusInRaw: JavaRDD<String>, subTopic: String?): Dataset<DocumentRow> {

    val corpusRow = subTopic?.let {
        filterToOneCategory(corpusInRaw, subTopic)
    } ?: filterToTopCategories(corpusInRaw)

    corpusInRaw.unpersist()

    return documentRddToDF(sparkSession, corpusRow)
}

fun documentRddToDF(sparkSession: SparkSession, corpusRow: JavaRDD<DocumentRow>): Dataset<DocumentRow> {
    println("corpus size: " + corpusRow.count())

    val reutersEncoder = Encoders.bean(DocumentRow::class.java)

    val textDataFrame = sparkSession.createDataset(corpusRow.rdd(), reutersEncoder)

    corpusRow.unpersist()

    return textDataFrame
}

private fun filterToOneCategory(corpusInRaw: JavaRDD<String>, category: String): JavaRDD<DocumentRow> {
    val corpusRow = corpusInRaw.map { line ->
        val doc = ReutersDocumentClassification.MAPPER.readValue(line, ReutersDocument::class.java)
        val topics = doc.topics?.intersect(ReutersDocumentClassification.topCategories) ?: listOf<String>()
        val content = doc.body + (doc.title ?: "")

        val row = if (topics.contains(category)) {
            DocumentRow(category, content, "", "")
        } else {
            DocumentRow("other", content, "", "")
        }
        row
    }
    return corpusRow
}

private fun filterToTopCategories(corpusInRaw: JavaRDD<String>): JavaRDD<DocumentRow> {
    val corpusRow = corpusInRaw.map { line ->
        val doc = ReutersDocumentClassification.MAPPER.readValue(line, ReutersDocument::class.java)
        val topics = doc.topics?.intersect(ReutersDocumentClassification.topCategories) ?: listOf<String>()
        val content = doc.body + (doc.title ?: "")

        val intersectCategory = topics.intersect(ReutersDocumentClassification.topCategories)
        intersectCategory.first()
        val rows = intersectCategory.map { category ->
            DocumentRow(category, content, "", "")
        }.iterator()
        DocumentRow(intersectCategory.first(), content, "", "")
        //rows
    }
    return corpusRow
}

