package uy.com.collokia.ml.rdf

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.ml.classification.DocumentClassification
import uy.com.collokia.ml.util.REUTERS_DATA
import uy.com.collokia.ml.util.printMatrix
import uy.com.collokia.util.formatterToTimePrint
import uy.com.collokia.util.measureTimeInMillis

public class OneVsRestInSpark() {


    public fun evaulateOneVsRest(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile(REUTERS_DATA).cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val documentClassification = DocumentClassification()
        val (train, test) = documentClassification.constructVTMData(sparkSession, corpusInRaw, null).randomSplit(doubleArrayOf(0.9, 0.1))


        val impurity = "gini"
        val depth = 10
        val bins = 300
        val dt = DecisionTreeClassifier().setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)

        val oneVsRest = OneVsRest().setClassifier(dt).setFeaturesCol(DocumentClassification.featureCol).setLabelCol(DocumentClassification.labelIndexCol)

        val ovrModel = oneVsRest.fit(train)

        val predictions = ovrModel.transform(test)

        // evaluate the model
        val predictionsAndLabels = predictions.select("prediction", DocumentClassification.labelIndexCol).toJavaRDD().map({ row -> Tuple2(row.getDouble(0) as Any, row.getDouble(1) as Any) })

        val metrics = MulticlassMetrics(predictionsAndLabels.rdd())
        val confusionMatrix = metrics.confusionMatrix()

// compute the false positive rate per label
//        val predictionColSchema = predictions.schema().fields()[0]
//        val numClasses = MetadataUtils.getNumClasses(predictionColSchema).get()

        val fprs = (0..9).map({ p -> Tuple2(p, metrics.fMeasure(p.toDouble())) })

        println(printMatrix(confusionMatrix))

        println(fprs.joinToString("\n"))
    }

    public fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")

            val jsc = JavaSparkContext(sparkConf)
            evaulateOneVsRest(jsc)

        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }
}

fun main(args: Array<String>) {
    val ovr = OneVsRestInSpark()
    ovr.runOnSpark()
}

