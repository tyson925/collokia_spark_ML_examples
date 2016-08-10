package uy.com.collokia.ml.rdf

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.ml.classification.DocumentClassification
import uy.com.collokia.ml.classification.VTM_PIPELINE
import uy.com.collokia.ml.util.REUTERS_DATA
import uy.com.collokia.ml.util.deleteIfExists
import uy.com.collokia.ml.util.printMatrix
import uy.com.collokia.util.formatterToTimePrint
import uy.com.collokia.util.measureTimeInMillis
import java.io.File

public val OVR_MODEL = "./data/model/ovrDectisonTree"
public val LABELS = "./data/model/labelIndexer"

public class OneVsRestInSpark() {


    public fun evaulateOneVsRest(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile(REUTERS_DATA).cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val documentClassification = DocumentClassification()
        //val parsedCorpus = documentClassification.parseCorpus(sparkSession, corpusInRaw, null)
        val parsedCorpus = documentClassification.readDzoneFromEs(sparkSession, jsc)

        val vtmDataPipeline = documentClassification.constructVTMPipeline()

        val vtmPiplineModel = vtmDataPipeline.fit(parsedCorpus)
        val indexer = vtmPiplineModel.stages()[0] as StringIndexerModel
        if (deleteIfExists(LABELS)){
            indexer.save(LABELS)
        }

        val (train, test) = vtmPiplineModel.transform(parsedCorpus).randomSplit(doubleArrayOf(0.9, 0.1))
        if (deleteIfExists(VTM_PIPELINE)){
            vtmPiplineModel.save(VTM_PIPELINE)
        }

        //val (train, test) = documentClassification.constructVTMData(sparkSession, corpusInRaw, null).randomSplit(doubleArrayOf(0.9, 0.1))
        //val labels = train.select("category").toJavaRDD().map { it-> it.getString(0) }.groupBy({ it -> it }).keys().collect()

        val impurity = "gini"
        val depth = 10
        val bins = 300
        val dt = DecisionTreeClassifier().setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)

        val lr = LogisticRegression()
                .setMaxIter(10)
                .setTol(1E-6)
                .setFitIntercept(true)

        val oneVsRest = OneVsRest().setClassifier(dt).setFeaturesCol(DocumentClassification.featureCol).setLabelCol(DocumentClassification.labelIndexCol)

        val ovrModel = oneVsRest.fit(train)

        if (deleteIfExists(OVR_MODEL)){
            ovrModel.save(OVR_MODEL)
        }

        // Convert indexed labels back to original labels.
        val labelConverter = IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(indexer.labels())


        val predicatePipeline = Pipeline().setStages(arrayOf(ovrModel, labelConverter))

        //val predictions = ovrModel.transform(test)

        val predictions = predicatePipeline.fit(test).transform(test)


        predictions.show(3)
        // evaluate the model
        val predictionsAndLabels = predictions.select("prediction", DocumentClassification.labelIndexCol).toJavaRDD().map({ row -> Tuple2(row.getDouble(0) as Any, row.getDouble(1) as Any) })


        val metrics = MulticlassMetrics(predictionsAndLabels.rdd())
        val confusionMatrix = metrics.confusionMatrix()

// compute the false positive rate per label
//        val predictionColSchema = predictions.schema().fields()[0]
//        val numClasses = MetadataUtils.getNumClasses(predictionColSchema).get()

        val fprs = (0..indexer.labels().size - 1).map({ p -> Tuple2(indexer.labels()[p], metrics.fMeasure(p.toDouble())) })

        println(printMatrix(confusionMatrix))

        println(fprs.joinToString("\n"))
    }

    public fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200").set("es.nodes.discovery", "false")

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

