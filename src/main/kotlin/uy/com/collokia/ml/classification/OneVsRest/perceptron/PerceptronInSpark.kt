package uy.com.collokia.ml.classification.OneVsRest.perceptron

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.classification.OneVsRest.corpusFileName
import uy.com.collokia.ml.classification.OneVsRest.generateVtm
import uy.com.collokia.ml.classification.nlp.vtm.CONTENT_VTM_VOC_SIZE
import uy.com.collokia.ml.classification.nlp.vtm.TAG_VTM_VOC_SIZE
import uy.com.collokia.ml.classification.nlp.vtm.TITLE_VTM_VOC_SIZE
import uy.com.collokia.util.LABELS
import uy.com.collokia.util.featureCol
import uy.com.collokia.util.labelIndexCol
import uy.com.collokia.util.predictionCol
import java.io.File
import java.io.Serializable

class PerceptronInSpark() : Serializable {

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            val perceptron = PerceptronInSpark()
            perceptron.runOnSpark()
        }


    }


    fun evaluatePerceptron(dataset: Dataset<Row>): MulticlassMetrics {
        val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))
        val indexer = StringIndexerModel.load(LABELS)

        val cachedTrain = train.cache()
        val cachedTest = test.cache()

        train.show(3, false)
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
        val numberOfFeatures = CONTENT_VTM_VOC_SIZE + TITLE_VTM_VOC_SIZE + TAG_VTM_VOC_SIZE
        val layers = arrayOf(numberOfFeatures, numberOfFeatures, numberOfFeatures - 1000, 11).toIntArray()
// create the trainer and set its parameters
        val trainer = MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100)
                .setFeaturesCol(featureCol)
                .setLabelCol(labelIndexCol)
//                .setPredictionCol(predictionCol)

        val perceptronModel = trainer.fit(cachedTrain)

        val predictions = perceptronModel.transform(cachedTest)

        predictions.show(3)
        // evaluate the model
        val predictionsAndLabels = predictions.select(predictionCol, labelIndexCol).toJavaRDD().map({ row ->
            Tuple2(row.getDouble(0) as Any, row.getDouble(1) as Any)
        })

        val metrics = MulticlassMetrics(predictionsAndLabels.rdd())
        println(metrics)
        println("F-measure:\t${metrics.weightedFMeasure()}")
        return metrics
    }


    fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200")
                    .set("es.nodes.discovery", "true")
                    .set("es.nodes.wan.only", "false")

            val jsc = JavaSparkContext(sparkConf)
            val sparkSession = SparkSession.builder().master("local").appName("one vs rest example").orCreate

            val dataset = if (File(corpusFileName).exists()) {
                sparkSession.read().load(corpusFileName)
            } else {
                generateVtm(jsc, sparkSession)
            }

            evaluatePerceptron(dataset)

        }
        println("Execution time is ${time.second}")
    }

}
