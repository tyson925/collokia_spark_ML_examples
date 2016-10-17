@file:Suppress("UNUSED_VARIABLE")

package uy.com.collokia.ml.rdf

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Serializable
import uy.com.collokia.common.utils.deleteIfExists
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.ml.classification.OneVsRest.corpusFileName
import uy.com.collokia.ml.classification.OneVsRest.evaluateModelConfusionMTX
import uy.com.collokia.ml.classification.OneVsRest.generateVtm
import uy.com.collokia.util.OVR_MODEL
import uy.com.collokia.util.featureCol
import uy.com.collokia.util.labelIndexCol
import java.io.File


class OneVsRestInSpark() : Serializable {

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            val ovr = OneVsRestInSpark()
            ovr.runOnSpark()
        }
    }

    fun runOnSpark() {
        val time = measureTimeInMillis {

            val jsc = getLocalSparkContext("one vs rest example")
            val sparkSession = SparkSession.builder().master("local").appName("one vs rest example").orCreate

            val dataset = if (File(corpusFileName).exists()) {
                sparkSession.read().load(corpusFileName)
            } else {
                generateVtm(jsc, sparkSession)
            }
            //evaluateOneVsRestDecisionTrees(dataset)
            evaluateOneVsRest(dataset)
            //evaluateOneVsRestNaiveBayes(dataset)
            closeSpark(jsc)

        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }


    fun evaluateOneVsRest(dataset: Dataset<Row>) {

        val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))

        val impurity = "gini"
        val depth = 10
        val bins = 300
        val dt = DecisionTreeClassifier().setImpurity(impurity).setMaxDepth(depth).setMaxBins(bins)
        //dt.maxMemoryInMB = 512

        val lr = LogisticRegression().setMaxIter(300).setTol(1E-6).setFitIntercept(true).setStandardization(true)

        //val nb = NaiveBayes()

        val oneVsRest = OneVsRest().setClassifier(dt)
                .setFeaturesCol(featureCol)
                .setLabelCol(labelIndexCol)

        train.show(3)

        val ovrModel = oneVsRest.fit(train)
        //val ovrModel = perceptron.fit(train)

        if (deleteIfExists(OVR_MODEL)) {
            ovrModel.save(OVR_MODEL)
        }
        evaluateModelConfusionMTX(ovrModel, test)
    }




}


