package uy.com.collokia.ml.svm

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.SVMWithSGD

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import scala.Tuple2
import java.io.Serializable

public class SVMSpark() : Serializable {

    public fun simpleSVM(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int) {
// Run training algorithm to build the model
        val numIterations = 300
        //val model = SVMWithSGD.train(trainData.rdd(), numIterations)
val model = LogisticRegressionWithLBFGS().setNumClasses(2).run(trainData.rdd())

        val predictionsAndLabels = cvData.map({ example ->
            Tuple2(model.predict(example.features()) as Any, example.label() as Any)
        })

        println(predictionsAndLabels.collect().joinToString("\n"))
        val multimetrics = MulticlassMetrics(predictionsAndLabels.rdd())
        println("F-measure:\t${multimetrics.weightedFMeasure()}, precision:\t${multimetrics.weightedPrecision()}, recall:\t${multimetrics.weightedRecall()}")
// Clear the default threshold.
        model.clearThreshold()

// Compute raw scores on the test set.
        val scoreAndLabels = cvData.map { point ->
            val score = model.predict(point.features())
            Tuple2(score as Any, point.label() as Any)
        }

// Get evaluation metrics.
        val metrics = BinaryClassificationMetrics(scoreAndLabels.rdd())

        val auROC = metrics.areaUnderROC()
        println("areaUnderROC:\t${auROC}")

    }

}

fun main(args: Array<String>) {

}