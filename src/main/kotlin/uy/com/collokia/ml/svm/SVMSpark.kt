package uy.com.collokia.ml.svm

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.SVMWithSGD

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import scala.Tuple2
import uy.com.collokia.ml.util.predicateRandomForest
import uy.com.collokia.ml.util.predicateSVM
import uy.com.collokia.ml.util.printBinaryClassificationMetrics
import uy.com.collokia.ml.util.printMulticlassMetrics
import java.io.Serializable

public class SVMSpark() : Serializable {

    public fun buildSimpleSVM(trainData: JavaRDD<LabeledPoint>, numClasses: Int): SVMModel {
// Run training algorithm to build the model
        val numIterations = 300
        println("Build SVM with ${numClasses} classes...")
        val model = SVMWithSGD.train(trainData.rdd(), numIterations)
        return model
    }

    public fun evaulateSVM(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): Double {

        val model = buildSimpleSVM(trainData, numClasses)


        trainData.unpersist()

        println("evaulate decision tree model...")

        val evaulateTest = predicateSVM(model, cvData)
        val FMeasure = if (numClasses == 2) {
            val evaulationBin = BinaryClassificationMetrics(evaulateTest, 100)
            val evaulation = MulticlassMetrics(evaulateTest)
            println(printMulticlassMetrics(evaulation))
            println(printBinaryClassificationMetrics(evaulationBin))
            evaulation.fMeasure(1.0)
        } else {
            val evaulation = MulticlassMetrics(evaulateTest)
            println(printMulticlassMetrics(evaulation))
            evaulation.fMeasure(1.0)
        }

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
        return FMeasure
    }

}

fun main(args: Array<String>) {

}