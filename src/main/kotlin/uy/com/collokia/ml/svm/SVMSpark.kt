package uy.com.collokia.ml.svm

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.SVMWithSGD

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import scala.Tuple2
import uy.com.collokia.ml.util.*
import uy.com.collokia.scala.ClassTagger
import uy.com.collokia.util.component1
import uy.com.collokia.util.component2
import java.io.Serializable

public class SVMSpark() : Serializable {


    public fun evaulate10Fold(data : JavaRDD<LabeledPoint>) : Double{
        val tenFolds = MLUtils.kFold(data.rdd(),10,10, ClassTagger.scalaClassTag(LabeledPoint::class.java))

        val resultsInFmeasure = tenFolds.mapIndexed { i, fold ->
            val (trainData,testData) = fold
            println("number of fold:\t${i}")
            val Fmeasure = evaulateSVM(trainData.toJavaRDD(),testData.toJavaRDD(),2)
            Fmeasure
        }
        return resultsInFmeasure.average()
    }

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

        val testPrediction = predicateSVM(model, cvData)
        val FMeasure = evaulateAndPrintPrediction(numClasses,testPrediction)

/*// Clear the default threshold.
        model.clearThreshold()


// Compute raw scores on the test set.
        val scoreAndLabels = cvData.map { point ->
            val score = model.predict(point.features())
            Tuple2(score as Any, point.label() as Any)
        }

// Get evaluation metrics.
        val metrics = BinaryClassificationMetrics(scoreAndLabels.rdd())

        val auROC = metrics.areaUnderROC()
        println("areaUnderROC:\t${auROC}")*/
        return FMeasure
    }

}

fun main(args: Array<String>) {

}