package uy.com.collokia.ml.svm

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.machineLearning.evaluateAndPrintPrediction
import uy.com.collokia.common.utils.machineLearning.predicateMLModel
import uy.com.collokia.scala.ClassTagger
import java.io.Serializable

class SVMSpark() : Serializable {


    fun evaluate10Fold(data : JavaRDD<LabeledPoint>) : Double{
        val tenFolds = MLUtils.kFold(data.rdd(),10,10, ClassTagger.scalaClassTag(LabeledPoint::class.java))

        val resultsInFmeasure = tenFolds.mapIndexed { i, fold ->
            val (trainData,testData) = fold
            println("number of fold:\t$i")
            val Fmeasure = evaluateSVM(trainData.toJavaRDD(),testData.toJavaRDD(),2)
            Fmeasure
        }
        return resultsInFmeasure.average()
    }

    fun buildSimpleSVM(trainData: JavaRDD<LabeledPoint>, numClasses: Int): SVMModel {
// Run training algorithm to build the model
        val numIterations = 300
        println("Build SVM with $numClasses classes...")

        val model = SVMWithSGD.train(trainData.rdd(), numIterations)
        return model
    }

    fun evaluateSVM(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int): Double {

        val model = buildSimpleSVM(trainData, numClasses)

        trainData.unpersist()

        println("evaulate decision tree model...")

        val testPrediction = predicateMLModel(model, cvData)
        val FMeasure = evaluateAndPrintPrediction(numClasses,testPrediction)

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
