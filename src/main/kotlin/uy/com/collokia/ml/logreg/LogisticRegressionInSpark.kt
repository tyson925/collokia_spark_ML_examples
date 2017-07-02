package uy.com.collokia.ml.logreg

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions
import scala.Tuple2
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.machineLearning.evaluateAndPrintPrediction
import uy.com.collokia.common.utils.machineLearning.predicateMLModel
import uy.com.collokia.scala.ClassTagger

class LogisticRegressionInSpark {

    fun evaluate10Fold(data : JavaRDD<LabeledPoint>) : Double{
        val tenFolds = MLUtils.kFold(data.rdd(),10,10, ClassTagger.scalaClassTag(LabeledPoint::class.java))

        val resultsInFmeasure = tenFolds.mapIndexed { i, fold ->
            val (trainData,testData) = fold
            println("number of fold:\t$i")
            val Fmeasure = evaluateSimpleLogReg(trainData.toJavaRDD(),testData.toJavaRDD(),2)
            Fmeasure
        }
        return resultsInFmeasure.average()
    }

    fun evaluateSimpleLogReg(trainData: JavaRDD<LabeledPoint>, testData: JavaRDD<LabeledPoint>, numClasses: Int): Double {

        // Building the model
        val numIterations = 100
        val stepSize = 0.00000001

        val model = buildLogReg(trainData,numIterations,stepSize,numClasses)

        trainData.unpersist()

        println("evaluate logistic regression model...")
        val testPrediction = predicateMLModel(model, testData)

        return evaluateAndPrintPrediction(numClasses,testPrediction)

    }

    fun evaluateLogRegModel(lrModel: org.apache.spark.ml.classification.LogisticRegressionModel): Tuple2<Double, Double> {
        // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
        val trainingSummary = lrModel.summary()

        // Obtain the objective per iteration.
        val objectiveHistory = trainingSummary.objectiveHistory()
        objectiveHistory.forEach(::println)

        val binarySummary = trainingSummary as BinaryLogisticRegressionSummary

        // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
        val roc = binarySummary.roc()
        roc.show()
        println("areaUnderROC:\t${binarySummary.areaUnderROC()}")

        // Set the model threshold to maximize F-Measure
        val fMeasure = binarySummary.fMeasureByThreshold()

        //fMeasure.show(100, false)
        val maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0)
        val bestThreshold = fMeasure.where(fMeasure.col("F-Measure").`$eq$eq$eq`(maxFMeasure)).select("threshold").head().getDouble(0)

        lrModel.threshold = bestThreshold

        println("Coefficients: ${lrModel.coefficients()} Intercept: ${lrModel.intercept()}")
        println("maxFMeasure: $fMeasure\tthreshold: $bestThreshold")
        return Tuple2(maxFMeasure, bestThreshold)
    }


    fun buildLogReg(trainData: JavaRDD<LabeledPoint>, numIterations : Int,stepSize : Double, numClasses: Int) : LogisticRegressionModel {
        println("Build logReg model with $numClasses with parameters numIterations=$numIterations, stepSize=$stepSize")
        return LogisticRegressionWithLBFGS().setNumClasses(2).run(trainData.rdd())
    }

}

