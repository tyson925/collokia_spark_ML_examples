package uy.com.collokia.ml.util

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import scala.Tuple2


public fun getMulticlassMetrics(model: DecisionTreeModel, data: JavaRDD<LabeledPoint>): MulticlassMetrics {

    val predictionsAndLabels = data.map { instance ->

        Tuple2(model.predict(instance.features()) as Any, instance.label() as Any)
    }
    return MulticlassMetrics(predictionsAndLabels.rdd())
}

public fun getBinaryClassificationMetrics(model: DecisionTreeModel, data: JavaRDD<LabeledPoint>): BinaryClassificationMetrics {
    val predictionsAndLabels = data.map { instance ->

        Tuple2(model.predict(instance.features()) as Any, instance.label() as Any)
    }
    return BinaryClassificationMetrics(predictionsAndLabels.rdd(),100)
}

public fun printMulticlassMetrics(evaulation: MulticlassMetrics): String {
    val res = StringBuffer("F-measure:\t${evaulation.fMeasure()},\tprecision:${evaulation.precision()},\trecall:\t${evaulation.recall()}\n")
    res.append("W. F-measure:\t${evaulation.weightedFMeasure()},\tW. precision:${evaulation.weightedPrecision()},\tW. recall:\t${evaulation.weightedRecall()}\n")
    res.append("label stats:\n")
    println("labels:\t"+evaulation.labels().joinToString("\t"))
    res.append((0..evaulation.labels().size - 1).map({ category ->
        "label:\t${category}:\tF-measure:\t${evaulation.fMeasure(category.toDouble())},\tprecision:\t${evaulation.precision(category.toDouble())},\trecall:\t${evaulation.recall(category.toDouble())}"

    }).joinToString("\n")+"\n")
    println("confusionMatrix:\n")
    //res.append("${evaulation.confusionMatrix()}\n")
    res.append(printMatrix(evaulation.confusionMatrix(),evaulation.labels()))
    return res.toString()
}

private fun printMatrix(matrix : Matrix,labels : DoubleArray) : String{
    val res = StringBuffer()
    res.append("\t")
    labels.forEach { label ->
        res.append("${label},\t")
    }
    res.append("\n")
    (0..labels.size-1).forEach { i ->
        res.append("${labels[i]},\t")
        (0..matrix.numCols()-1).forEach { col ->
            res.append("${matrix.apply(i,col)},\t")
        }
        res.append("\n")
    }

    return res.toString()
}

public fun printBinaryClassificationMetrics(evaulation: BinaryClassificationMetrics) : String {
    val res = StringBuffer("areaUnderROC:\t${evaulation.areaUnderROC()}\n")
    //val roc = evaulation.roc().collect() as Tuple2<*,*>
    //res.append("${roc._1}\t${roc._2}\n")

    return res.toString()
}

public fun randomClassifier(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>) {
    val trainPriorProbabilities = classProbabilities(trainData)
    val cvPriorProbabilities = classProbabilities(cvData)
    val accuracy = trainPriorProbabilities.zip(cvPriorProbabilities).map { probapilities ->
        val (trainProb, cvProb) = probapilities
        trainProb * cvProb
    }.sum()
    println(accuracy)
}

public fun classProbabilities(data: JavaRDD<LabeledPoint>): DoubleArray {
    // Count (category,count) in data
    val countsByCategory = data.map({ instance -> instance.label() }).countByValue()
    // order counts by category and extract counts
    val counts = countsByCategory.toSortedMap().map { it -> it.value }

    return counts.map({ it -> it.toDouble() / counts.sum() }).toDoubleArray()
}

