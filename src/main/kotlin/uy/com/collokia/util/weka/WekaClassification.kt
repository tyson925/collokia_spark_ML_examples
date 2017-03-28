package uy.com.collokia.util.weka

import uy.com.collokia.common.utils.machineLearning.loadArff
import weka.classifiers.evaluation.Evaluation
import weka.classifiers.trees.J48
import weka.core.Instances
import java.io.File
import java.util.*


class WekaClassification {

    fun evaluateTenFold(data: Instances): Double {

        val classifier = J48()

        //val model = classifier.buildClassifier(data)

        data.setClassIndex(0)
        val evaluation = Evaluation(data)
        evaluation.crossValidateModel(classifier, data, 10, Random(1))
        println(classifier)
        println(evaluation.toSummaryString())
        println(evaluation.toMatrixString())
        println(evaluation.toClassDetailsString())

        return evaluation.fMeasure(1)
    }

    fun evaluateReuters() {
        val results = File("./testData/reuters/arff/").listFiles().filter { file -> file.name.endsWith(".arff") }.map { file ->
            val category = file.name.substringBefore(".arff")
            val data = loadArff(file.canonicalPath)
            Pair(category, evaluateTenFold(data))
        }

        println(results.sortedByDescending { value -> value.second }.joinToString("\n"))
    }

    companion object{
        @JvmStatic fun main(args: Array<String>) {

            //val testData = loadArff("./testData/reuters/arff/acq.arff")
            val weka = WekaClassification()
            //weka.evaluateTenFold(testData)
            weka.evaluateReuters()
        }
    }

}

