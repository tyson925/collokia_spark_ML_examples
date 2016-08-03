package uy.com.collokia.ml.classification.weka

import uy.com.collokia.ml.util.loadArff
import weka.classifiers.evaluation.Evaluation
import weka.classifiers.trees.J48
import weka.core.Instances
import java.util.*
import java.io.File


public class WekaClassification() {

    public fun evaulateTenFold(data: Instances): Double {

        val classifier = J48()

        val model = classifier.buildClassifier(data)

        model

        data.setClassIndex(0)
        val evaulation = Evaluation(data)
        evaulation.crossValidateModel(classifier, data, 10, Random(1))
        println(classifier)
        println(evaulation.toSummaryString())
        println(evaulation.toMatrixString())
        println(evaulation.toClassDetailsString())

        return evaulation.fMeasure(1)
    }

    public fun evaulateReuters() {
        val results = File("./data/reuters/arff/").listFiles().filter { file -> file.name.endsWith(".arff") }.map { file ->
            val category = file.name.substringBefore(".arff")
            val data = loadArff(file.canonicalPath)
            Pair(category,evaulateTenFold(data))
        }

        println(results.sortedByDescending { value -> value.second }.joinToString("\n"))
    }

}

fun main(args: Array<String>) {

    //val data = loadArff("./data/reuters/arff/acq.arff")
    val weka = WekaClassification()
    //weka.evaulateTenFold(data)
    weka.evaulateReuters()
}