package uy.com.collokia.ml.classification.weka

import uy.com.collokia.ml.util.loadArff
import weka.classifiers.evaluation.Evaluation
import weka.classifiers.evaluation.EvaluationUtils
import weka.classifiers.trees.J48
import weka.core.Instances
import java.util.*


public class WekaClassification() {

    public fun evaulateTenFold(data: Instances) {

        val classifier = J48()

        //val model = classifier.buildClassifier(data)

        data.setClassIndex(0)
        val evaulation = Evaluation(data)
        evaulation.crossValidateModel(classifier, data, 10, Random(1))
        println(classifier)
        println(evaulation.toSummaryString())
        println(evaulation.toMatrixString())
        println(evaulation.toClassDetailsString())
    }


}

fun main(args: Array<String>) {

    val data = loadArff("./data/reuters/arff/acq.arff")
    val weka = WekaClassification()
    weka.evaulateTenFold(data)
}