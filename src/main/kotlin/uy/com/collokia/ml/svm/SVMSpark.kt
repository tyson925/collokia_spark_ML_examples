package uy.com.collokia.ml.svm

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.LabeledPoint
import java.io.Serializable

public class SVMSpark() : Serializable {

    public fun simpleSVM(trainData: JavaRDD<LabeledPoint>, cvData: JavaRDD<LabeledPoint>, numClasses: Int){

    }

}

fun main(args: Array<String>) {

}