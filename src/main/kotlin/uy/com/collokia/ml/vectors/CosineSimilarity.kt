package uy.com.collokia.ml.vectors

import org.apache.spark.api.java.function.DoubleFunction
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scala.Tuple2
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession


class CosineSimilarity() {
    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val testCosineSimilarity = CosineSimilarity()
            testCosineSimilarity.run()
        }
    }

    fun run() {
        val jsc = getLocalSparkContext("valami")
        val sparkSession = getLocalSparkSession("valami")


        // Load and parse the data file.
        val rows = jsc.textFile("./data/vectors/vectors.txt").map { line ->
            val values = line.split(' ').map({ it -> it.toDouble() }).toDoubleArray()
            Vectors.dense(values)
        }.cache()
        val mat = RowMatrix(rows.rdd())

        // Compute similar columns perfectly, with brute force.
        val exact = mat.columnSimilarities()

        // Compute similar columns with estimation using DIMSUM
        val approx = mat.columnSimilarities(0.1)
        val exactEntries = exact.entries().toJavaRDD().mapToPair { matrixEntry -> Tuple2(Tuple2(matrixEntry.i(),matrixEntry.j()),matrixEntry.value()) }
        val approxEntries = approx.entries().toJavaRDD().mapToPair { matrixEntry -> Tuple2(Tuple2(matrixEntry.i(),matrixEntry.j()),matrixEntry.value()) }

        val MAE = exactEntries.leftOuterJoin(approxEntries).values().mapToDouble<Double> (DoubleFunction{ entry ->
            val (key, value) = entry
            if (value.isPresent){
                Math.abs(key - value.get())
            } else {
                Math.abs(key)
            }
        }).mean()


        println(exactEntries.collect().joinToString("\n"))
        println("---------------------------------------")
        println(approxEntries.collect().joinToString("\n"))

        println("Average absolute error in estimate is: $MAE")

    }

}

