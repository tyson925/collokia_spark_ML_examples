package uy.com.collokia.ml.vectors

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scala.Tuple2
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.common.utils.rdd.sortByValue
import java.io.Serializable


class CosineSimilarity : Serializable{
    companion object {
        @JvmStatic fun main(args: Array<String>) {

            val time = measureTimeInMillis{
                val testCosineSimilarity = CosineSimilarity()
                testCosineSimilarity.run()
            }
            println("execution time was ${time.second}")

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

        val matT = transposeRowMatrix(mat)

        println(matT)

        // Compute similar columns perfectly, with brute force.
        val exact = matT.columnSimilarities()

        // Compute similar columns with estimation using DIMSUM
        val approx = matT.columnSimilarities(0.1)
        val exactEntries = exact.entries().toJavaRDD().mapToPair { matrixEntry -> Tuple2(Tuple2(matrixEntry.i(),matrixEntry.j()),matrixEntry.value()) }
        /*val approxEntries = approx.entries().toJavaRDD().mapToPair { matrixEntry -> Tuple2(Tuple2(matrixEntry.i(),matrixEntry.j()),matrixEntry.value()) }

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

        println("Average absolute error in estimate is: $MAE")*/
        println(exactEntries.sortByValue(false).collect().joinToString("\n"))
    }

    fun transposeRowMatrix(m: RowMatrix): RowMatrix {

        val valami = m.rows().zipWithIndex()

        val transposedRowsRDD = m.rows().zipWithIndex().toJavaRDD().map{ indexedRow ->
            val (row, rowIndex)  = indexedRow
            rowToTransposedTriplet(row, rowIndex as Long)
        }.flatMapToPair{x -> x.iterator()} // now we have triplets (newRowIndex, (newColIndex, value))
        .groupByKey()
        .sortByKey().map({it -> it._2}) // sort rows and remove row indexes
                .map({it -> buildRow(it)}) // restore order of elements in each row and remove column indexes
        return RowMatrix(transposedRowsRDD.rdd())
    }


    fun rowToTransposedTriplet(row: Vector, rowIndex: Long) : List<Tuple2<Long,Tuple2<Long,Double>>> {
        val indexedRow = row.toArray()
       return indexedRow.mapIndexed { colIndex,value ->  Tuple2(colIndex.toLong(), Tuple2(rowIndex, value))}
    }


    fun buildRow(rowWithIndexes: Iterable<Tuple2<Long, Double>>): Vector {
        //TODO
        val resArr = DoubleArray(rowWithIndexes.toList().size)
        rowWithIndexes.forEach { indexedValue ->
            val (index, value) = indexedValue
            resArr[index.toInt()] = value
        }
        return Vectors.dense(resArr)
    }



}

