package uy.com.collokia.util

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.api.java.JavaSparkContext
import scala.Tuple2
import uy.com.collokia.common.utils.joinToString
import uy.com.collokia.common.utils.measureTimeInMillis
import java.io.Serializable


class RDDUtil : Serializable {


    //test how can I "update" an RDD
    fun upDateRDD(dataRDD: JavaPairRDD<Int, String>, updateData: JavaPairRDD<Int, String>) {

        //filter
        //val oldData = dataRDD.subtract(updateData)
        //union

        //println(oldData.collectAsMap().joinToString("\t"))
        //println()

        val result = dataRDD.union(updateData)



        println(result.collectAsMap().joinToString("\t"))

    }


    fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200")
                    .set("es.nodes.discovery", "true")
                    .set("es.nodes.wan.only", "false")

            val jsc = JavaSparkContext(sparkConf)

            val testRDD = jsc.parallelizePairs(listOf(Tuple2(1, "alma"), Tuple2(1, "alma"), Tuple2(2, "alma"),
                    Tuple2(3, "alma"), Tuple2(4, "alma"), Tuple2(5, "alma")))

            val updateRDD = jsc.parallelizePairs(listOf(Tuple2(2, "korte"), Tuple2(3, "eper"), Tuple2(4, "burgonya")))

            upDateRDD(testRDD, updateRDD)
        }
    }

    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val rddUtil = RDDUtil()
            rddUtil.runOnSpark()
        }
    }

}