package uy.com.collokia.runSparkOnEMR.jobs

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import uy.com.collokia.common.utils.ES_HOST_NAME
import java.io.Serializable

public class TestJob() : Serializable {
    companion object {

        public fun testRun(jsc: JavaSparkContext){

            val list = listOf(1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9)

            val listRdd = jsc.parallelize(list)

            println(listRdd.count())

            val sum = listRdd.reduce { a, b ->  a+b }
            println(sum)
        }

        @JvmStatic fun main(args: Array<String>) {
            val sparkConf = SparkConf().setAppName("test")
                    .set("es.nodes", "${ES_HOST_NAME}").set("es.nodes.discovery", "false")
                    .set("num-executors", "3")
                    .set("executor-cores", "4")
                    .set("executor-memory", "14G")


            val jsc = JavaSparkContext(sparkConf)

            testRun(jsc)
            jsc.clearJobGroup()
            jsc.close()
            jsc.stop()
            println("exit spark job conf")

        }
    }
}

