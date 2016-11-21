package uy.com.collokia.runSparkOnEMR.jobs

import org.apache.spark.api.java.JavaSparkContext
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.getSparkContextOnEMR
import java.io.Serializable

class TestJob() : Serializable {
    companion object {

        fun testRun(jsc: JavaSparkContext){

            val list = listOf(1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9)

            val listRdd = jsc.parallelize(list)

            println(listRdd.count())

            val sum = listRdd.reduce { a, b ->  a+b }
            println(sum)
        }

        @JvmStatic fun main(args: Array<String>) {

            val jsc = getSparkContextOnEMR("Test")
            testRun(jsc)
            closeSpark(jsc)

        }
    }
}