package uy.com.collokia.coreNlp.stanford


import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.SparkSession
import scala.Serializable
import scala.Tuple2
import uy.com.collokia.stanford.coreNLP.CoreNLP
import java.util.*

public class TestData(val id: Int, val text: String) : Serializable


@Suppress("UNCHECKED_CAST")
public class CoreNlpTest() {

    public fun testCoreNlp(jsc: JavaSparkContext, sparkSession: SparkSession) {

        val test = LinkedList<Tuple2<Int, String>>()
        test.add(Tuple2(1, "<xml>Stanford University is located in California. It is a great university.</xml>"))
        test.add(Tuple2(2, "<xml>University of Szeged is located in Hungary. It is a great university.</xml>"))
        test.add(Tuple2(3, "<xml>Collokia is located in Uruguay.</xml>"))
        test.add(Tuple2(4, "<xml>Collokia is located in Uruguay.</xml>"))
        test.add(Tuple2(5, "<xml>Collokia is located in Uruguay.</xml>"))
        test.add(Tuple2(6, "<xml>University of Szeged is located in Hungary. It is a great university.</xml>"))
        test.add(Tuple2(7, "<xml>University of Szeged is located in Hungary. It is a great university.</xml>"))
        test.add(Tuple2(8, "<xml>Stanford University is located in California. It is a great university.</xml>"))
        test.add(Tuple2(9, "<xml>Stanford University is located in California. It is a great university.</xml>"))
        test.add(Tuple2(10, "<xml>Collokia is located in Uruguay.</xml>"))

        val testRdd = jsc.parallelizePairs(test).map { item ->
            TestData(item._1, item._2)
        }

        val input = sparkSession.createDataFrame(testRdd, TestData::class.java).toDF("id", "text");

        //println(input.collect())
        val coreNLP = CoreNLP(sparkSession, "pos, lemma, parse, ner").setInputCol("text")

        val parsed = coreNLP.transform(input)


        /*val ner = parsed.select("lemma").javaRDD().map { row ->
            val nerInDoc = row.get(0) as ArrayBuffer<ArrayBuffer<String>>


            for (nerInSentence in nerInDoc) {
                for (nerInToken in nerInSentence) {
                    print(nerInToken + " ")
                }
                println()
            }
            nerInDoc
        }
        ner.collect()*/
        parsed?.show(false)
        //val first = parsed.first().getAs[Row]("parsed")
    }

}

fun main(args: Array<String>) {
    val sparkConf = SparkConf().setAppName("classificationTest").setMaster("local[2]")

    val jsc = JavaSparkContext(sparkConf)
    val sparkSession = SparkSession.builder().master("local").appName("prediction").getOrCreate()

    /*val test = LinkedList<Tuple2<Int,String>>()
    test.add(Tuple2(1, "<xml>Stanford University is located in California. It is a great university.</xml>"))
    test.add(Tuple2(2, "<xml>University of Szeged is located in Hungary. It is a great university.</xml>"))
    test.add(Tuple2(3, "<xml>Collokia is located in Uruguay.</xml>"))

    val testRdd = ctx.parallelizePairs(test).map{ item ->
        TestData(item._1,item._2)
    }

    testRdd.collect()*/

    val test = CoreNlpTest()
    test.testCoreNlp(jsc, sparkSession)


    jsc.close()
    jsc.stop()

}






