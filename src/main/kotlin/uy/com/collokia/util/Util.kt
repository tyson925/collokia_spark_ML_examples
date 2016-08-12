package uy.com.collokia.util

import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.api.java.function.Function
import org.apache.spark.api.java.function.Function2
import scala.Tuple2
import java.text.DecimalFormat
import java.text.Normalizer

public operator fun <T1, T2> Tuple2<T1, T2>.component1(): T1 = this._1
public operator fun <T1, T2> Tuple2<T1, T2>.component2(): T2 = this._2

public inline fun <reified T : Any> measureTimeInMillis(functionToTime: () -> T): Pair<T, Long> {
    val begin = System.currentTimeMillis()
    val ret = functionToTime()
    val end = System.currentTimeMillis()
    return Pair(ret, end - begin)
}

val formatterToTimePrint = DecimalFormat("#0.00000")

val normalizerRegex = Regex("[^\\p{ASCII}]")

public fun <K, V> JavaPairRDD<K, V>.combineByKeyIntoList(): JavaPairRDD<K, MutableList<V>> {

    val createListCombiner = object : Function<V, MutableList<V>> {
        override fun call(x: V): MutableList<V> {
            val res = mutableListOf<V>()
            res.add(x)
            return res
        }
    };

    val addCombiner = object : Function2<MutableList<V>, V, MutableList<V>> {
        override fun call(list: MutableList<V>, x: V): MutableList<V> {
            list.add(x)
            return list
        }
    };

    val combine = object : Function2<MutableList<V>, MutableList<V>, MutableList<V>> {
        override fun call(list1: MutableList<V>, list2: MutableList<V>): MutableList<V> {
            list1.addAll(list2)
            return list1
        }
    };

    val combinedKeysWithLists = this.combineByKey(createListCombiner, addCombiner, combine);

    return combinedKeysWithLists

}


public fun cleantText(input: String) : String{
    return Normalizer.normalize(input, Normalizer.Form.NFD).replace(normalizerRegex, "")
}
