package uy.com.collokia.ml.util

import scala.Tuple2
import java.text.DecimalFormat

public operator fun <T1, T2> Tuple2<T1, T2>.component1(): T1 = this._1
public operator fun <T1, T2> Tuple2<T1, T2>.component2(): T2 = this._2

public inline fun <reified T : Any> measureTimeInMillis(functionToTime: () -> T): Pair<T, Long> {
    val begin = java.lang.System.currentTimeMillis()
    val ret = functionToTime()
    val end = java.lang.System.currentTimeMillis()
    return Pair(ret, end - begin)
}

val formatterToTimePrint = DecimalFormat("#0.00000");



