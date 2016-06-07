package uy.com.collokia.ml.util

import scala.Tuple2

public operator fun <T1, T2> Tuple2<T1, T2>.component1(): T1 = this._1
public operator fun <T1, T2> Tuple2<T1, T2>.component2(): T2 = this._2




