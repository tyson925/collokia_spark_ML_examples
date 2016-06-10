package uy.com.collokia.scala

import scala.reflect.ClassTag

public inline fun <reified T : Any> scalaClassTag(): ClassTag<T> = ClassTagger.scalaClassTag(T::class.java)
