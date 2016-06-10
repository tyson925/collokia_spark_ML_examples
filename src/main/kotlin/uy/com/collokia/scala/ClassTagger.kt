package uy.com.collokia.scala

import scala.reflect.ClassTag
import uy.com.collokia.scala.klass.ClassTagger


public inline fun <reified T : Any> scalaClassTag(): ClassTag<T> = ClassTagger.scalaClassTag(T::class.java)
