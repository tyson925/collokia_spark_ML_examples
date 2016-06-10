package uy.com.collokia.scala;

import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

public class ClassTagger {
    public static <T> ClassTag<T> scalaClassTag(Class<T> c) {
        return ClassTag$.MODULE$.apply(c);
    }
}

