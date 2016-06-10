package uy.com.collokia.ml.lsa

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.pipeline.Annotation
import scala.Tuple2
import uy.com.collokia.ml.util.component1
import uy.com.collokia.ml.util.component2
import java.io.File
import java.util.*
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat


public  val  START_TAG_KEY = "xmlinput.start";
public  val END_TAG_KEY = "xmlinput.end";

fun loadStopWords(path: String): Set<String> {
    return File(path).readLines().toSet()
}

public fun saveDocFreqs(path: String, docFreqs: Array<Tuple2<String, Int>>) {

    File(path).bufferedWriter().use { writer ->
        docFreqs.forEach { docFreq ->
            val (doc, freq) = docFreq
            writer.write("$doc\t$freq\n")
        }
    }
}

public fun readFile(path: String, jsc: JavaSparkContext): JavaRDD<String> {
    val conf = Configuration()
    conf.set(START_TAG_KEY, "<page>")
    conf.set(END_TAG_KEY, "</page>")
    val rawXmls = jsc.newAPIHadoopFile(path, TextInputFormat::class.java, LongWritable::class.java,
            Text::class.java, conf)
    return rawXmls.map({p -> p._2.toString()})
}

public fun createNLPPipeline(): StanfordCoreNLP {
    val props = Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    return StanfordCoreNLP(props)
}

public fun plainTextToLemmas(text: String, stopWords: Set<String>, pipeline: StanfordCoreNLP) : List<String> {
    val doc = Annotation(text)
    pipeline.annotate(doc)
    val sentences = doc.get(CoreAnnotations.SentencesAnnotation::class.java)
    val lemmas = sentences.flatMap { sentence ->
        val tokens = sentence.get(CoreAnnotations.TokensAnnotation::class.java)
        tokens.map { token ->
            token.lemma().toLowerCase()
        }
    }.filter { lemma -> lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma) }

    return lemmas
}

public fun isOnlyLetters(str: String): Boolean {
    // While loop for high performance
    var i = 0
    val charArray = str.toCharArray()
    while (i < str.length) {
        if (!Character.isLetter(charArray[i])) {
            return false
        }
        i += 1
    }
    return true
}



