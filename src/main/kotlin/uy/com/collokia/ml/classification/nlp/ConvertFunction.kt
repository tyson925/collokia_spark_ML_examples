package uy.com.collokia.ml.classification.nlp

import scala.collection.mutable.WrappedArray
import scala.runtime.AbstractFunction1
import java.io.Serializable
import java.util.*

class ConvertFunction : AbstractFunction1<WrappedArray<String>, Array<String>>(), Serializable {
    override fun apply(p0: WrappedArray<String>?): Array<String> {

        return wordNGrams(scala.collection.JavaConversions.seqAsJavaList(p0) ?: listOf(), 2, true, " ").toTypedArray()
    }

    fun wordNGrams(tokens: List<String>, N: Int, oneToN: Boolean, separator: String = "_"): List<String> {
        val RET = LinkedList<String>()

        for (i in (if (oneToN) 1 else N)..N + 1 - 1) {
            RET.addAll(wordNGramsLevel(tokens, i, separator))
        }

        return RET
    }


    /**
     * @param tokens
     * *
     * @param N
     * *
     * @return
     */
    private fun wordNGramsLevel(tokens: List<String>, N: Int, separator: String = "_"): List<String> {
        val RET: MutableList<String>

        if (N < 2) {
            RET = tokens.toMutableList()
        } else {
            RET = mutableListOf<String>()
            for (i in 0..tokens.size - N + 1 - 1) {
                val buf = StringBuffer()
                for (j in 0..N - 1) {
                    buf.append(tokens[i + j])
                    if (j < (N - 1)) {
                        buf.append(separator)
                    }
                }
                RET.add(buf.toString())
            }
        }

        return RET
    }
}