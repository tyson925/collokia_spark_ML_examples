package uy.com.collokia.ml.classification.nlp

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.runtime.AbstractFunction1
import java.io.Serializable
import java.util.*
import uy.com.collokia.common.utils.nlp.wordNGramms
import kotlin.properties.Delegates

class OwnNGram : UnaryTransformer<Array<String>, Array<String>, OwnNGram>(), DefaultParamsWritable {
    override fun save(p0: String?) {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun write(): MLWriter {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }


    override fun createTransformFunc(): Function1<Array<String>, Array<String>> {
        return ConvertFunction()
    }

    class ConvertFunction : AbstractFunction1<Array<String>, Array<String>>(), Serializable {
        override fun apply(p0: Array<String>?): Array<String>? {
            return wordNGrams(p0?.toList() ?: listOf(), 2, true, " ").toTypedArray()
        }

        fun wordNGrams(tokens: List<String>, N: Int, oneToN: Boolean, separator : String = "_"): List<String>
        {
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



    override fun outputDataType(): DataType {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String {
        return UUID.randomUUID().toString()
    }

}

