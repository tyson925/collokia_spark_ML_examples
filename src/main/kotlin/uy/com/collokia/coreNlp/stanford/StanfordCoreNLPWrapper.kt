package uy.com.collokia.coreNlp.stanford


import java.io.Serializable
import java.util.*
import edu.stanford.nlp.pipeline.StanfordCoreNLP

public class StanfordCoreNLPWrapper(private val props: Properties) : Serializable {

    @Transient private var coreNLP: StanfordCoreNLP? = null

    fun get() : StanfordCoreNLP? {

        if (coreNLP == null) {
            coreNLP = StanfordCoreNLP(props)
        }
        return coreNLP
    }
}
