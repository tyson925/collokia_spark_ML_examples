package uy.com.collokia.coreNlp.stanford


//import scala.annotation.meta.field
import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Column
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import java.io.Serializable
import java.util.*

public data class ParsedSentenceBean(val category : String, val categoryIndex : Double, val content : String,var tokens: List<String>, var poses: List<String>, var lemmas: List<String>,
                                     var parses: List<String>, var ners: List<String>) : Serializable


public class CoreNLP : Transformer {
    override fun copy(p0: ParamMap): Transformer {
        return CoreNLP(sparkSession, annotations)
    }

    override fun transformSchema(p0: StructType?): StructType? {
        //throw UnsupportedOperationException()
        var res = p0?.add(DataTypes.createStructField("tokens", DataTypes.createArrayType(DataTypes.StringType), false))
        res = res?.add(DataTypes.createStructField("poses", DataTypes.createArrayType(DataTypes.StringType), false))
        res = res?.add(DataTypes.createStructField("lemmas", DataTypes.createArrayType(DataTypes.StringType), false))
        res = res?.add(DataTypes.createStructField("parses", DataTypes.createArrayType(DataTypes.StringType), false))
        res = res?.add(DataTypes.createStructField("ners", DataTypes.createArrayType(DataTypes.StringType), false))
        //res = res?.add(DataTypes.createStructField("null", DataTypes.NullType, true))

        return res
    }

    var wrapper: StanfordCoreNLPWrapper
    var sparkSession: SparkSession
    var annotations: String
    //var inputColName: String by Delegates.notNull<String>()
    var inputColName: String? = null
    var outputCol: String? = "lemmas"

    constructor(sparkSession: SparkSession, annotations: String) {
        this.sparkSession = sparkSession
        this.annotations = annotations

        //CoreNLP("corenlp_" + UUID.randomUUID().toString().drop(12))
        val props = Properties()
        //tokenize,
        props.setProperty("annotators", "tokenize, ssplit, ${annotations}")
        wrapper = StanfordCoreNLPWrapper(props)
    }


    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {


        val paesedContent = dataset?.select(dataset.col("category"),dataset.col("categoryIndex"),dataset.col(inputColName))?.javaRDD()?.map { text ->
            val content = text.getString(text.fieldIndex(inputColName))
            val doc = Annotation(content)
            wrapper.get()?.annotate(doc)
            val sentences = doc.get(CoreAnnotations.SentencesAnnotation::class.java)
//            val numberOfSentences = sentences.size
            //val tokens = ArrayList<List<String>>(numberOfSentences)
            val tokens = LinkedList<String>()
            val poses = LinkedList<String>()
            val lemmas = LinkedList<String>()
            val parses = LinkedList<String>()
            val ners = LinkedList<String>()

            for ((sentenceIndex, sentence) in sentences.withIndex()) {
                val documentTokensList = sentence.get(CoreAnnotations.TokensAnnotation::class.java)
                //println(sentence)


                val tokenArray = documentTokensList.map { token ->
                    token.get(CoreAnnotations.TextAnnotation::class.java)
                }
                //tokens.add(sentenceIndex,tokenArray)
                tokens.addAll(tokenArray)

                val posArray = documentTokensList.map { token ->
                    token.get(CoreAnnotations.PartOfSpeechAnnotation::class.java)
                }

                poses.addAll(posArray)

                val lemmaArray = documentTokensList.map { token ->
                    token.get(CoreAnnotations.LemmaAnnotation::class.java)
                }
                lemmas.addAll(lemmaArray)

                if (annotations.contains("parse")) {

                    val dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation::class.java)
                    //SemanticGraph dependencies = sentence.get(CoreAnnotations.CoNLLDepAnnotation.class);
                    val edgeList = dependencies.edgeListSorted()

                    val parseArray = edgeList.map { edge ->
                        edge.toString()
                    }
                    parses.addAll(parseArray)
                }

                if (annotations.contains("ner")) {
                    val nerArray = documentTokensList.map { token ->
                        token.get(CoreAnnotations.NamedEntityTagAnnotation::class.java)
                    }
                    ners.addAll(nerArray)
                }
            }


            val res = if (!annotations.contains("parse") && !annotations.contains("ner")) {
                outputCol = "lemmas"
                ParsedSentenceBean(text.getString(0),text.getDouble(1),content,tokens, poses, lemmas, ArrayList<String>(), ArrayList<String>())
//                res = RowFactory.create(text.getLong(0), tokens, poses, lemmas)
            } else if (annotations.contains("parse") && !annotations.contains("ner")) {
                outputCol = "parses"
                //               res = RowFactory.create(text.getLong(0), tokens, poses, lemmas, parses)
                ParsedSentenceBean(text.getString(0),text.getDouble(1),content,tokens, poses, lemmas, parses, ArrayList<String>())
            } else if (annotations.contains("parse") && annotations.contains("ner")) {
                outputCol = "ners"
//                res = RowFactory.create(text.getLong(0), tokens, poses, lemmas, parses, ners)
                ParsedSentenceBean(text.getString(0),text.getDouble(1),content,tokens, poses, lemmas, parses, ners)
            } else {
                ParsedSentenceBean(text.getString(0),text.getDouble(1),content,tokens, ArrayList<String>(), ArrayList<String>(), ArrayList<String>(), ArrayList<String>())
            }
            res
        }
        //val encoder = Encoders.bean(ParsedSentenceBean::class.java)
        return sparkSession.createDataFrame(paesedContent, ParsedSentenceBean::class.java)
    }


    override fun uid(): String? {
        return ""
    }

    public fun setInputCol(inputCol: String): CoreNLP {
        inputColName = inputCol
        return this
    }


}



