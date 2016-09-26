package uy.com.collokia.ml.classification.nlp.vtm

//import org.apache.spark.mllib.linalg.SparseVector
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.ml.regression.LabeledPoint
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import scala.Tuple2
import uy.com.collokia.ml.classification.nlp.OwnNGram
import uy.com.collokia.util.DocumentRow
import uy.com.collokia.util.featureCol
import uy.com.collokia.util.labelIndexCol


@Suppress("UNUSED_VARIABLE")

val tokenizerOutputCol = "words"
val removeOutputCol = "filteredWords"
val ngramOutputCol = "ngrams"
val cvModelOutputCol = "tfFeatures"
val contentOutputCol = "content_features"
val titleTokenizerOutputCol = "title_words"
val titleRemoverOutputCol = "filtered_titleWords"
val titleNgramsOutputCol = "title_ngrams"
val titleCvModelOutputCol = "tf_titleFeatures"
val titleOutputCol = "title_Features"
val tagTokenizerOutputCol = "tag_words"
val tagCvModelOutputCol = "tag_tfFeatures"
val tagOutputCol = "tag_features"
val CONTENT_VTM_VOC_SIZE = 2000
val TITLE_VTM_VOC_SIZE = 800
val TAG_VTM_VOC_SIZE = 400

fun extractFeaturesFromCorpus(textDataFrame: Dataset<DocumentRow>): Dataset<Row> {

    val indexer = StringIndexer().setInputCol(DocumentRow::category.name).setOutputCol("categoryIndex").fit(textDataFrame)
    println(indexer.labels().joinToString("\t"))

    val indexedTextDataFrame = indexer.transform(textDataFrame)

    val tokenizer = Tokenizer().setInputCol(DocumentRow::content.name).setOutputCol(tokenizerOutputCol)
    val wordsDataFrame = tokenizer.transform(indexedTextDataFrame)

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(featureCol)

    val filteredWordsDataFrame = remover.transform(wordsDataFrame)

    //val ngramTransformer = NGram().setInputCol("filteredWords").setOutputCol("ngrams").setN(4)
    val ngramTransformer = OwnNGram().setInputCol(remover.outputCol).setOutputCol(ngramOutputCol)

    //       val ngramsDataFrame = ngramTransformer.transform(filteredWordsDataFrame)
    val ngramsDataFrame = ngramTransformer.transform(wordsDataFrame)

    //return ngramsDataFrame
    return filteredWordsDataFrame
}

fun constructVTMPipeline(stopwords: Array<String>, vocabSize : Int): Pipeline {
    val indexer = StringIndexer().setInputCol(DocumentRow::category.name).setOutputCol(labelIndexCol)

    val tokenizer = RegexTokenizer().setInputCol(DocumentRow::content.name).setOutputCol(tokenizerOutputCol)
            .setMinTokenLength(3)
            .setToLowercase(false)
            .setPattern("\\w+")
            .setGaps(false)

    val stopwordsApplied = if (stopwords.size == 0) {
        println("Load default english stopwords...")
        StopWordsRemover.loadDefaultStopWords("english")
    } else {
        println("Load stopwords...")
        stopwords
    }

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(removeOutputCol)
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)

    val ngram = OwnNGram().setInputCol(remover.outputCol).setOutputCol(ngramOutputCol)

    val cvModel = CountVectorizer().setInputCol(ngram.outputCol)
            .setVocabSize(vocabSize)
            .setMinDF(3.0)
            .setOutputCol(cvModelOutputCol)

    //it is useless
    //val idf = IDF().setInputCol(cvModel.outputCol).setOutputCol("idfFeatures").setMinDocFreq(3)

    val normalizer = Normalizer().setInputCol(cvModel.outputCol).setOutputCol(contentOutputCol).setP(1.0)
    val scaler = StandardScaler()
            .setInputCol(cvModel.outputCol)
            .setOutputCol(contentOutputCol)
            .setWithStd(true)
            .setWithMean(false)

    val pipeline = Pipeline().setStages(arrayOf(indexer, tokenizer, remover, ngram, cvModel, scaler))

    return pipeline
}

fun constructTitleVtmDataPipeline(stopwords: Array<String>, vocabSize : Int): Pipeline {

    val stopwordsApplied = if (stopwords.size == 0) {
        println("Load default english stopwords...")
        StopWordsRemover.loadDefaultStopWords("english")
    } else {
        println("Load stopwords...")
        stopwords
    }

    val titleTokenizer = RegexTokenizer().setInputCol(DocumentRow::title.name).setOutputCol(titleTokenizerOutputCol)
            .setMinTokenLength(3)
            .setToLowercase(true)
            .setPattern("\\w+")
            .setGaps(false)

    val titleRemover = StopWordsRemover().setInputCol(titleTokenizer.outputCol)
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)
            .setOutputCol(titleRemoverOutputCol)

    val ngram = OwnNGram().setInputCol(titleRemover.outputCol).setOutputCol(titleNgramsOutputCol)

    //val concatWs = ConcatWSTransformer().setInputCols(arrayOf(titleRemover.outputCol, ngram.outputCol)).setOutputCol("title_bigrams")

    val titleCVModel = CountVectorizer().setInputCol(ngram.outputCol)
            .setOutputCol(titleCvModelOutputCol)
            .setVocabSize(vocabSize)
            .setMinDF(2.0)

    val titleNormalizer = Normalizer().setInputCol(titleCVModel.outputCol)
            .setOutputCol(titleOutputCol)
            .setP(1.0)

    val scaler = StandardScaler()
            .setInputCol(titleCVModel.outputCol)
            .setOutputCol(titleOutputCol)
            .setWithStd(true)
            .setWithMean(false)

    val pipeline = Pipeline().setStages(arrayOf(titleTokenizer, titleRemover, ngram, titleCVModel, scaler))
    return pipeline
}

fun constructTagVtmDataPipeline(vocabSize : Int): Pipeline {
    val tagTokenizer = RegexTokenizer().setInputCol(DocumentRow::labels.name).setOutputCol(tagTokenizerOutputCol)
            .setMinTokenLength(2)
            .setToLowercase(true)
            .setPattern("\\w+")
            .setGaps(false)

    //val ngram = NGram().setInputCol(tagTokenizer.setOutputCol).setOutputCol("tag_ngrams").setN(3)

    val tagCVModel = CountVectorizer().setInputCol(tagTokenizer.outputCol)
            .setOutputCol(tagCvModelOutputCol)
            .setVocabSize(vocabSize)
            .setMinDF(1.0)

    val tagNormalizer = Normalizer().setInputCol(tagCVModel.outputCol)
            .setOutputCol(tagOutputCol)
            .setP(1.0)

    val scaler = StandardScaler()
            .setInputCol(tagCVModel.outputCol)
            .setOutputCol(tagOutputCol)
            .setWithStd(true)
            .setWithMean(false)

    val pipeline = Pipeline().setStages(arrayOf(tagTokenizer, tagCVModel, scaler))
    return pipeline

}


fun convertDataFrameToLabeledPoints(data: Dataset<Row>): JavaRDD<LabeledPoint> {
    val converter = IndexToString()
            .setInputCol("categoryIndex")
            .setOutputCol("originalCategory")
    val converted = converter.transform(data)


    val featureData = converted.select("normIdfFeatures", "categoryIndex", "originalCategory")

    val labeledDataPoints = featureData.toJavaRDD().map({ feature ->
        val features = feature.getAs<SparseVector>(0)
        val label = feature.getDouble(1)
//            println(label)
        LabeledPoint(label, org.apache.spark.mllib.linalg.SparseVector(features.size(), features.indices(), features.values()))
    })

    println("number of testData: " + labeledDataPoints.count())

    val labelStat = featureData.select("originalCategory").javaRDD().mapToPair { label ->
        Tuple2(label.getString(0), 1L)
    }.reduceByKey { a, b -> a + b }

    println(labelStat.collectAsMap())

    return labeledDataPoints
}

fun setTfIdfModel(corpus: Dataset<Row>): IDFModel {
    val idf = IDF().setInputCol("tfFeatures").setOutputCol("idfFeatures").setMinDocFreq(3)

    val idfModel = idf.fit(corpus)

    return idfModel
}


