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

fun extractFeaturesFromCorpus(textDataFrame: Dataset<DocumentRow>): Dataset<Row> {

    val indexer = StringIndexer().setInputCol(DocumentRow::category.name).setOutputCol("categoryIndex").fit(textDataFrame)
    println(indexer.labels().joinToString("\t"))

    val indexedTextDataFrame = indexer.transform(textDataFrame)

    val tokenizer = Tokenizer().setInputCol(DocumentRow::content.name).setOutputCol("words")
    val wordsDataFrame = tokenizer.transform(indexedTextDataFrame)

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(featureCol)

    val filteredWordsDataFrame = remover.transform(wordsDataFrame)

    //val ngramTransformer = NGram().setInputCol("filteredWords").setOutputCol("ngrams").setN(4)
    val ngramTransformer = NGram().setInputCol("words").setOutputCol("ngrams").setN(4)

    //       val ngramsDataFrame = ngramTransformer.transform(filteredWordsDataFrame)
    val ngramsDataFrame = ngramTransformer.transform(wordsDataFrame)

    //return ngramsDataFrame
    return filteredWordsDataFrame
}


fun constructVTMPipeline(stopwords: Array<String>): Pipeline {
    val indexer = StringIndexer().setInputCol(DocumentRow::category.name).setOutputCol(labelIndexCol)

    val tokenizer = RegexTokenizer().setInputCol(DocumentRow::content.name).setOutputCol("words")
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

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol("filteredWords")
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)

    val ngram = OwnNGram().setInputCol(remover.outputCol).setOutputCol("ngrams")

    //val concatWs = ConcatWSTransformer().setInputCols(arrayOf(remover.outputCol, ngram.outputCol)).setOutputCol("bigrams")

    val cvModel = CountVectorizer().setInputCol(ngram.outputCol)
            .setOutputCol("tfFeatures")
            .setVocabSize(2000)
            .setMinDF(3.0)
    //val cvModel = CountVectorizer().setInputCol(remover.setOutputCol).setOutputCol(featureCol).setVocabSize(2000).setMinDF(2.0)

    val idf = IDF().setInputCol(cvModel.outputCol).setOutputCol("idfFeatures").setMinDocFreq(3)

    val normalizer = Normalizer().setInputCol(idf.outputCol).setOutputCol("content_features").setP(1.0)
    val scaler = StandardScaler()
            .setInputCol(cvModel.outputCol)
            .setOutputCol("content_features")
            .setWithStd(true)
            .setWithMean(false)

    val pipeline = Pipeline().setStages(arrayOf(indexer, tokenizer, remover, ngram, cvModel, idf, normalizer))

    return pipeline
}

fun constructTitleVtmDataPipeline(stopwords: Array<String>): Pipeline {

    val stopwordsApplied = if (stopwords.size == 0) {
        println("Load default english stopwords...")
        StopWordsRemover.loadDefaultStopWords("english")
    } else {
        println("Load stopwords...")
        stopwords
    }

    val titleTokenizer = RegexTokenizer().setInputCol(DocumentRow::title.name).setOutputCol("title_words")
            .setMinTokenLength(3)
            .setToLowercase(true)
            .setPattern("\\w+")
            .setGaps(false)

    val titleRemover = StopWordsRemover().setInputCol(titleTokenizer.outputCol)
            .setOutputCol("filtered_titleWords")
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)

    val ngram = OwnNGram().setInputCol(titleRemover.outputCol).setOutputCol("title_ngrams")

    //val concatWs = ConcatWSTransformer().setInputCols(arrayOf(titleRemover.outputCol, ngram.outputCol)).setOutputCol("title_bigrams")

    val titleCVModel = CountVectorizer().setInputCol(ngram.outputCol)
            .setOutputCol("tf_titleFeatures")
            .setVocabSize(1000)
            .setMinDF(2.0)

    val titleNormalizer = Normalizer().setInputCol(titleCVModel.outputCol)
            .setOutputCol("title_features")
            .setP(1.0)

    val pipeline = Pipeline().setStages(arrayOf(titleTokenizer, titleRemover, ngram, titleCVModel, titleNormalizer))
    return pipeline
}

fun constructTagVtmDataPipeline(): Pipeline {
    val tagTokenizer = RegexTokenizer().setInputCol(DocumentRow::labels.name).setOutputCol("tag_words")
            .setMinTokenLength(2)
            .setToLowercase(true)
            .setPattern("\\w+")
            .setGaps(false)

    //val ngram = NGram().setInputCol(tagTokenizer.setOutputCol).setOutputCol("tag_ngrams").setN(3)

    val tagCVModel = CountVectorizer().setInputCol(tagTokenizer.outputCol)
            .setOutputCol("tag_tfFeatures")
            .setVocabSize(1000)
            .setMinDF(1.0)

    val tagNormalizer = Normalizer().setInputCol(tagCVModel.outputCol)
            .setOutputCol("tag_features")
            .setP(1.0)

    val pipeline = Pipeline().setStages(arrayOf(tagTokenizer, tagCVModel, tagNormalizer))
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


