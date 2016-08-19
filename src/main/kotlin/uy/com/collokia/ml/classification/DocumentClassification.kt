package uy.com.collokia.ml.classification

//import org.apache.spark.mllib.linalg.SparseVector
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.ml.regression.LabeledPoint
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark
import scala.Tuple2
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.machineLearning.convertLabeledPointToArff
import uy.com.collokia.common.utils.machineLearning.saveArff
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.ml.logreg.LogisticRegressionInSpark
import uy.com.collokia.ml.rdf.DecisionTreeInSpark
import uy.com.collokia.ml.rdf.RandomForestInSpark
import uy.com.collokia.ml.svm.SVMSpark
import uy.com.collokia.util.REUTERS_DATA
import java.io.File
import java.io.Serializable

public data class ReutersDocument(val title: String?, var body: String?, val date: String,
                                  val topics: List<String>?, val places: List<String>?, val organisations: List<String>?, val id: Int) : Serializable

//required "var" according to `Encoders.bean`
public data class DocumentRow(var category: String, var content: String) : Serializable

public data class ClassifierResults(val category: String, val decisiontTree: Double, val randomForest: Double, val svm: Double, val logReg: Double) : Serializable

public val VTM_PIPELINE = "./data/model/vtmPipeLine"

@Suppress("UNUSED_VARIABLE")
public class DocumentClassification() : Serializable {

    companion object {
        val MAPPER = jacksonObjectMapper()
        val topCategories = listOf("ship", "grain", "money-fx", "corn", "trade", "crude", "earn", "wheat", "acq", "interest")
        //val topCategories = listOf("earn", "acq")
        public val featureCol = "normIdfFeatures"
        public val labelIndexCol = "categoryIndex"

    }

    public fun readDzoneFromEs(sparkSession: SparkSession,jsc: JavaSparkContext) : Dataset<DocumentRow> {
        val corpusRow = JavaEsSpark.esRDD(jsc, "dzone_data/DocumentRow").map { line ->
            val (id, map) = line
            val category = map.getOrElse("category") { "other" } as String
            val content = map.getOrElse("lemmas") { "other" } as String
            val title = map.getOrElse("title") { "other" } as String
            val taggedTitle = title.split(Regex("W")).map { titleToken ->
                "title:${titleToken}"
            }.joinToString(" ")
            val labels = map.getOrElse("labels") { listOf<String>() } as List<String>
            val taggedLabels = labels.map { label ->
                "label:${label}"
            }.joinToString(" ")
            DocumentRow(category, content + "\n" + taggedTitle + "\n" + taggedLabels)
        }
        return documentRddToDF(sparkSession,corpusRow)
    }

    public fun parseCorpus(sparkSession: SparkSession, corpusInRaw: JavaRDD<String>, subTopic: String?): Dataset<DocumentRow> {

        val corpusRow = subTopic?.let {
            filterToOneCategory(corpusInRaw, subTopic)
        } ?: filterToTopCategories(corpusInRaw)

        corpusInRaw.unpersist()

        return documentRddToDF(sparkSession, corpusRow)
    }

    public fun documentRddToDF(sparkSession: SparkSession, corpusRow: JavaRDD<DocumentRow>) : Dataset<DocumentRow> {
        println("corpus size: " + corpusRow.count())

        val reutersEncoder = Encoders.bean(DocumentRow::class.java)

        val textDataFrame = sparkSession.createDataset(corpusRow.rdd(), reutersEncoder)

        corpusRow.unpersist()

        return textDataFrame
    }

    private fun filterToOneCategory(corpusInRaw: JavaRDD<String>, category: String): JavaRDD<DocumentRow> {
        val corpusRow = corpusInRaw.map { line ->
            val doc = MAPPER.readValue(line, ReutersDocument::class.java)
            val topics = doc.topics?.intersect(topCategories) ?: listOf<String>()
            val content = doc.body + (doc.title ?: "")

            val row = if (topics.contains(category)) {
                DocumentRow(category, content)
            } else {
                DocumentRow("other", content)
            }
            row
        }
        return corpusRow
    }

    private fun filterToTopCategories(corpusInRaw: JavaRDD<String>): JavaRDD<DocumentRow> {
        val corpusRow = corpusInRaw.map { line ->
            val doc = MAPPER.readValue(line, ReutersDocument::class.java)
            val topics = doc.topics?.intersect(topCategories) ?: listOf<String>()
            val content = doc.body + (doc.title ?: "")

            val intersectCategory = topics.intersect(topCategories)
            intersectCategory.first()
            val rows = intersectCategory.map { category ->
                DocumentRow(category, content)
            }.iterator()
            DocumentRow(intersectCategory.first(), content)
            //rows
        }
        return corpusRow
    }

    public fun exractFeaturesFromCorpus(textDataFrame: Dataset<DocumentRow>): Dataset<Row> {

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

    public fun reutersDataEvaulation(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile("./testData/reuters/json/reuters.json").cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val results = topCategories.map { category ->
            val parsedCorpus = exractFeaturesFromCorpus(parseCorpus(sparkSession, corpusInRaw, category))

            val cvModel = CountVectorizer().setInputCol(featureCol).setOutputCol("tfFeatures").setVocabSize(2000).setMinDF(3.0)
                    .fit(parsedCorpus)

            val hashedCorpusDF = cvModel.transform(parsedCorpus)

            parsedCorpus.unpersist()

            val (hashedTrainDF, hashedTestDF) = hashedCorpusDF.randomSplit(doubleArrayOf(0.9, 0.1))

            val idfModel = setTfIdfModel(hashedTrainDF)

            val trainTfIdfDF = idfModel.transform(hashedTrainDF)

            hashedTrainDF.unpersist()

            val normalizer = Normalizer().setInputCol(idfModel.outputCol).setOutputCol("normIdfFeatures").setP(1.0)
            //val normalizer = Normalizer().setInputCol("features").setOutputCol("normIdfFeatures").setP(1.0)

            val normTrainTfIdfDF = normalizer.transform(trainTfIdfDF)

            trainTfIdfDF.unpersist()

            val trainData = convertDataFrameToLabeledPoints(normTrainTfIdfDF).cache()

            normTrainTfIdfDF.unpersist()

            val testTfIdfDF = idfModel.transform(hashedTestDF)
            val normTestTfIdfDF = normalizer.transform(testTfIdfDF)
            testTfIdfDF.unpersist()

            val testData = convertDataFrameToLabeledPoints(normTestTfIdfDF).cache()

            val dt = DecisionTreeInSpark()

            val Fmeasure = dt.evaulateDecisionTreeModel(trainData, testData, 2)

            trainData.unpersist()
            testData.unpersist()
            Pair(category, Fmeasure)

        }.joinToString("\n")

        println(results)
    }

    public fun tenFoldReutersDataEvaulationWithClassifiers(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile(REUTERS_DATA).cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val decisionTree = DecisionTreeInSpark()
        val randomForest = RandomForestInSpark()
        val svm = SVMSpark()
        val logReg = LogisticRegressionInSpark()

        val results = topCategories.map { category ->

            val data = convertDataFrameToLabeledPoints(constructVTMData(sparkSession, corpusInRaw, category))

            val dtFMeasure = decisionTree.evaulate10Fold(data)
            val rfFMeasure = randomForest.evaulate10Fold(data)
            val svmFMeasure = svm.evaulate10Fold(data)
            val logRegFMeasure = logReg.evaulate10Fold(data)

            data.unpersist()
            ClassifierResults(category, dtFMeasure, rfFMeasure, svmFMeasure, logRegFMeasure)

        }

        println(results.joinToString("\n"))

    }

    public fun tenFoldReutersDataEvaulation(jsc: JavaSparkContext) {
        val corpusInRaw = jsc.textFile(REUTERS_DATA).cache().repartition(8)

        val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()

        val decisionTree = DecisionTreeInSpark()

        val logisticRegression = LogisticRegressionInSpark()

        val results = topCategories.map { category ->

            val data = convertDataFrameToLabeledPoints(constructVTMData(sparkSession, corpusInRaw, category))

            val arffData = convertLabeledPointToArff(data)
            saveArff(arffData, "./testData/reuters/arff/${category}.arff")

            //val fMeasure = decisionTree.evaulate10Fold(data)
            val fMeasure = logisticRegression.evaulate10Fold(data)

            //val Fmeasure = 1.0

            data.unpersist()
            Pair(category, fMeasure)

        }.joinToString("\n")

        println(results)
    }

    public fun constructVTMPipeline(sparkSession : SparkSession): Pipeline {
        val indexer = StringIndexer().setInputCol(DocumentRow::category.name).setOutputCol(labelIndexCol)

        val tokenizer = Tokenizer().setInputCol(DocumentRow::content.name).setOutputCol("words")

        //val coreNLP = CoreNLP(sparkSession, "pos, lemma").setInputCol(DocumentRow::content.name)

        val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol("filteredWords").setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
        println(StopWordsRemover.loadDefaultStopWords("english").toList())
        //val ngram = NGram().setInputCol(remover.outputCol).setOutputCol("ngrams").setN(3)

        //val cvModel = CountVectorizer().setInputCol(ngram.outputCol).setOutputCol("tfFeatures").setVocabSize(2000).setMinDF(2.0)
        val cvModel = CountVectorizer().setInputCol(remover.outputCol).setOutputCol("tfFeatures").setVocabSize(2000).setMinDF(3.0)
        //val cvModel = CountVectorizer().setInputCol(remover.outputCol).setOutputCol(featureCol).setVocabSize(1000).setMinDF(3.0)

        val idf = IDF().setInputCol(cvModel.outputCol).setOutputCol("idfFeatures").setMinDocFreq(3)

        val normalizer = Normalizer().setInputCol(idf.outputCol).setOutputCol(featureCol).setP(1.0)
        val scaler = StandardScaler()
                .setInputCol(cvModel.outputCol)
                .setOutputCol(featureCol)
                .setWithStd(true)
                .setWithMean(false)
        //val normalizer = Normalizer().setInputCol(cvModel.outputCol).setOutputCol(featureCol).setP(1.0)

        //val pipeline = Pipeline().setStages(arrayOf(indexer,coreNLP, remover,ngram, cvModel, idf, normalizer))
        val pipeline = Pipeline().setStages(arrayOf(indexer,tokenizer, remover,cvModel,scaler))

        return pipeline
    }

    public fun constructVTMData(sparkSession: SparkSession, corpusInRaw: JavaRDD<String>, category: String?): Dataset<Row> {
        val parsedCorpus = parseCorpus(sparkSession, corpusInRaw, category)

        corpusInRaw.unpersist()

        println("category:\t${category}")

        val vtmPipeline = constructVTMPipeline(sparkSession)
        val data = vtmPipeline.fit(parsedCorpus).transform(parsedCorpus)
        data.show(3)
        return data
    }


    public fun convertDataFrameToLabeledPoints(data: Dataset<Row>): JavaRDD<org.apache.spark.mllib.regression.LabeledPoint> {
        val converter = IndexToString()
                .setInputCol("categoryIndex")
                .setOutputCol("originalCategory")
        val converted = converter.transform(data)


        val featureData = converted.select("normIdfFeatures", "categoryIndex", "originalCategory")

        val labeledDataPoints = featureData.toJavaRDD().map({ feature ->
            val features = feature.getAs<SparseVector>(0)
            val label = feature.getDouble(1)
//            println(label)
            org.apache.spark.mllib.regression.LabeledPoint(label, org.apache.spark.mllib.linalg.SparseVector(features.size(), features.indices(), features.values()))
        })

        println("number of testData: " + labeledDataPoints.count())

        val labelStat = featureData.select("originalCategory").javaRDD().mapToPair { label ->
            Tuple2(label.getString(0), 1L)
        }.reduceByKey { a, b -> a + b }

        println(labelStat.collectAsMap())

        return labeledDataPoints
    }

    public fun setTfIdfModel(corpus: Dataset<Row>): IDFModel {
        val idf = IDF().setInputCol("tfFeatures").setOutputCol("idfFeatures").setMinDocFreq(3)

        val idfModel = idf.fit(corpus)

        return idfModel
    }


    public fun readJson() {
        var index = 0
        File("./testData/reuters/json/reuters.json").bufferedWriter().use { writer ->
            val time = File("./testData/reuters/").listFiles().filter { file -> file.name.endsWith(".json") }.forEach { file ->
                val jsons = file.readLines().joinToString("\n").split("},").toMutableList()
                jsons[0] = jsons[0].substring(1)
                jsons[jsons.lastIndex] = jsons[jsons.lastIndex].substringBeforeLast("]")

                jsons.forEach { json ->
                    //println(cleanJson(json))
                    val reutersDoc = MAPPER.readValue(cleanJson(json), ReutersDocument::class.java)

                    reutersDoc.body?.let { body ->
                        println(index++)
                        reutersDoc.body = reutersDoc.body?.replace("\n", "")
                        if (reutersDoc.topics != null && reutersDoc.topics.intersect(topCategories).isNotEmpty()) {
                            writer.write(MAPPER.writeValueAsString(reutersDoc) + "\n")
                        }
                    }
                }
            }
        }
    }


    private fun cleanJson(json: String): String {
        return json.replace("\n", " ").replace("\\\n", " ").replace("\u0003", "") + "}"
    }

    public fun runOnSpark() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("reutersTest").setMaster("local[8]")
                    .set("es.nodes", "localhost:9200").set("es.nodes.discovery", "false")

            val jsc = JavaSparkContext(sparkConf)


            val sparkSession = SparkSession.builder().master("local").appName("reuters classification").getOrCreate()


            val test = JavaEsSpark.esRDD(jsc, "dzone_data/DocumentRow").mapToPair { line ->
                Tuple2(line._2["category"], 1)
            }.groupByKey()
            println(test.take(11).joinToString("\n"))
            //val test = sparkSession.read().format("org.elasticsearch.spark.sql").option(
            //        "es.field.read.as.array.exclude","labels").load("dzone_data/Article")
// inspect the data
            println(test.count())

            //val testData = parseCorpus(jsc)

            //reutersDataEvaulation(jsc)
            //tenFoldReutersDataEvaulation(jsc)
            //tenFoldReutersDataEvaulationWithClassifiers(jsc)
            //val dt = DecisionTreeInSpark()
            //dt.evaulateSimpleForest(testData)
            //dt.evaluate(trainData, testData, testData, 10)
            //println(dt.classProbabilities(trainData).joinToString("\n"))
            //dt.buildDecisionTreeModel(trainData,testData,10)
        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }

}

fun main(args: Array<String>) {

    val docClassifier = DocumentClassification()
    //docClassifier.readJson()
    docClassifier.runOnSpark()

}

