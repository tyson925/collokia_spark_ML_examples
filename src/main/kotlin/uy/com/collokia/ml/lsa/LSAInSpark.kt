package uy.com.collokia.ml.lsa

public class LSAInSpark() {

    companion object {
        @JvmStatic fun main(args: Array<String>) {

        }

    }

    public fun runLSA() {

    }

    /**
     * Returns an RDD of rows of the document-term matrix, a mapping of column indices to terms, and a
     * mapping of row IDs to document titles.
     */
    /*public fun preprocessing(sampleSize: Double, numTerms: Int, jsc: JavaSparkContext)
    : (JavaRDD<Vector>, Map<Int, String>, Map<Long, String>, Map<String, Double>) {
        val pages = readFile("hdfs:///user/ds/Wikipedia/", jsc)
                .sample(false, sampleSize, 11L)

        val plainText = pages.filter({page -> page != null}).flatMap(wikiXmlToPlainText)

        val stopWords = jsc.broadcast(loadStopWords("stopwords.txt")).value

        val lemmatized = plainText.mapPartitions(iter => {
            val pipeline = createNLPPipeline()
            iter.map{ case(title, contents) => (title, plainTextToLemmas(contents, stopWords, pipeline))}
        })

        val filtered = lemmatized.filter(_._2.size > 1)

        documentTermMatrix(filtered, stopWords, numTerms, sc)
    }*/

}



