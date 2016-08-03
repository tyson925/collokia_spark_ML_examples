
package uy.com.collokia.ml.collaborativeFiltering.lastFm

import org.apache.log4j.BasicConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.DoubleFunction
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2
import uy.com.collokia.util.component1
import uy.com.collokia.util.component2
import uy.com.collokia.util.formatterToTimePrint
import uy.com.collokia.util.measureTimeInMillis
import java.util.*


public class LastFmALS() {

    public fun preparation(rawUserArtistData: JavaRDD<String>, rawArtistData: JavaRDD<String>, rawArtistAlias: JavaRDD<String>) {
        val userIDStats = rawUserArtistData.mapToDouble<Double>(DoubleFunction
        { row -> row.split(" ")[0].toDouble() }).stats()
        val itemIDStats = rawUserArtistData.mapToDouble<Double>(DoubleFunction
        { row -> row.split(" ")[0].toDouble() }).stats()
        println(userIDStats)
        println(itemIDStats)

        val artistByID = buildArtistByID(rawArtistData)
        val artistAlias = buildArtistAlias(rawArtistAlias)

        artistByID.count()
    }


    public fun buildArtistByID(rawArtistData: JavaRDD<String>): JavaPairRDD<Int, String> {
        return rawArtistData.mapToPair { line ->
            val res = if (line.split("\t").size == 2) {
                val (id, name) = line.split("\t")
                if (name.isEmpty()) {
                    Tuple2(0, "")
                } else {
                    try {
                        Tuple2(id.toInt(), name.trim())
                    } catch (e: NumberFormatException) {
                        Tuple2(0, "")
                    }
                }
            } else {
                Tuple2(0, "")
            }
            res
        }
    }

    public fun buildArtistAlias(rawArtistAlias: JavaRDD<String>): Map<Int, Int> {
        return rawArtistAlias.mapToPair { line ->
            val tokens = line.split('\t')
            if (tokens[0].isEmpty()) {
                Tuple2(0, 0)
            } else {
                Tuple2(tokens[0].toInt(), tokens[1].toInt())
            }
        }.collectAsMap()
    }

    public fun buildRatings(rawUserArtistData: JavaRDD<String>, bArtistAlias: Broadcast<Map<Int, Int>>): JavaRDD<Rating> {
        return rawUserArtistData.map { line ->
            val (userID, artistID, count) = line.split(' ').map({ item -> item.toInt() })
            val finalArtistID = bArtistAlias.value.getOrElse(artistID, { artistID })
            Rating(userID, finalArtistID, count.toDouble())
        }
    }


    public fun model(jsc: JavaSparkContext, rawUserArtistData: JavaRDD<String>, rawArtistData: JavaRDD<String>, rawArtistAlias: JavaRDD<String>) {

        val bArtistAlias = jsc.broadcast(buildArtistAlias(rawArtistAlias))

        val trainData = buildRatings(rawUserArtistData, bArtistAlias).cache()

        val rank = 10
        val numIterations = 10
        val lambda = 0.01
        val alpha = 1.0

        val model = ALS.trainImplicit(trainData.rdd(), rank, numIterations, lambda, -1, alpha)
        trainData.unpersist()

        println(model.userFeatures().toJavaRDD().take(1).first()._2.joinToString(", "))

        val userID = 2093760
        val recommendationsToUser = model.recommendProducts(userID, 5)
        println("The top 5. recommendation to user ${userID} are ${recommendationsToUser.joinToString("\n")}")

        val recommendedProductIDs = recommendationsToUser.map({ recommendation -> recommendation.product() }).toSet()

        val rawArtistsForUser = rawUserArtistData.map({ line -> line.split(' ') }).filter { lineArray ->
            lineArray[0].toInt() == userID
        }

        val existingProducts = rawArtistsForUser.map { lineArray -> lineArray[1].toInt() }.collect().toSet()

        val artistByIDs = buildArtistByID(rawArtistData)

        println("Existing products are the follows: \n ${artistByIDs.filter { artistByID -> existingProducts.contains(artistByID._1) }.
                values().collect().joinToString("\n")}")
        println("-------------------------------------")
        println("Recommended products to ${userID} are the follows: \n${artistByIDs.filter { artistByID -> recommendedProductIDs.contains(artistByID._1) }.
                values().collect().joinToString("\n")}")

        unpersist(model)
    }

    public fun areaUnderCurve(positiveData: JavaRDD<Rating>, bAllItemIDs: Broadcast<List<Int>>,
                              predictFunction: (JavaPairRDD<Int, Int>) -> JavaRDD<Rating>): Double {

        // What this actually computes is AUC, per user. The result is actually something
        // that might be called "mean AUC".

        // Take held-out testData as the "positive", and map to tuples
        val positiveUserProducts = positiveData.mapToPair({ rating -> Tuple2(rating.user(), rating.product()) })
        // Make predictions for each of them, including a numeric score, and gather by user

        val positivePredictions = predictFunction(positiveUserProducts).groupBy({ rating -> rating.user() })

        // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
        // small AUC problems, and it would be inefficient, when a direct computation is available.

        // Create a set of "negative" products for each user. These are randomly chosen
        // from among all of the other items, excluding those that are "positive" for the user.


        val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitionsToPair { userIDAndPosItemIDs ->

            // mapPartitions operates on many (user,positive-items) pairs at once
// Init an RNG and the item IDs set once for partition
            val random = Random()
            val allItemIDs = bAllItemIDs.value
            val res = mutableListOf<Tuple2<Int, Int>>()
            userIDAndPosItemIDs.forEach { item ->
                val (userID, posItemIDs) = item
                val posItemIDSet = posItemIDs.toSet()
                val negative = mutableListOf<Int>()
                var i = 0
                // Keep about as many negative examples per user as positive.
                // Duplicates are OK
                while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
                    val itemID = allItemIDs[random.nextInt(allItemIDs.size)]
                    if (!posItemIDSet.contains(itemID)) {
                        negative.add(itemID)
                    }
                    i += 1
                }
                // Result is a collection of (user,negative-item) tuples
                res.addAll(negative.map({ itemID -> Tuple2(userID, itemID) }))
            }
            res.iterator()
        }

// Make predictions on the rest:
        val negativePredictions = predictFunction(negativeUserProducts).groupBy({ rating -> rating.user() })

        // Join positive and negative by user
        return positivePredictions.join(negativePredictions).values().mapToDouble<Double>(DoubleFunction { positiveAndNegativRatings ->

            // AUC may be viewed as the probability that a random positive item scores
            // higher than a random negative one. Here the proportion of all positive-negative
            // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
            var correct = 0L
            var total = 0L

            // For each pairing,
            positiveAndNegativRatings._1.forEach { positive ->
                positiveAndNegativRatings._2.forEach { negative ->
                    // Count the correctly-ranked pairs
                    if (positive.rating() > negative.rating()) {
                        correct += 1
                    }
                }
                total += 1
            }
            // Return AUC: fraction of pairs ranked correctly
            correct.toDouble() / total
        }).mean() // Return mean AUC over users
    }

    public fun predictMostListened(jsc: JavaSparkContext, train: JavaRDD<Rating>): JavaRDD<Rating> {
        val bListenCount = jsc.broadcast(train.mapToPair({ rating -> Tuple2(rating.product(), rating.rating()) }).reduceByKey({ a, b -> a + b }).collectAsMap())
        return train.map { rating ->
            Rating(rating.user(), rating.product(), bListenCount.value.getOrElse(rating.product(), { 0.0 }))
        }
    }


    public fun evaluate(jsc: JavaSparkContext, rawUserArtistData: JavaRDD<String>,rawArtistAlias: JavaRDD<String>) {

        val bArtistAlias = jsc.broadcast(buildArtistAlias(rawArtistAlias))

        val allData = buildRatings(rawUserArtistData, bArtistAlias)
        val (trainData, cvData) = allData.randomSplit(doubleArrayOf(0.9, 0.1))
        trainData.cache()
        cvData.cache()

        val allItemIDs = allData.map({ rating -> rating.product() }).distinct().collect()
        val bAllItemIDs = jsc.broadcast(allItemIDs)

        val mostListenedAUC = areaUnderCurve(cvData, bAllItemIDs, { predictMostListened(jsc, trainData) })
        println(mostListenedAUC)


        val evaluations = intArrayOf(10, 50).flatMap { rank ->
            doubleArrayOf(0.0001, 1.0).flatMap { lambda ->
                doubleArrayOf(1.0, 40.0).map { alpha ->
                    println("Train ALS with parameters:\trank:\t${rank}\tlambda:\t${lambda}\talpha:\t${alpha}")
                    val model = ALS.trainImplicit(trainData.rdd(), rank, 10, lambda, alpha)
                    val auc = areaUnderCurve(cvData, bAllItemIDs, { model.predict(cvData.mapToPair({ rating -> Tuple2(rating.user(), rating.product()) })) })
                    unpersist(model)
                    Triple(rank, lambda, alpha) to auc
                }
            }
        }


        println(evaluations.sortedBy ({ it.second }).reversed().joinToString("\n"))

        trainData.unpersist()
        cvData.unpersist()

    }

    public fun recommend(jsc: JavaSparkContext, rawUserArtistData: JavaRDD<String>, rawArtistData: JavaRDD<String>, rawArtistAlias: JavaRDD<String>) {

        val bArtistAlias = jsc.broadcast(buildArtistAlias(rawArtistAlias))
        val allData = buildRatings(rawUserArtistData, bArtistAlias).cache()
        val model = ALS.trainImplicit(allData.rdd(), 50, 10, 1.0, 40.0)
        allData.unpersist()

        val userID = 2093760
        val recommendations = model.recommendProducts(userID, 5)

        val recommendedProductIDs = recommendations.map({ rating -> rating.product() }).toSet()

        val artistByIDs = buildArtistByID(rawArtistData)

        println(artistByIDs.filter { artistByID ->
            val (id, name) = artistByID
            recommendedProductIDs.contains(id)
        }.values().collect().joinToString("\n"))

        val someUsers = allData.map({rating -> rating.user()}).distinct().take(100)

        val someRecommendations = someUsers.map({userID -> model.recommendProducts(userID, 5)})
        println(someRecommendations.map{recs -> "${recs.first().user()} -> ${recs.map{rating -> rating.product()}.joinToString(", ")}" }.joinToString("\n"))

        unpersist(model)
    }

    public fun unpersist(model: MatrixFactorizationModel) {
        // At the moment, it's necessary to manually unpersist the RDDs inside the model
        // when done with it in order to make sure they are promptly uncached
        model.userFeatures().unpersist(true)
        model.productFeatures().unpersist(true)
    }


    public fun run() {
        val time = measureTimeInMillis {
            val sparkConf = SparkConf().setAppName("LastFMRecommendation").setMaster("local[6]")
            //sparkConf.set("spark.driver.maxResultSize", "2g")
            val jsc = JavaSparkContext(sparkConf)
            val rootDirectory = "./../ES/testData/profiledata_06-May-2005/"
            val rawUserArtistData = jsc.textFile(rootDirectory + "user_artist_data.txt")
            val rawArtistData = jsc.textFile(rootDirectory + "artist_data.txt")
            val rawArtistAlias = jsc.textFile(rootDirectory + "artist_alias.txt")

            //println(rawArtistAlias.count())

            //preparation(rawUserArtistData, rawArtistData, rawArtistAlias)
            //model(jsc, rawUserArtistData, rawArtistData, rawArtistAlias)
            //evaluate(jsc, rawUserArtistData, rawArtistAlias)
            recommend(jsc, rawUserArtistData, rawArtistData, rawArtistAlias)


        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }
}

fun main(args: Array<String>) {
    BasicConfigurator.configure()
    val lastFmTest = LastFmALS()
    lastFmTest.run()
}



