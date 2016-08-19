package uy.com.collokia.ml.collaborativeFiltering.movie

import org.apache.log4j.BasicConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.DoubleFunction
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2
import uy.com.collokia.common.utils.component1
import uy.com.collokia.common.utils.component2
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.combineByKeyIntoList
import uy.com.collokia.scala.scalaClassTag
import uy.com.collokia.util.*
import java.io.Serializable

data class Movie(val movieId: Int, val title: String, var genres: List<String>?, var prediction: Double) : Serializable

public class MovieRank() : Serializable {
    public fun simpleExampleRun(data : JavaRDD<String>) {

        // Load and parse the testData
        val ratings = data.map({ line ->
            val (user, product, rating) = line.split(',')
            Rating(user.toInt(), product.toInt(), rating.toDouble())
        })

        val rank = 10
        val numIterations = 20;

        // Build the recommendation model using ALS
        val model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);


        // Evaluate the model on rating testData
        val userProducts = ratings.mapToPair({ rating ->
            Tuple2(rating.user(), rating.product())
        })


        val predictions = model.predict(userProducts).mapToPair({ rating ->
            Tuple2(Tuple2(rating.user(), rating.product()), rating.rating())
        })

        val ratesAndPreds = ratings.mapToPair({ rating ->
            Tuple2(Tuple2(rating.user(), rating.product()), rating.rating())
        }).join(predictions).values()

        // TODO: revisit what is weird about this not working with a lambda
        val MSE = ratesAndPreds.mapToDouble<Double>(DoubleFunction { resultsPair ->
            val err = resultsPair._1() - resultsPair._2();
            err * err
        }).mean()

        System.out.println("Mean Squared Error = " + MSE);

    }

    public fun evaluateALS(movieRatingDataInLine : JavaRDD<String>) {



        val movieRatingData = movieRatingDataInLine.map({ line ->
            val (user, product, rating) = line.split(Regex("::"))
            Rating(user.toInt(), product.toInt(), rating.toDouble() - 2.5)
        })

        val binarizedRatings = movieRatingData.map({ rating ->
            Rating(rating.user(), rating.product(),
                    if (rating.rating() > 0) 1.0 else 0.0)
        }).cache()

        // Summarize ratings
        val numRatings = movieRatingData.count()
        val numUsers = movieRatingData.map({ rating -> rating.user() }).distinct().count()
        val numMovies = movieRatingData.map({ rating -> rating.product() }).distinct().count()
        println("Got $numRatings ratings from $numUsers users on $numMovies movies.")

        // Build the model
        val numIterations = 10
        val rank = 10
        val lambda = 0.01
        val model = ALS.train(movieRatingData.rdd(), rank, numIterations, lambda)


// Get sorted top ten predictions for each user and then scale from [0, 1]
        val userRecommended = model.recommendProductsForUsers(10).toJavaRDD().mapToPair { recommendation ->

            Tuple2(recommendation._1 as Int, recommendation._2.map({ rating ->
                scaledRating(rating)
            }))
        }

        println("recommend product to user:\t ${model.recommendUsers(2, 5).toList()}")

        // Assume that any movie a user rated 3 or higher (which maps to a 1) is a relevant document
// Compare with top ten most relevant documents
        val userMovies = binarizedRatings.groupBy({ rating -> rating.user() })
        val relevantDocuments = userMovies.join(userRecommended).map { predictionsToUsers ->
            val predictions = predictionsToUsers._2._2.map { prediction -> prediction.product() }.toTypedArray()
            val actual = predictionsToUsers._2._1.filter { actual ->
                (actual.rating() > 0.0)
            }.map { item ->
                item.product()
            }.toTypedArray()
            Tuple2(predictions as Any, actual as Any)
        }
        val metrics = RankingMetrics(relevantDocuments.rdd(), scalaClassTag<Int>())

        // Precision at K
        arrayOf(1, 3, 5).forEach { k ->
            println("Precision at $k = ${metrics.precisionAt(k)}")
        }

// Mean average precision
        println("Mean average precision = ${metrics.meanAveragePrecision()}")

// Normalized discounted cumulative gain
        arrayOf(1, 3, 5).forEach {
            k ->
            println("NDCG at $k = ${metrics.ndcgAt(k)}")
        }

        // Get predictions for each testData point
        val allPredictions = model.predict(movieRatingData.mapToPair({ rating -> Tuple2(rating.user(), rating.product()) })).mapToPair({ rating ->
            Tuple2(Tuple2(rating.user(),
                    rating.product()), rating.rating())
        })

        val allRatings = movieRatingData.mapToPair({ rating -> Tuple2(Tuple2(rating.user(), rating.product()), rating.rating()) })
        val predictionsAndLabels = allPredictions.join(allRatings).map { item ->
            Tuple2(item._2._1 as Any, item._2._2 as Any)
        }

        // Get the RMSE using regression metrics
        val regressionMetrics = RegressionMetrics(predictionsAndLabels.rdd())
        println("RMSE = ${regressionMetrics.rootMeanSquaredError()}")

// R-squared
        println("R-squared = ${regressionMetrics.r2()}")

    }

    private fun scaledRating(rating: Rating): Rating {
        val scaledRating = Math.max(Math.min(rating.rating(), 1.0), 0.0)
        return Rating(rating.user(), rating.product(), scaledRating)
    }

    public fun movieLensALS(movieDataInLine : JavaRDD<String>,movieRatingDataInLine : JavaRDD<String>) {
        val sparkConf = SparkConf().setAppName("Collaborative Filtering Example").setMaster("local[6]")
        val ctx = JavaSparkContext(sparkConf)

        val recomendGenre = "Anime"

        val movieData = movieDataInLine.mapToPair({ line ->
            val sArray = line.split(' ')
            val movie = Movie(sArray[0].split(Regex("::"))[0].toInt(), sArray[0].split(Regex("::"))[1], sArray[1].split(Regex("::"))[1].split('|').toList(), 0.0)
            Tuple2(movie.movieId, movie)
        })

        val movieRatingData = movieRatingDataInLine.map({ line ->
            val (user, product, rating) = line.split(Regex("::"))
            Rating(user.toInt(), product.toInt(), rating.toDouble())
        })

        // Build the recommendation model using ALS
        val rank = 10
        val numIterations = 10
        val lambda = 0.01

        val (training, test) = movieRatingData.randomSplit(doubleArrayOf(0.7, 0.3))
        training.cache()
        test.cache()

        val numTraining = training.count()
        val numTest = test.count()
        println("Training: $numTraining, test: $numTest.")

        //val model = ALS.train(training.rdd(), rank, numIterations, lambda);
        val model = ALS.trainImplicit(movieRatingData.rdd(), rank, numIterations, lambda, -1, 1.0)

        val recommendProductsForUsers = model.recommendProductsForUsers(10).toJavaRDD()
        println(model.recommendProducts(1, 10).toList())

        println(recommendProductsForUsers.mapToPair({ recommendation -> Tuple2(recommendation._1 as Int, recommendation._2().toList()) }).collect().joinToString("\n"))

        val testData = test.mapToPair({ rating ->
            Tuple2(rating.user(), rating.product())
        })

        val predictions = model.predict(testData).mapToPair({ rating ->
            Tuple2(Tuple2(rating.user(), rating.product()), rating.rating())
        })

        val predicatedData = test.mapToPair({ rating ->
            Tuple2(Tuple2(rating.user(), rating.product()), rating.rating())
        }).join(predictions).values()


        // TODO: revisit what is weird about this not working with a lambda
        val MSE = predicatedData.mapToDouble<Double>(DoubleFunction { resultsPair ->
            val err = resultsPair._1() - resultsPair._2();
            err * err
        }).mean()

        System.out.println("Mean Squared Error = " + MSE);

        val genreTestData = test.mapToPair({ rating ->
            Tuple2(rating.product(), rating.user())
        }).join(movieData).filter({ item ->
            (item._2._2.genres!!.contains(recomendGenre))
        }).mapToPair({ item ->
            Tuple2(item._2._1, item._1)
        })

        println(model.recommendProducts(1, 10).joinToString("\n"))

        val predictionsGenre = model.predict(genreTestData).mapToPair({ rating ->
            Tuple2(rating.product(), Tuple2(rating.user(), rating.rating()))
        }).join(movieData).mapToPair({ item ->
            val (movieId, movieData) = item
            val (rank, movie) = movieData
            Tuple2(rank._1, Movie(movieId, movie.title, movie.genres, rank._2))
        }).combineByKeyIntoList().mapToPair({ item ->
            val list = item._2.sortedByDescending { prediction -> prediction.prediction }
            Tuple2(item._1, list)
        })
        predictionsGenre.collect().joinToString("\n")

        ctx.stop()
    }


    public fun runMovieRank() {
        val time = measureTimeInMillis {
            BasicConfigurator.configure()
            val sparkConf = SparkConf().setAppName("Collaborative Filtering Example").setMaster("local[6]")
            val jsc = JavaSparkContext(sparkConf)

            val path = "./testData/collaborativeFiltering/test.txt";
            val testData = jsc.textFile(path);
            val movieRatingPath = "./testData/collaborativeFiltering/sample_movielens_ratings.txt";
            val movieRatingDataInLine = jsc.textFile(movieRatingPath);
            val moviePath = "./testData/collaborativeFiltering/sample_movielens_movies.txt";
            val movieDataInLine = jsc.textFile(moviePath);

            //movieLensALS()
            evaluateALS(movieRatingDataInLine)
        }
        println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
    }

}

fun main(args: Array<String>) {
    val movieRank = MovieRank()
    movieRank.runMovieRank()

}

