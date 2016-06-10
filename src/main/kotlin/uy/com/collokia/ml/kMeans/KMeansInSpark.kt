package uy.com.collokia.ml.kMeans

import org.apache.log4j.BasicConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.DoubleFunction
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import scala.Tuple2
import uy.com.collokia.util.component1
import uy.com.collokia.util.component2
import java.io.Serializable


public class KMeansInSpark() : Serializable {

    public fun runKMeans() {
        val sparkConf = SparkConf().setAppName("KMeans").setMaster("local[6]")

        val jsc = JavaSparkContext(sparkConf)

        val rawData = jsc.textFile("./data/KMeans/kddcup.data_10_percent.gz").cache()
        //clusteringTake0(rawData)
        //clusteringTake1(rawData)
        //clusteringTake2(rawData)
        //clusteringTake3(rawData)
        //clusteringTake4(rawData)
        anomalies(rawData)
    }

    public fun readData(rawData: JavaRDD<String>): JavaPairRDD<String, Vector> {
        val labelsAndData = rawData.mapToPair { line ->
            val buffer = line.split(',').toMutableList()

            buffer.removeAt(1)
            buffer.removeAt(1)
            buffer.removeAt(1)
            val label = buffer.removeAt(buffer.size - 1)
            val vector = Vectors.dense(buffer.map({ value -> value.toDouble() }).toDoubleArray())
            Tuple2(label, vector)
        }
        return labelsAndData

    }

    public fun clusteringTake0(rawData: JavaRDD<String>) {

        println(rawData.map({ line -> line.split(',').last() }).countByValue().toList().sortedBy({ value -> value.second }).reversed().joinToString("\n"))

        val labelsAndData = readData(rawData)

        val data = labelsAndData.values().cache()

        val kmeans = KMeans()
        val model = kmeans.run(data.rdd())

        model.clusterCenters().forEach { clusterCenter ->
            println(clusterCenter)
        }

        val clusterLabelCount = labelsAndData.mapToPair { labelAndData ->
            val (label, datum) = labelAndData
            val cluster = model.predict(datum)
            Tuple2(cluster, label)
        }.countByValue()

        println("cluster\tlabel\tcount")
        clusterLabelCount.toList().forEach { clusterLabelFreq ->
            val (clusterLabel, count) = clusterLabelFreq
            val (cluster, label) = clusterLabel
            println("$cluster\t$label\t$count")
        }

        data.unpersist()

    }

    public fun distance(a: Vector, b: Vector): Double {
        return Math.sqrt(a.toArray().zip(b.toArray()).map({ p -> p.first - p.second }).map({ d -> d * d }).sum())
    }

    public fun distToCentroid(datum: Vector, model: KMeansModel): Double {
        val cluster = model.predict(datum)
        val centroid = model.clusterCenters()[cluster]
        return distance(centroid, datum)
    }

    public fun clusteringScore(data: JavaRDD<Vector>, k: Int): Double {
        val kmeans = KMeans()
        kmeans.setK(k)
        val model = kmeans.run(data.rdd())
        return data.mapToDouble<Double>(DoubleFunction { datum -> distToCentroid(datum, model) }).mean()
    }

    public fun clusteringScore2(data: JavaRDD<Vector>, k: Int): Double {
        val kmeans = KMeans()
        kmeans.setK(k)
        kmeans.setRuns(10)
        kmeans.setEpsilon(1.0e-6)
        val model = kmeans.run(data.rdd())
        return data.mapToDouble<Double>(DoubleFunction { datum -> distToCentroid(datum, model) }).mean()
    }

    public fun clusteringTake1(rawData: JavaRDD<String>) {

        val data = readData(rawData).values().cache()

        println((5..30 step 5).map({ k -> Tuple2(k, clusteringScore(data, k)) }).joinToString("\n"))


        println((30..100 step 10).map({ k -> Tuple2(k, clusteringScore2(data, k)) }).
                toList().joinToString("\n"))

        data.unpersist()
    }

    // Clustering, Take 2

    public fun buildNormalizationFunction(data: JavaRDD<Vector>): (Vector) -> Vector {
        val dataAsArray = data.map({ vector -> vector.toArray() })
        val numCols = dataAsArray.first().size
        val n = dataAsArray.count()
        val sums = dataAsArray.reduce({ a, b -> a.zip(b).map({ t -> t.first + t.second }).toDoubleArray() })

        val sumSquares = dataAsArray.fold(
                DoubleArray(numCols),
                { a, b -> a.zip(b).map({ t -> t.first + t.second * t.second }).toDoubleArray() }
        )

        val stdevs = sumSquares.zip(sums).map { sumSquare ->

            val (sumSq, sum) = sumSquare
            Math.sqrt(n * sumSq - sum * sum) / n
        }
        val means = sums.map({ it -> it / n })

        return { datum ->

            val normalizedArray = datum.toArray().zip(means).zip(stdevs).map { item ->
                val (valueMean, stdev) = item
                val (value, mean) = valueMean
                if (stdev <= 0) (value - mean) else (value - mean) / stdev
            }
            Vectors.dense(normalizedArray.toDoubleArray())
        }

    }

    public fun clusteringTake2(rawData: JavaRDD<String>) {
        val data = readData(rawData).values().cache()

        val normalizedData = data.map(buildNormalizationFunction(data)).cache()

        println((60..120 step 10).map({ k ->
            Tuple2(k, clusteringScore2(normalizedData, k))
        }).toList().joinToString("\n"))

        normalizedData.unpersist()
    }


    // Clustering, Take 3

    public fun buildCategoricalAndLabelFunction(rawData: JavaRDD<String>): (String) -> Tuple2<String, Vector> {
        val splitData = rawData.map({ line -> line.split(',') })

        val index = (0..splitData.count().toInt() - 1).map { i ->
            i
        }

        val protocols = splitData.map({ it -> it[1] }).distinct().collect().zip(index).toMap()
        val services = splitData.map({ it -> it[2] }).distinct().collect().zip(index).toMap()
        val tcpStates = splitData.map({ it -> it[3] }).distinct().collect().zip(index).toMap()
        return { line ->
            val buffer = line.split(',').toMutableList()
            val protocol = buffer.removeAt(1)
            val service = buffer.removeAt(1)
            val tcpState = buffer.removeAt(1)
            val label = buffer.removeAt(buffer.size - 1)
            val vector = buffer.map({ value -> value.toDouble() }).toMutableList()

            val newProtocolFeatures = DoubleArray(protocols.size)
            newProtocolFeatures[protocols[protocol]!!] = 1.0
            val newServiceFeatures = DoubleArray(services.size)
            newServiceFeatures[services[service]!!] = 1.0
            val newTcpStateFeatures = DoubleArray(tcpStates.size)
            newTcpStateFeatures[tcpStates[tcpState]!!] = 1.0


            vector.addAll(1, newTcpStateFeatures.toList())
            vector.addAll(1, newServiceFeatures.toList())
            vector.addAll(1, newProtocolFeatures.toList())

            Tuple2(label, Vectors.dense(vector.toDoubleArray()))
        }
    }

    public fun clusteringTake3(rawData: JavaRDD<String>) {
        val parseFunction = buildCategoricalAndLabelFunction(rawData)
        val data = rawData.mapToPair(parseFunction).values()
        val normalizedData = data.map(buildNormalizationFunction(data)).cache()

        println((80..160 step 10).map({ k ->
            Tuple2(k, clusteringScore2(normalizedData, k))
        }).toList().joinToString ("\n"))

        normalizedData.unpersist()
    }

    public fun entropy(counts: Iterable<Int>): Double {
        val values = counts.filter({ it -> it > 0 })
        val n = values.sum().toDouble()
        return values.map { v ->
            val p = v / n
            -p * Math.log(p)
        }.sum()
    }

    public fun clusteringScore3(normalizedLabelsAndData: JavaPairRDD<String, Vector>, k: Int): Double {
        val kmeans = KMeans()
        kmeans.setK(k)
        kmeans.setRuns(10)
        kmeans.setEpsilon(1.0e-6)

        val model = kmeans.run(normalizedLabelsAndData.values().rdd().cache())

        // Predict cluster for each datum
        val labelsAndClusters = normalizedLabelsAndData.mapToPair({ instance ->
            Tuple2(instance._1, model.predict(instance._2))
        })

        // Swap keys / values
        val clustersAndLabels = labelsAndClusters.mapToPair({ it -> it.swap() })

        // Extract collections of labels, per cluster
        val labelsInCluster = clustersAndLabels.groupByKey().values()

        // Count labels in collections
        val labelCounts = labelsInCluster.map { it -> it.groupBy({ label -> label }).map { label -> label.value.size } }

        // _.groupBy(l => l).map(_._2.size)
        // Average entropy weighted by cluster size
        val n = normalizedLabelsAndData.count()

        return labelCounts.map({ m -> (m.sum() * entropy(m)) }).collect().sum() / n
    }

    public fun clusteringTake4(rawData: JavaRDD<String>) {
        val parseFunction = buildCategoricalAndLabelFunction(rawData)
        val labelsAndData = rawData.mapToPair(parseFunction)
        val normalizedLabelsAndData =
                labelsAndData.mapValues(buildNormalizationFunction(labelsAndData.values())).cache()

        println((80..160 step 10).map({ k ->
            Tuple2(k, clusteringScore3(normalizedLabelsAndData, k))
        }).joinToString("\n"))

        normalizedLabelsAndData.unpersist()
    }

    // Detect anomalies

    public fun buildAnomalyDetector(data: JavaRDD<Vector>, normalizeFunction: (Vector) -> Vector ): (Vector) -> Boolean {
        val normalizedData = data.map(normalizeFunction)
        normalizedData.cache()

        val kmeans =  KMeans()
        kmeans.setK(150)
        kmeans.setRuns(10)
        kmeans.setEpsilon(1.0e-6)
        val model = kmeans.run(normalizedData.rdd())

        normalizedData.unpersist()

        val distances = normalizedData.map({datum -> distToCentroid(datum, model)})
        val threshold = distances.top(100).last()

        return {
            datum -> distToCentroid(normalizeFunction(datum), model) > threshold
        }
    }

    public fun anomalies(rawData: JavaRDD<String>) {
        val parseFunction = buildCategoricalAndLabelFunction(rawData)
        val originalAndData = rawData.mapToPair({line -> Tuple2(line, parseFunction(line)._2)})
        val data = originalAndData.values()
        val normalizeFunction = buildNormalizationFunction(data)
        val anomalyDetector = buildAnomalyDetector(data, normalizeFunction)
        val anomalies = originalAndData.filter { data ->
            val (original, datum) = data
            anomalyDetector(datum)
        }.keys()
        println(anomalies.take(10).joinToString("\n"))




    }

    public fun zipTest() {
        val list1 = listOf(1, 2, 3, 4, 5, 6, 7, 8)
        val list2 = listOf("egy", "ketto", "harom", "negy")
        println(list1.zip(list2))
    }


}

fun main(args: Array<String>) {
    BasicConfigurator.configure()
    val kmeans = KMeansInSpark()
    //kmeans.zipTest()
    kmeans.runKMeans()

    //val list = mutableListOf(1,2,3,4,5)
    //println(list.removeAt(1))
    //println(list)
}

