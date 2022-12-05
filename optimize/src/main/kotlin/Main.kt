@file:Suppress("DuplicatedCode")

import common.createModel
import common.relu
import common.sigmoid
import dataset.iris.datasets
import dataset.janken.JankenDataset
import dataset.wine.WineDataset
import network.DevNetwork
import network.InputConfig
import network.LayerConfig
import network.LayerType
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import network.JankenNearest

fun main(): Unit = runBlocking {
    val jankenDatasets = JankenDataset.load()
    (0..100)
        .map {
            val (train, test) = jankenDatasets.shuffled().chunked(80)
            val nearest = JankenNearest(train)
            test.count { data -> nearest.expect(input = data) == data.label } to test.size
        }
        .fold(0.0 to 0.0) { acc, (score, size) -> acc.first + score to acc.second + size}
        .also { println(it.first / it.second) }
//    println(jankenDatasets.first().data.size)
//    (1..100).map {
//        val (train, test) = jankenDatasets.shuffled().chunked(80)
//        async { createJankenModel(train = train, test = test, epoc = 3000, seed = 42) to test.size }
//    }
//        .awaitAll()
//        .fold(0 to 0) { acc, (correct, size) -> acc.first + correct to acc.second + size }
//        .let { it.first.toDouble() / it.second.toDouble() }
//        .also { println(it) }
//    (0..100)
//        .map { async { createJankenModel(train, test, 100, it) to it } }
//        .awaitAll()
//        .asSequence()
//        .sortedByDescending { (score, _) -> score }
//        .take(10)
//        .onEach { (score, seed) -> println("seed: $seed, score, $score") }
//        .toList()
//        .map { (_, seed) ->
//            (1..100).map {
//                val (train2, test2) = jankenDatasets.shuffled().chunked(80)
//                async { createJankenModel(train = train2, test = test2, epoc = 1000, seed = seed) to test.size }
//            }
//                .awaitAll()
//                .fold(0 to 0) { acc, (correct, size) -> acc.first + correct to acc.second + size }
//                .let { it.first.toDouble() / it.second.toDouble() to seed}
//        }
//        .sortedByDescending { (score, _) -> score }
//        .onEach { (score, seed) -> println("seed: $seed, score, $score") }
}

suspend fun createJankenModel(
    train: List<JankenDataset>,
    test: List<JankenDataset>,
    epoc: Int,
    seed: Int? = null,
) = withContext(Dispatchers.Default) {
    val network = DevNetwork.create(
        InputConfig(train.first().data.size),
        listOf(
            LayerConfig(50, ::relu, LayerType.MatMul),
            LayerConfig(3, ::sigmoid, LayerType.MatMul),
        ),
        seed?.let { Random(seed) } ?: Random,
        0.01,
    )
    (1..epoc).forEach { _ ->
//        println("epoc: $epoc")
        train.forEach { data ->
            network.train(
                input = data.data,
                label = data.label,
            )
        }
    }
    test.count { data ->
        network.expect(input = data.data) == data.label
    }
}

fun createWineModel(
    train: List<WineDataset>,
    test: List<WineDataset>,
    epoc: Int,
    seed: Int? = null,
): Int {
    val network = DevNetwork.create(
        InputConfig(13),
        listOf(
            LayerConfig(100, ::relu, LayerType.MatMul),
            LayerConfig(30, ::relu, LayerType.MatMul),
            LayerConfig(3, ::sigmoid, LayerType.MatMul),
        ),
        seed?.let { Random(it) } ?: Random,
        0.01,
    )
    (1..epoc).forEach { epoc ->
        train.forEach { data ->
            network.train(
                input = listOf(
                    data.alcohol,
                    data.malicAcid,
                    data.ash,
                    data.alcalinityOfAsh,
                    data.magnesium,
                    data.totalPhenols,
                    data.flavanoids,
                    data.nonflavAnoidPhenols,
                    data.proanthocyanins,
                    data.colorIntensity,
                    data.hue,
                    data.wines,
                    data.proline,
                ),
                label = data.label,
            )
        }
    }
    return test.count { data ->
        network.expect(
            input = listOf(
                data.alcohol,
                data.malicAcid,
                data.ash,
                data.alcalinityOfAsh,
                data.magnesium,
                data.totalPhenols,
                data.flavanoids,
                data.nonflavAnoidPhenols,
                data.proanthocyanins,
                data.colorIntensity,
                data.hue,
                data.wines,
                data.proline,
            ),
        ) == data.label
    }.also { println(it.toDouble() / test.size) }
}
