@file:Suppress("DuplicatedCode")

import common.conv
import common.relu
import common.sigmoid
import dataset.iris.datasets
import dataset.mnist.MnistDataset
import kotlinx.coroutines.runBlocking
import network.DevNetwork
import network.InputConfig
import network.LayerConfig
import network.LayerType
import org.jetbrains.bio.viktor.F64Array
import org.jetbrains.bio.viktor._I
import kotlin.random.Random
import kotlin.system.measureTimeMillis

fun main(): Unit = runBlocking {
//    val network = DevNetwork.create(
//        InputConfig(4),
//        listOf(
//            LayerConfig(50, ::relu, LayerType.MatMul),
//            LayerConfig(3, ::sigmoid, LayerType.MatMul),
//        ),
//        Random(1652),
//        0.01
//    )
//    val (train, test) = datasets.shuffled().chunked(120)
//    (1..1000).forEach { epoc ->
//        println("epoc: $epoc")
//        train.forEach { data ->
//            network.train(
//                input = F64Array.of(data.petalLength, data.petalWidth, data.sepalLength, data.sepalWidth),
//                label = data.label
//            )
//        }
//    }
//    test.count { data ->
//        network.expect(
//            input = F64Array.of(data.petalLength, data.petalWidth, data.sepalLength, data.sepalWidth),
//        ) == data.label
//    }.let { println(it.toDouble() / test.size) }
    val (train, test) = MnistDataset.read().chunked(2000)
    val network = DevNetwork.create(
        InputConfig(size = 1),
        listOf(
            LayerConfig(size = 32, activationFunction = ::relu, type = LayerType.Conv),
            LayerConfig(size = 64, activationFunction = ::relu, type = LayerType.Conv),
            LayerConfig(size = 32, activationFunction = ::relu, type = LayerType.MatMul),
            LayerConfig(size = 10, activationFunction = ::sigmoid, type = LayerType.MatMul),
        ),
        random = Random(1652),
        rate = 0.01,
    )
    (1..5).forEach { epoc ->
        println("epoc: $epoc")
        measureTimeMillis {
            train.shuffled().take(100).forEach { data ->
                network.train(
                    input = data.pix,
                    label = data.label,
                )
            }
        }.let { println(it) }
    }
    test.count { data ->
        network.expect(
            input = data.pix,
        ) == data.label
    }.let { println(it.toDouble() / test.size) }
}
