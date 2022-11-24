import dataset.datasets
import kotlin.random.Random
import layer.Layer
import layer.random

fun main() {
//    val (train, test) = datasets.shuffled().chunked(120)
    val (train, test) = datasets to datasets
    (1..100).maxBy {
        random = Random(it)
        val model = (1..50).fold(
            Layer.create(input = 4, center = 50, output = 3, rate = 0.01),
        ) { model, index ->
            println("epoc: $index")
            train.fold(model) { acc, element ->
                acc.fd(
                    input = listOf(
                        element.petalLength,
                        element.petalWidth,
                        element.sepalLength,
                        element.sepalWidth,
                    ),
                    label = element.label,
                )
            }
        }
        test.count { data ->
            model.forward(
                value = listOf(
                    data.petalLength,
                    data.petalWidth,
                    data.sepalLength,
                    data.sepalWidth,
                ),
            ) == data.label
        }.also { println(it.toDouble() / test.size.toDouble()) }
    }
}
