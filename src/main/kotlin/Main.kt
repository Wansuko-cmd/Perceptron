import dataset.datasets
import layer.Layer

fun main() {
    val (train, test) = datasets.shuffled().chunked(120)
    val model = (1..1000).fold(
        Layer.create(input = 4, center = 50, output = 3, rate = 0.01),
    ) { model, index ->
        println("epoc: $index")
        train.fold(model) { acc, element ->
            acc.train(
                value = listOf(
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
    }.let { println(it.toDouble() / test.size.toDouble()) }
}
