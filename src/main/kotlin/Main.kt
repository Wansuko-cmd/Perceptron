import dataset.datasets
import layer.Layer
import layer.step

fun main() {
    val model = (1..1000).fold(
        Layer.create(input = 4, center = 50, output = 3, rate = 0.01),
    ) { model, index ->
        println("epoc: $index")
        datasets.fold(model) { acc, element ->
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
    datasets.count { data ->
        model.forward(
            value = listOf(
                data.petalLength,
                data.petalWidth,
                data.sepalLength,
                data.sepalWidth,
            ),
        ).map { it.sum().step().toInt() }[data.label] == 1
    }.let { println(it.toDouble() / datasets.size.toDouble()) }
}
