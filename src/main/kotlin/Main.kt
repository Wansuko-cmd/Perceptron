import dataset.IrisDataset
import dataset.datasets
import kotlin.random.Random
import layer.Layer
import layer.random

fun main() {
    val (train, test) = datasets.shuffled().chunked(120)
    random = Random(2)
    createModel(train, test)
//    (1..10).maxBy {
//        random = Random(it)
//        createModel(train, test)
//    }.also { println(it) }
}

fun createModel(train: List<IrisDataset>, test: List<IrisDataset>): Int {
    val model = (1..10).fold(
        Layer.create(input = 4, center = 50, output = 3, rate = 0.01),
    ) { model, index ->
        println("epoc: $index")
        train.fold(model) { acc, element ->
            acc.train(
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
    return test.count { data ->
        model.forward(
            input = listOf(
                data.petalLength,
                data.petalWidth,
                data.sepalLength,
                data.sepalWidth,
            ),
        ) == data.label
    }.also { println(it.toDouble() / test.size.toDouble()) }
}