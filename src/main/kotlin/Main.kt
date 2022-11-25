import dataset.IrisDataset
import dataset.datasets
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import layer.Layer

fun main(): Unit = runBlocking {
    val (train, test) = datasets.shuffled().chunked(120)

    createModel(train, test, 12)
//    (10..30)
//        .map { async { createModel(train, test, it) to it } }
//        .maxBy { it.await().first }
//        .also { println(it.await().second) }
}

suspend fun createModel(
    train: List<IrisDataset>,
    test: List<IrisDataset>,
    seed: Int,
): Int = withContext(Dispatchers.Default){

    val model = (1..50).fold(
        Layer.create(input = 4, center = 50, output = 3, rate = 0.01, random = Random(seed)),
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
    return@withContext test.count { data ->
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