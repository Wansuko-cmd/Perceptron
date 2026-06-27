import dataset.mnist.createMnistModel
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    createMnistModel(epoch = 10)
}
