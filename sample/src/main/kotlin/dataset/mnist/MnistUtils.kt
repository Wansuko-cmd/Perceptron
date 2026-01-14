package dataset.mnist

import com.wsr.network.Network
import com.wsr.network.NetworkBuilder
import com.wsr.network.NetworkSerializer
import com.wsr.network.initializer.He
import com.wsr.network.optimizer.Scheduler
import com.wsr.network.optimizer.adam.AdamW
import com.wsr.network.output.softmax.softmaxWithLoss
import com.wsr.network.process.compute.affine.affine
import com.wsr.network.process.compute.bias.d1.bias
import com.wsr.network.process.compute.function.relu.reLU
import com.wsr.network.process.compute.norm.layer.d1.layerNorm
import com.wsr.network.process.reshape.reshape.reshapeToD1

private const val TRAIN_IMAGE_PATH = "mnist/train-images-idx3-ubyte.gz"
private const val TRAIN_LABEL_PATH = "mnist/train-labels-idx1-ubyte.gz"

private const val TEST_IMAGE_PATH = "mnist/t10k-images-idx3-ubyte.gz"
private const val TEST_LABEL_PATH = "mnist/t10k-labels-idx1-ubyte.gz"

fun createMnistModel(epoc: Int, seed: Int? = null): Network<List<Float>, Int> {
    // カスタムした層をSerializerに登録
    NetworkSerializer.apply {
        register(PixelConverter::class)
        register(LabelConverter::class)
    }

    // ニューラルネットワークを構築
    val network = NetworkBuilder
        .inputPx(
            x = 28,
            y = 28,
            optimizer = AdamW(scheduler = Scheduler.Fix(0.001f)),
            initializer = He(seed = seed),
        )
        .reshapeToD1()
        .layerNorm()
        .affine(neuron = 256).bias().reLU()
        .affine(neuron = 128).bias().reLU()
        .affine(neuron = 10)
        .softmaxWithLoss(converter = { LabelConverter(inputSize) })

    println("データ読み込み")
    val train = MnistDataset.read(imagePath = TRAIN_IMAGE_PATH, labelPath = TRAIN_LABEL_PATH)
    val test = MnistDataset.read(imagePath = TEST_IMAGE_PATH, labelPath = TEST_LABEL_PATH)

    println("訓練開始")
    repeat(epoc) { epoc ->
        println("epoc: $epoc")
        train.chunked(256).forEach { data ->
            network.train(
                input = data.map { it.pixels },
                label = data.map { it.label },
            )
        }
    }

    println("評価開始")
    val accuracy = test
        .count { data -> network.expect(input = data.pixels) == data.label }
        .let { it.toFloat() / test.size.toFloat() }
    println("${accuracy * 100}%")

    return network
}
