package dataset.mnist

import com.wsr.knist.network.Network
import com.wsr.knist.network.NetworkSerializer
import com.wsr.knist.network.create
import com.wsr.knist.network.initializer.He
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.adam.AdamW
import com.wsr.knist.network.output.softmax.softmaxWithLoss
import com.wsr.knist.network.port
import com.wsr.knist.network.process.compute.affine.affine
import com.wsr.knist.network.process.compute.bias.d1.bias
import com.wsr.knist.network.process.compute.function.relu.reLU
import com.wsr.knist.network.process.compute.norm.layer.d1.layerNorm
import com.wsr.knist.network.process.reshape.reshape.reshapeToD1
import kotlinx.coroutines.runBlocking

private const val TRAIN_IMAGE_PATH = "mnist/train-images-idx3-ubyte.gz"
private const val TRAIN_LABEL_PATH = "mnist/train-labels-idx1-ubyte.gz"

private const val TEST_IMAGE_PATH = "mnist/t10k-images-idx3-ubyte.gz"
private const val TEST_LABEL_PATH = "mnist/t10k-labels-idx1-ubyte.gz"

fun createMnistModel(epoch: Int, seed: Int? = null): Network.Src1.Sink1<List<List<Float>>, List<Int>> = runBlocking {
    // カスタムした層をSerializerに登録
    NetworkSerializer.apply {
        register(PixelConverter::class)
        register(LabelConverter::class)
    }

    // ニューラルネットワークを構築
    val network = Network.create(
        port = port(PixelConverter(outputI = 28, outputJ = 28)),
        optimizer = AdamW(scheduler = Scheduler.Fix(0.001f)),
        initializer = He(seed = seed),
    ) { input ->
        input.reshapeToD1()
            .layerNorm()
            .affine(neuron = 256).bias().reLU()
            .affine(neuron = 128).bias().reLU()
            .affine(neuron = 10)
            .softmaxWithLoss(converter = { LabelConverter(inputI) })
    }

    println("データ読み込み")
    val train = MnistDataset.read(imagePath = TRAIN_IMAGE_PATH, labelPath = TRAIN_LABEL_PATH)
    val test = MnistDataset.read(imagePath = TEST_IMAGE_PATH, labelPath = TEST_LABEL_PATH)

    println("訓練開始")
    repeat(epoch) { epoch ->
        println("epoch: $epoch")
        train.chunked(256).forEach { data ->
            network.train(
                input = data.map { it.pixels },
                label = data.map { it.label },
            )
        }
    }

    println("評価開始")
    val accuracy = test
        .let { data ->
            data.map { it.pixels }.chunked(5064)
                .flatMap { network.expect(it) }
                .zip(data.map { it.label })
                .count { (e, a) -> e == a }
        }
        .let { it.toFloat() / test.size.toFloat() }
    println("${accuracy * 100}%")

    network
}
