@file:Suppress("NonAsciiCharacters", "RemoveRedundantBackticks")

package mnist

import com.wsr.knist.network.Network
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.NetworkSerializer
import com.wsr.knist.network.initializer.He
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.adam.AdamW
import com.wsr.knist.network.output.softmax.softmaxWithLoss
import com.wsr.knist.network.process.compute.affine.affine
import com.wsr.knist.network.process.compute.bias.d1.bias
import com.wsr.knist.network.process.compute.function.relu.reLU
import com.wsr.knist.network.process.compute.function.relu.swish
import com.wsr.knist.network.process.compute.norm.layer.d1.layerNorm
import com.wsr.knist.network.process.compute.skip.skip
import com.wsr.knist.network.process.reshape.reshape.reshapeToD1
import dataset.mnist.LabelConverter
import dataset.mnist.MnistDataset
import dataset.mnist.PixelConverter
import dataset.mnist.inputPx
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

private const val TRAIN_IMAGE_PATH = "mnist/train-images-idx3-ubyte.gz"
private const val TRAIN_LABEL_PATH = "mnist/train-labels-idx1-ubyte.gz"

private const val TEST_IMAGE_PATH = "mnist/t10k-images-idx3-ubyte.gz"
private const val TEST_LABEL_PATH = "mnist/t10k-labels-idx1-ubyte.gz"

class MnistTest {
    @Test
    fun `Mnistモデルの精度が落ちていないか確認`() = runTest {
        NetworkSerializer.apply {
            register(PixelConverter::class)
            register(LabelConverter::class)
        }

        println("ネットワーク構築")
        val network = createNetwork()

        println("Json変換")
        network.toJson()
            .also { println(it.take(100) + "...") }
            .also { println(Network.fromJson<List<List<Float>>, List<Int>>(it)) }

        println("訓練開始")
        val train = MnistDataset.read(imagePath = TRAIN_IMAGE_PATH, labelPath = TRAIN_LABEL_PATH)
        train.chunked(240).mapIndexed { i, data ->
            if (i % 10 == 0) println("train: $i")
            network.train(
                input = data.map { it.pixels },
                label = data.map { it.label },
            )
        }

        println("評価開始")
        val test = MnistDataset
            .read(imagePath = TEST_IMAGE_PATH, labelPath = TEST_LABEL_PATH)
            .take(100)

        val accuracy = test
            .count { data -> network.expect(input = listOf(data.pixels))[0] == data.label }
            .let { it.toFloat() / test.size.toFloat() }

        println("${accuracy * 100}%")

        assertTrue(actual = accuracy > 0.9f, message = "精度が90%を割っています")
    }

    private fun createNetwork(): Network<List<List<Float>>, List<Int>> = NetworkBuilder
        .inputPx(
            x = 28,
            y = 28,
            optimizer = AdamW(scheduler = Scheduler.Fix(0.001f)),
            initializer = He(seed = 0),
        )
        .reshapeToD1()
        .affine(neuron = 512).bias().reLU()
        .repeat(5) {
            skip {
                this
                    .layerNorm().affine(neuron = 512).bias().swish()
                    .layerNorm().affine(neuron = 512).bias().swish()
            }
        }
        .layerNorm().affine(neuron = 512).bias().swish()
        .layerNorm().affine(neuron = 128).bias().swish()
        .affine(neuron = 10)
        .softmaxWithLoss(converter = { LabelConverter(inputI) })
}
