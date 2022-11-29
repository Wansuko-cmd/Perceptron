@file:Suppress("DuplicatedCode")

package network

import common.add
import common.conv
import common.maxIndex
import org.jetbrains.bio.viktor.F64Array
import org.jetbrains.bio.viktor._I
import kotlin.random.Random

/**
 * layers: 各層の情報を保持する
 * Int -> 層のニューロン数
 *
 * weight: 重みを保持する
 * List[層][前のニューロン][後ろのニューロン]
 * 0からカウントを開始する
 *
 * rate: 学習率
 */
class DevNetwork private constructor(
    private val layers: List<LayerConfig>,
    private val weights: List<F64Array>,
    private val rate: Double,
) {
    private val windowedLayers = layers.windowed(2) { (before, after) -> before to after }

    /**
     * 推定を行う関数
     * inputからラベル値を返す
     */
    fun expect(input: F64Array): Int {
        val output = forward(input)
        return (0 until layers.last().size).map { output[layers.size - 1][it] }.maxIndex()
    }

    /**
     * 学習を行う関数
     * inputの値とラベルを用いてSGD学習を行う
     */
    fun train(input: F64Array, label: Int) {
        val output = forward(input)
        backward(output, calcDelta(output, label))
    }

    /**
     * 順伝搬を行う関数
     */
    private fun forward(input: F64Array): List<F64Array> {
        val output = mutableListOf<F64Array>()

        // 入力を第1層の出力とする
        output.add(input)

        windowedLayers
            .mapIndexed { layer, (before, after) ->
                // 後ろのレイヤーに合わせて処理を変える
                when (after.type) {
                    // Input層は最初以外ないはずだからエラーとなる
                    is LayerType.Input -> throw Exception()
                    // 全結合層
                    is LayerType.MatMul ->
                        matMul(
                            input = when (before.type) {
                                // 入力層の次に全結合なのでList<Double>が来るはず
                                is LayerType.Input -> output[layer]
                                is LayerType.MatMul -> output[layer]
                                // TODO
                                is LayerType.Conv -> output[layer].reshape(output[layer].shape.fold(1) { acc, i -> acc * i })
                            },
                            layer = layer,
                            before = before,
                            after = after,
                        )
                    is LayerType.Conv ->
                        conv(
                            // TODO: 全結合の次にConvがくる場合を想定していない
                            input = output[layer],
                            layer = layer,
                            before = before,
                            after = after,
                        )
                }.let { output.add(it) }
            }
        return output
    }

    // 全結合の計算結果を出す
    private fun matMul(
        input: F64Array,
        layer: Int,
        before: LayerConfig,
        after: LayerConfig,
    ): F64Array {
        val out = F64Array(after.size)
        for (b in 0 until before.size) {
            out += weights[layer].view(b, 0) * input[b]
        }
        return out
    }

    /**
     * 畳み込みの結果を出す
     *
     * F64Array[横, 縦, チャンネル]
     */
    private fun conv(
        input: F64Array,
        layer: Int,
        before: LayerConfig,
        after: LayerConfig,
    ): F64Array =
        (0 until before.size).flatMap { b ->
            (0 until after.size).map { a ->
                input.conv(weights[layer].view(b, 0).view(a, 0))
            }
        }.reduce { a, b -> a + b }

    private fun pool(
        input: List<List<List<Double>>>,
    ): List<List<List<Double>>> =
        input.map { i ->
            (0 until i.size - 1).map { j ->
                (0 until i.size - 1).map { k ->
                    maxOf(i[j][k], i[j][k + 1], i[j + 1][k], i[j + 1][k + 1])
                }
            }
        }

    /**
     * 誤差逆伝搬のためのdeltaを取得する関数
     */
    private fun calcDelta(output: List<F64Array>, label: Int): List<F64Array> {
        val delta = mutableListOf<F64Array>()

        // 最終層のDeltaを計算
        val y = output[layers.size - 1]
        val t = F64Array(layers.last().size) { if (it == label) 0.9 else 0.1 }
        delta.add(y - t + (-y + 1.0) * y)

//        windowedLayers
//            .foreachDownIndexed { index, (before, after) ->
//                output[index] * delta[0] * weights[index]
//                (0 until before.size).map { b ->
//                    when (before.type) {
//                        // Input層は計算する必要がないため除外
//                        is LayerType.Input -> 0.0
//                        is LayerType.MatMul -> step(output[index][b]) *
//                            (0 until after.size).sumOf { a -> delta[0][a] * weights[index][b, a] }
//                        is LayerType.Conv -> {
//                            when (after.type) {
//                                is LayerType.Input, LayerType.MatMul ->
//                                    (output[index][b] as List<List<Double>>)
//                                        .sumOf {
//                                            it.sumOf { x ->
//                                                step(x) * (0 until after.size).sumOf { a -> delta[0][a] * weights[index][b][a] as Double }
//                                            }
//                                        }
//                                is LayerType.Conv ->
//                                    (output[index][b] as List<List<Double>>)
//                                        .sumOf { row ->
//                                            row.sumOf { x ->
//                                                step(x) * (0 until after.size).sumOf { a ->
//                                                    (weights[index][b][a] as List<List<Double>>)
//                                                        .sumOf { weightRow -> weightRow.sumOf { delta[0][a] * it } }
//                                                }
//                                            }
//                                        }
//                            }
//                        }
//                    }
//                }.let { delta.add(0, it) }
//            }
        return delta
    }

    /**
     * 誤差逆伝搬を行う関数
     */
    private fun backward(
        output: List<F64Array>,
        delta: List<F64Array>,
    ) {
//        windowedLayers
//            .mapIndexed { index, (before, after) ->
//                (0 until before.size).forEach { b ->
//                    (0 until after.size).forEach { a ->
//                        weights[index][b][a] = when (after.type) {
//                            // Input層には更新する値はないはずだからエラーとなる
//                            is LayerType.Input -> throw Exception()
//                            is LayerType.MatMul ->
//                                weights[index][b][a] as Double - rate * delta[index + 1][a] *
//                                    when (before.type) {
//                                        is LayerType.Input, LayerType.MatMul ->
//                                            output[index][b] as Double
//                                        is LayerType.Conv ->
//                                            (output[index] as List<List<List<Double>>>).map { it.flatten() }
//                                                .flatten()[b]
//                                    }
//                            is LayerType.Conv ->
//                                (weights[index][b][a] as List<List<Double>>).map { weightMatrix ->
//                                    weightMatrix.map { weight ->
//                                        weight - (output[index][b] as List<List<Double>>)
//                                            .sumOf { row ->
//                                                row.sumOf { column ->
//                                                    column * rate * delta[index + 1][a]
//                                                }
//                                            }
//                                    }
//                                }
//                        }
//                    }
//                }
//            }
    }

    companion object {
        fun create(
            input: InputConfig,
            layerConfigs: List<LayerConfig>,
            random: Random,
            rate: Double,
        ): DevNetwork {
            val weights = mutableListOf<F64Array>()
            val layers = listOf(input.toLayoutConfig()) + layerConfigs
            layers
                .windowed(2) { (before, after) -> before to after }
                .forEach { (before, after) ->
                    when (after.type) {
                        is LayerType.Input -> throw Exception()
                        is LayerType.MatMul -> F64Array(before.size, after.size) { _, _ -> random.nextDouble(from = -1.0, until = 1.0) }
                        is LayerType.Conv ->
                            F64Array(before.size, after.size, 4, 4, 1).also { it.V[_I] = random.nextDouble(from = -1.0, until = 1.0) }
                    }.let { weights.add(it) }
                }
            return DevNetwork(layers, weights, rate)
        }
    }
}

data class LayerConfig(
    val size: Int,
    val activationFunction: (Double) -> Double,
    val type: LayerType,
)

data class InputConfig(val size: Int) {
    fun toLayoutConfig() = LayerConfig(size, { it }, LayerType.Input)
}

sealed class LayerType {
    internal object Input : LayerType()
    object MatMul : LayerType()
    object Conv : LayerType()
}
