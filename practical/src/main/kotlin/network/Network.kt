package network

import common.maxIndex
import kotlin.random.Random
import layers.IOType
import layers.layer0d.Input0dConfig
import layers.layer0d.Layer0dConfig
import layers.layer0d.Output0dConfig

class Network(
    private val weights: Array<Array<Array<IOType>>>,
    private val output: Array<Array<IOType>>,
    private val delta: Array<Array<Double>>,
    private val forward: () -> Unit,
    private val calcDelta: (label: Int) -> Unit,
    private val backward: () -> Unit,
) {
    fun expect(input: List<Double>): Int {
        output[0] = input.map { IOType.IOType0d(it) }.toTypedArray()
        forward()
        return (0 until output.last().size).map { output.last()[it].asIOType0d().value }.maxIndex()
    }

    fun train(input: List<Double>, label: Int) {
        output[0] = input.map { IOType.IOType0d(it) }.toTypedArray()
        forward()
        calcDelta(label)
        backward()
    }

    companion object {
        fun create(
            inputConfig: Input0dConfig,
            centerConfig: List<Layer0dConfig>,
            outputConfig: Output0dConfig,
            random: Random,
            rate: Double,
        ): Network {
            val layers = listOf(inputConfig.toLayoutConfig()) + centerConfig + listOf(outputConfig.toLayoutConfig())

            // 値を全てバラバラにするために分割
            val weights: Array<Array<Array<IOType>>> =
                Array(layers.size - 1) { i ->
                    Array(layers[i].numOfNeuron) { Array(layers[i + 1].numOfNeuron) { layers[i + 1].createWeight(random) } }
                }

            val output: Array<Array<IOType>> = Array(layers.size) { i -> layers[i].createOutput() }
            val delta: Array<Array<Double>> = Array(layers.size + 1) { i ->
                Array(layers.getOrElse(i) { layers.last() }.numOfNeuron) { 0.0 }
            }
            val forward = {
                for (index in 0 until layers.size - 1) {
                    layers[index + 1].type.forward(
                        input = output[index],
                        output = output[index + 1],
                        weight = weights[index],
                        activationFunction = layers[index + 1].activationFunction,
                    )
                }
            }

            val calcDelta = { label: Int ->
                for (index in delta.last().indices) {
                    delta.last()[index] = if (index == label) 0.9 else 0.1
                }
                for (index in layers.size - 1 downTo 1) {
                    layers[index].type.calcDelta(
                        delta = delta[index],
                        output = output[index],
                        afterDelta = delta[index + 1],
                        afterWeight = weights.getOrElse(index) { arrayOf() },
                    )
                }
            }

            val backward = {
                for (index in 0 until layers.size - 1) {
                    layers[index + 1].type.backward(
                        weight = weights[index],
                        delta = delta[index + 1],
                        input = output[index],
                        rate = rate,
                    )
                }
            }

            return Network(
                weights = weights,
                output = output,
                delta = delta,
                forward = forward,
                calcDelta = calcDelta,
                backward = backward,
            )
        }
    }
}
