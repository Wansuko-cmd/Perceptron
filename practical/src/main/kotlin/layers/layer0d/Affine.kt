@file:Suppress("OVERRIDE_BY_INLINE", "NOTHING_TO_INLINE")

package layers.layer0d

import common.step
import layers.IOType
import layers.LayerType

object Affine : LayerType {
    override inline fun forward(
        input: IOType,
        output: IOType,
        weight: Array<IOType>,
        activationFunction: (Double) -> Double,
    ) {
        val inputArray = input.asIOType0d().value
        val outputArray = output.asIOType0d().value
        for (outputIndex in outputArray.indices) {
            var sum = 0.0
            for (inputIndex in inputArray.indices) {
                sum += inputArray[inputIndex] * weight[inputIndex].asIOType0d().value[outputIndex]
            }
            outputArray[outputIndex] = activationFunction(sum)
        }
    }

    /**
     * delta -> 計算結果を格納するdelta
     * output -> 自分の層の出力
     * afterDelta -> 後ろの層のdelta
     * afterWeight -> 自分と後ろの層の重み Array[前の層のニューロン][後ろの層のニューロン]
     */
    override inline fun calcDelta(
        delta: Array<Double>,
        output: IOType,
        afterDelta: Array<Double>,
        afterWeight: Array<IOType>,
    ) {
        val outputArray = output.asIOType0d().value
        for (i in delta.indices) {
            val afterWeightArray = afterWeight[i].asIOType0d().value
            delta[i] = step(outputArray[i]) * (afterWeightArray.indices)
                .sumOf { afterDelta[it] * afterWeightArray[it] }
        }
    }

    override inline fun backward(
        weight: Array<IOType>,
        delta: Array<Double>,
        input: IOType,
        rate: Double,
    ) {
        val inputArray = input.asIOType0d().value
        for (before in weight.indices) {
            for (after in weight[before].asIOType0d().value.indices) {
                weight[before].asIOType0d().value[after] -= rate * delta[after] * inputArray[before]
            }
        }
    }
}
