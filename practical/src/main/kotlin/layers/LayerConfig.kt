package layers

import kotlin.random.Random

interface LayerConfig<T : IOType> {
    val numOfNeuron: Int
    val numOfOutput: Int
    val activationFunction: (Double) -> Double
    val type: LayerType
    fun createWeight(random: Random): T
    fun createOutput(): T
}
