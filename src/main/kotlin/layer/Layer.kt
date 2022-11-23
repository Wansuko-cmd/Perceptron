package layer

import kotlin.random.Random

class Layer(
    private val output: List<Node>,
    private val rate: Double,
) {
    fun forward(value: List<Double>) = output.map { it.input(value) }

    fun train(value: List<Double>, label: Int): Layer =
        output
            .map { it to it.input(value) }
            .mapIndexed { index, (node, out) ->
                val error = error(if (index == label) 1.0 else 0.0, out.sum().step())
                Node(node.before?.zip(out)?.map { (no, o) -> no.first to no.second + rate * error * o })
            }
            .let { Layer(it, rate) }

    private fun error(label: Double, output: Double) = label - output

    companion object {
        fun create(input: Int, center: Int, output: Int, rate: Double): Layer {
            val inputNode = List(input) { Node.create() }
            val centerNode = List(center) { Node.create(inputNode) }
            val outputNode = List(output) { Node.create(centerNode) }
            return Layer(outputNode, rate)
        }
    }
}

class Node(
    val before: List<Pair<Node, Double>>?
) {
    fun input(value: List<Double>): List<Double> =
        (before?.map { it.first.input(value).sum().step() * it.second } ?: value)

    companion object {
        fun create(node: List<Node>): Node = Node(
            before = node.map { it to Random.nextDouble(from = -1.0, until = 1.0) }
        )

        fun create() = Node(null)
    }
}

fun Double.step() = if (this > 0) 1.0 else 0.0
