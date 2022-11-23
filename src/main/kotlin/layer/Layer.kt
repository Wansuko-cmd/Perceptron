package layer

import kotlin.random.Random

const val SEED = 160
val random = Random(SEED)

class Layer(
    val output: List<Node>,
    private val rate: Double,
) {
    fun forward(value: List<Double>) = output.map { it.input(value) }

    fun train(value: List<Double>, label: Int): Layer =
        output
            .map { it to it.input(value) }
            .mapIndexed { index, (node, out) ->
                val error = error(if (index == label) 1.0 else 0.0, out.sum().step())
                Node(
                    node
                        .before
                        ?.zip(out)
                        ?.map { (no, o) -> (no.first to no.second + rate * error * o) }
                )
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
    override fun toString(): String = before
        ?.mapIndexed { index, (_, weight) -> index to weight }
        ?.joinToString(prefix = "\n    ", postfix = "\n") { "[Node${it.first} weight: ${it.second}]" } ?: ""

    fun input(value: List<Double>): List<Double> {
        if (before?.map { it.first.before }?.all { it == null } == true) {
            return before.zip(value) { (_, weight), input -> input * weight }
        }

        return before?.map { (node, weight) ->
            node.input(value).sum().step() * weight
        } ?: value
    }


    companion object {
        fun create(node: List<Node>): Node = Node(
            before = node.map { it to random.nextDouble(from = -1.0, until = 1.0) }
        )

        fun create() = Node(null)
    }
}

fun Double.step() = if (this > 0.0) 1.0 else 0.0

fun <T : Comparable<T>> List<T>.maxIndex(): Int =
    this.foldIndexed(null) { index: Int, acc: Pair<Int, T>?, element: T ->
        when {
            acc == null -> index to element
            acc.second > element -> acc
            else -> index to element
        }
    }?.first ?: throw Exception()
