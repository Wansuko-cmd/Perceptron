package layer

import kotlin.random.Random

const val SEED = 160
val random = Random(SEED)

class Layer(
    private val output: List<Node>,
    private val rate: Double
) {
    fun forward(value: List<Double>): Int = output.map { it.getY(value) }.maxIndex()

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
    val before: List<Pair<Node, Double>>?,
    val f: (Double) -> Double = { it.step() },
) {
    override fun toString(): String = before
        ?.mapIndexed { index, (_, weight) -> index to weight }
        ?.joinToString(prefix = "\n    ", postfix = "\n") { "[Node${it.first} weight: ${it.second}]" } ?: ""

    fun input(value: List<Double>): List<Double> = when {
        // 到達し得ないコード
        before == null -> throw Exception("Node::input, 到達しないはずのコード")

        // 入力層の次の層だとしたら入力に重みを掛けたものを返す
        before.map { it.first.before }.all { it == null } ->
            before.zip(value) { (_, weight), input -> input * weight }

        // それ以外の層は前の層の出力を計算して重みを掛けたものを返す
        else ->
            before
                .map { (node, weight) -> node.getY(value) * weight }
                .map { f(it) }
    }

    fun getY(value: List<Double>): Double = when {
        before == null -> throw Exception()
        before.map { it.first.before }.all { it == null } -> {
            val v = before.zip(value) { (_, weight), input -> input * weight }.sum()
            val y = f(v)
            y
        }
        else -> {
            val v = before.sumOf { (node, weight) -> node.getY(value) * weight }
            val y = f(v)
            y
        }
    }

    fun g(value: List<Double>) {
        when {
            before == null -> throw Exception()
            before.map { it.first.before }.all { it == null } -> {
                val v = before.zip(value) { (_, weight), input -> input * weight }.sum()
                val y = f(v)
                y
            }
            else -> {
                val v = before.sumOf { (node, weight) -> node.getY(value) * weight }
                val y = f(v)
                y
            }
        }
    }

    companion object {
        fun create(node: List<Node>): Node = Node(
            before = node.map { it to random.nextDouble(from = -1.0, until = 1.0) }
        )

        fun create() = Node(null)
    }
}

fun Double.step() = if (this > 0.0) 1.0 else 0.0

fun Double.relu() = if (this > 0.0) this else 0.0

fun <T : Comparable<T>> List<T>.maxIndex(): Int =
    this.foldIndexed(null) { index: Int, acc: Pair<Int, T>?, element: T ->
        when {
            acc == null -> index to element
            acc.second > element -> acc
            else -> index to element
        }
    }?.first ?: throw Exception()
