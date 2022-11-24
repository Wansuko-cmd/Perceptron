package layer

import N
import java.util.UUID
import kotlin.math.exp
import kotlin.math.pow
import kotlin.random.Random

const val SEED = 160
var random = Random(SEED)

class Layer(
    private val output: List<Node>,
    private val rate: Double
) {
    fun forward(value: List<Double>): Int = output.map { it.getY(value) }.maxIndex()

    private fun error(label: Double, output: Double) = 0.5 * (label - output).pow(2.0)

    fun fd(input: List<Double>, label: Int) = output
        .flatMap { node -> node.toList().map { it + (node to Double.NEGATIVE_INFINITY) } }
        .map { node -> listOf(Double.NEGATIVE_INFINITY to node.first().first) + node.windowed(2) { (left, right) -> left.second to right.first } }
        .toBe(input, label)
        .map { it.copy(second = it.second.fixWeight()) }
        .flatMap { node -> node.second.toList().map { listOf(Double.NEGATIVE_INFINITY to node.second) + it } }
        .map { node -> node.windowed(2) { (left, right) -> left.second to right.first } + (node.last().second to Double.NEGATIVE_INFINITY) }
        .toG()
        .let { Layer(it.map { it.first }, rate) }

    fun List<List<Pair<N, Double>>>.toG(): List<Pair<Node, Double>> = when {
        this.all { it.size == 1 } -> this.map {
            val (_, weight) = it.first()
            Node.create { it.relu() } to weight
        }
        else ->
            this
                .groupBy { it.last().first.id }
                .mapKeys { it.value.last().last() }
                .mapValues { (_, value) -> value.map { it.dropLast(1) }.toG() }
                .map {
                    Node(it.value) to it.key.second
                }
    }

    fun List<List<Pair<Double, Node>>>.toBe(input: List<Double>, label: Int): List<Pair<Double, N>> = when {
        this.all { it.size == 1 } -> this.map {
            val (weight, node) = it.first()
            val (v, y) = node.getVY(input)
            val error = if (node.id == label.toString()) error(1.0, y) else error(0.0, y)
            weight to N.OutputN(v = v, y = y, t = error, node.id)
        }
        else ->
            this
                .groupBy { it.first().second.id }
                .mapKeys { it.value.first().first() }
                .mapValues { (_, value) -> value.map { it.drop(1) }.toBe(input, label) }
                .map {
                    val (v, y) = it.key.second.getVY(input)
                    it.key.first to N.NormalN(
                        v = v,
                        y = y,
                        nodes = it.value,
                        it.key.second.id,
                    )
                }
    }

    companion object {
        fun create(input: Int, center: Int, output: Int, rate: Double): Layer {
            val inputNode = List(input) { Node.create { it.relu() } }
            val centerNode = List(center) { Node.create(inputNode) { it.relu() } }
            val outputNode = List(output) { index -> Node.create(centerNode, id = index.toString()) { it.sigmoid() } }
            return Layer(outputNode, rate)
        }
    }
}

class Node(
    val before: List<Pair<Node, Double>>?,
    val f: (Double) -> Double = { it.step() },
    val id: String = UUID.randomUUID().toString()
) {

    fun getVY(value: List<Double>): Pair<Double, Double> = when {
        before == null -> 0.0 to 0.0
        before.map { it.first.before }.all { it == null } -> {
            0.0 to value.sum()
        }
        else -> {
            before
                .sumOf { (node, weight) -> node.getY(value) * weight }
                .let { it to f(it) }
        }
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

    fun toList(): List<List<Pair<Node, Double>>> =
        before?.flatMap { (node, weight) ->
            node.toList().map { it + (node to weight) }
        } ?: listOf(listOf())

    companion object {
        fun create(
            node: List<Node>,
            id: String = UUID.randomUUID().toString(),
            f: (Double) -> Double,
        ): Node = Node(
            before = node.map { it to random.nextDouble(from = -1.0, until = 1.0) },
            f = f,
            id = id,
        )

        fun create(f: (Double) -> Double) = Node(null, f = f)
    }
}

fun Double.step() = if (this > 0.0) 1.0 else 0.0

fun Double.relu() = if (this > 0.0) this else 0.0

fun Double.sigmoid() = 1 / (1 + exp(-this))

fun <T : Comparable<T>> List<T>.maxIndex(): Int =
    this.foldIndexed(null) { index: Int, acc: Pair<Int, T>?, element: T ->
        when {
            acc == null -> index to element
            acc.second > element -> acc
            else -> index to element
        }
    }?.first ?: throw Exception()
