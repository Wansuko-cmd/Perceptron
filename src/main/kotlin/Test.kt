
import layer.step

sealed interface N {
    val delta: Double
    val id: String

    class OutputN(
        val v: Double,
        val y: Double,
        val t: Double,
        override val id: String,
    ) : N {
        override val delta: Double = (y - t) * (1 - v) * v
        override fun fixWeight(): N = this
        override fun toList(): List<List<Pair<Double, N>>> = listOf(listOf())
    }

    class NormalN(
        val v: Double,
        val y: Double,
        val nodes: List<Pair<Double, N>>,
        override val id: String,
    ) : N {
        override fun toString() = nodes.toString()
        override val delta: Double =
            v.step() * nodes.sumOf { (weight, node) -> node.delta * weight }

        override fun fixWeight(): N =
            nodes
                .map { (weight, node) -> (weight - node.delta * y) to node }
                .let { NormalN(v, y, it, id) }

        override fun toList(): List<List<Pair<Double, N>>> =
            nodes.flatMap { (weight, node) -> node.toList().map {  listOf((weight to node)) + it } }
    }

    fun fixWeight(): N

    fun toList(): List<List<Pair<Double, N>>>
}

sealed interface TestN {
    object OutputN : TestN
    class NormalN(val nodes: List<Pair<TestN, Double>>) : TestN
}
