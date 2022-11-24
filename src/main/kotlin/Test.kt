import layer.Node

sealed interface N {
    val delta: Double

    class OutputN(
        val v: Double,
        val y: Double,
        val t: Double,
    ) : N {
        override val delta: Double = (y - t) * (1 - v) * v
    }

    class NormalN(
        val v: Double,
        val nodes: List<Pair<N, Double>>,
    ) : N {
        override val delta: Double =
            (1 - v) * v * nodes.sumOf { (node, weight) -> node.delta * weight }
    }
}

class L(
    val nodes: List<N>
) {
    fun study() {
        nodes.map { it.delta }
    }
}