package common

import org.jetbrains.bio.viktor.F64Array
import kotlin.math.exp

fun step(x: Double) = if (x > 0.0) 1.0 else 0.0

fun relu(x: Double) = if (x > 0.0) x else 0.0

fun sigmoid(x: Double) = 1 / (1 + exp(-x))

/**
 * 正方形のkernelじゃないとエラーを投げる
 * また奇数サイズのkernelを想定
 */
fun List<List<Double>>.conv(kernel: List<List<Double>>, f: (Double) -> Double = { it }): List<List<Double>> {
    val output = mutableListOf<List<Double>>()
    for (i in 0 until (this.size - kernel.size)) {
        val o = mutableListOf<Double>()
        for (j in 0 until (this.size - kernel.size)) {
            kernel.indices.sumOf { p ->
                kernel.indices.sumOf { q ->
                    f(this[i + p][j + q] * kernel[p][q])
                }
            }.let { o.add(it) }
        }
        output.add(o)
    }
    return output
}

/**
 * F64Array[横, 縦, チャンネル]
 */
fun F64Array.conv(kernel: F64Array): F64Array {
    val (wx, wy) = kernel.shape
    val out = F64Array(shape = shape)
    for (i in 0 until wx) {
        out
            .slice(from = 0, to = out.shape[0] - wx + 1, axis = 0) +=
            this.slice(from = i, to = i + this.shape[0] - wx + 1, axis = 0)
        for (j in 0 until wy) {
            out
                .slice(from = 0, to = out.shape[1] - wy + 1, axis = 1) +=
                this.slice(from = j, to = j + this.shape[1] - wy + 1, axis = 1) * kernel[i, j, 0]
        }
    }
    return out.slice(from = 0, to = this.shape[0] - wx + 1, axis = 0).slice(from = 0, to = this.shape[1] - wy + 1, axis = 1)
}

fun List<List<Double>>.add(other: List<List<Double>>): List<List<Double>> {
    val output = mutableListOf<List<Double>>()
    for ((i, e) in this.withIndex()) {
        val o = mutableListOf<Double>()
        for ((j, _) in e.withIndex()) {
            o.add(this[i][j] + other[i][j])
        }
        output.add(o)
    }
    return output
}
