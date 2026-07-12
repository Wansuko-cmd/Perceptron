package com.wsr.knist.network.optimizer.adam

import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.d3
import com.wsr.knist.core.d4
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.optimizer.Scheduler
import kotlin.math.pow
import kotlinx.serialization.Serializable

private const val E = 1e-8f

@Serializable
data class AdamW(
    private val scheduler: Scheduler,
    private val momentum: Float = 0.9f,
    private val rms: Float = 0.999f,
    private val decay: Float = 0.01f,
    private val maxNorm: Float = Float.MAX_VALUE,
    private val stepUnit: Int = 1,
) : Optimizer {
    override fun d1(i: Int): Optimizer.D1 = AdamWD1(
        scheduler = scheduler,
        momentum = momentum,
        rms = rms,
        decay = decay,
        maxNorm = maxNorm,
        stepUnit = stepUnit,
        shape = listOf(i),
    )

    override fun d2(i: Int, j: Int): Optimizer.D2 = AdamWD2(
        scheduler = scheduler,
        momentum = momentum,
        rms = rms,
        decay = decay,
        maxNorm = maxNorm,
        stepUnit = stepUnit,
        shape = listOf(i, j),
    )

    override fun d3(i: Int, j: Int, k: Int): Optimizer.D3 = AdamWD3(
        scheduler = scheduler,
        momentum = momentum,
        rms = rms,
        decay = decay,
        maxNorm = maxNorm,
        stepUnit = stepUnit,
        shape = listOf(i, j, k),
    )

    override fun d4(i: Int, j: Int, k: Int, l: Int): Optimizer.D4 = AdamWD4(
        scheduler = scheduler,
        momentum = momentum,
        rms = rms,
        decay = decay,
        maxNorm = maxNorm,
        stepUnit = stepUnit,
        shape = listOf(i, j, k, l),
    )
}

@Serializable
internal data class AdamWD1(
    private val scheduler: Scheduler,
    private val momentum: Float,
    private val rms: Float,
    private val decay: Float,
    private val maxNorm: Float,
    private val stepUnit: Int,
    private val shape: List<Int>,
) : Optimizer.D1(maxNorm, stepUnit) {
    private var m: IOType.D1.Global = IOType.d1(shape)
    private var v: IOType.D1.Global = IOType.d1(shape)
    private var t: Int = 0

    override fun IOScope.adapt(weight: IOType.D1, dw: IOType.D1): IOType.D1 {
        t += 1

        m = (momentum * m + (1 - momentum) * dw).toGlobal()
        v = (rms * v + (1 - rms) * dw.pow(2)).toGlobal()

        val mHat = m / (1f - momentum.pow(t.toFloat()))
        val vHat = v / (1f - rms.pow(t.toFloat()))

        val rate = scheduler.calcRate(step = step)
        return (1 - rate * decay) * weight - rate * mHat / vHat.sqrt(e = E)
    }
}

@Serializable
internal data class AdamWD2(
    private val scheduler: Scheduler,
    private val momentum: Float,
    private val rms: Float,
    private val decay: Float,
    private val maxNorm: Float,
    private val stepUnit: Int,
    private val shape: List<Int>,
) : Optimizer.D2(maxNorm, stepUnit) {
    private var m: IOType.D2.Global = IOType.d2(shape)
    private var v: IOType.D2.Global = IOType.d2(shape)
    private var t: Int = 0

    override fun IOScope.adapt(weight: IOType.D2, dw: IOType.D2): IOType.D2 {
        t += 1

        m = (momentum * m + (1 - momentum) * dw).toGlobal()
        v = (rms * v + (1 - rms) * dw.pow(2)).toGlobal()

        val mHat = m / (1f - momentum.pow(t.toFloat()))
        val vHat = v / (1f - rms.pow(t.toFloat()))

        val rate = scheduler.calcRate(step = step)
        return (1 - rate * decay) * weight - rate * mHat / vHat.sqrt(e = E)
    }
}

@Serializable
internal data class AdamWD3(
    private val scheduler: Scheduler,
    private val momentum: Float,
    private val rms: Float,
    private val decay: Float,
    private val maxNorm: Float,
    private val stepUnit: Int,
    private val shape: List<Int>,
) : Optimizer.D3(maxNorm, stepUnit) {
    private var m: IOType.D3.Global = IOType.d3(shape)
    private var v: IOType.D3.Global = IOType.d3(shape)
    private var t: Int = 0

    override fun IOScope.adapt(weight: IOType.D3, dw: IOType.D3): IOType.D3 {
        t += 1

        m = (momentum * m + (1 - momentum) * dw).toGlobal()
        v = (rms * v + (1 - rms) * dw.pow(2)).toGlobal()

        val mHat = m / (1f - momentum.pow(t.toFloat()))
        val vHat = v / (1f - rms.pow(t.toFloat()))

        val rate = scheduler.calcRate(step = step)
        return (1 - rate * decay) * weight - rate * mHat / vHat.sqrt(e = E)
    }
}

@Serializable
internal data class AdamWD4(
    private val scheduler: Scheduler,
    private val momentum: Float,
    private val rms: Float,
    private val decay: Float,
    private val maxNorm: Float,
    private val stepUnit: Int,
    private val shape: List<Int>,
) : Optimizer.D4(maxNorm, stepUnit) {
    private var m: IOType.D4 = IOType.d4(shape)
    private var v: IOType.D4 = IOType.d4(shape)
    private var t: Int = 0

    override fun IOScope.adapt(weight: IOType.D4, dw: IOType.D4): IOType.D4 {
        t += 1

        m = (momentum * m + (1 - momentum) * dw).toGlobal()
        v = (rms * v + (1 - rms) * dw.pow(2)).toGlobal()

        val mHat = m / (1f - momentum.pow(t.toFloat()))
        val vHat = v / (1f - rms.pow(t.toFloat()))

        val rate = scheduler.calcRate(step = step)
        return (1 - rate * decay) * weight - rate * mHat / vHat.sqrt(e = E)
    }
}
