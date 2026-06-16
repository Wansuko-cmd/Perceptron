package com.wsr.knist.network.optimizer

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.reduction.average.batchAverage
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.compare.lt
import com.wsr.knist.core.elementwise.compare.where.where
import com.wsr.knist.core.elementwise.math.pow
import com.wsr.knist.core.elementwise.math.sqrt
import com.wsr.knist.core.elementwise.operation.div.div
import com.wsr.knist.core.elementwise.operation.times.times
import com.wsr.knist.core.reduction.sum
import kotlinx.serialization.Serializable

interface Optimizer {
    fun d1(size: Int): D1
    fun d2(i: Int, j: Int): D2
    fun d3(i: Int, j: Int, k: Int): D3
    fun d4(i: Int, j: Int, k: Int, l: Int): D4

    @Serializable
    abstract class D1(private val _maxNorm: Float = Float.MAX_VALUE, private val _stepUnit: Int = 1) {
        private var _step: Int = 0
        protected val step: Int get() = _step / _stepUnit
        protected abstract fun IOScope.adapt(weight: IOType.D1, dw: IOType.D1): IOType.D1

        context(scope: IOScope)
        fun adapt(weight: IOType.D1, dw: IOType.D1, enableClip: Boolean = _maxNorm != Float.MAX_VALUE): IOType.D1 {
            if (enableClip) {
                with(scope) {
                    val norm = dw.pow(2).sum().sqrt()
                    val scale = _maxNorm / norm
                    val clipped = scale.where(condition = scale lt 1f, onFalse = 1f)
                    return adapt(weight, dw * clipped).also { _step++ }
                }
            }
            return with(scope) { adapt(weight, dw) }.also { _step++ }
        }

        context(scope: IOScope)
        fun adapt(
            weight: IOType.D1,
            dw: Batch<IOType.D1>,
            enableClip: Boolean = _maxNorm != Float.MAX_VALUE,
        ): IOType.D1 = with(scope) { adapt(weight, dw.batchAverage(), enableClip) }
    }

    @Serializable
    abstract class D2(private val _maxNorm: Float = Float.MAX_VALUE, private val _stepUnit: Int = 1) {
        private var _step: Int = 0
        protected val step: Int get() = _step / _stepUnit
        protected abstract fun IOScope.adapt(weight: IOType.D2, dw: IOType.D2): IOType.D2

        context(scope: IOScope)
        fun adapt(weight: IOType.D2, dw: IOType.D2, enableClip: Boolean = _maxNorm != Float.MAX_VALUE): IOType.D2 {
            if (enableClip) {
                with(scope) {
                    val norm = dw.pow(2).sum().sqrt()
                    val scale = _maxNorm / norm
                    val clipped = scale.where(condition = scale lt 1f, onFalse = 1f)
                    return adapt(weight, dw * clipped).also { _step++ }
                }
            }
            return with(scope) { adapt(weight, dw) }.also { _step++ }
        }

        context(scope: IOScope)
        fun adapt(
            weight: IOType.D2,
            dw: Batch<IOType.D2>,
            enableClip: Boolean = _maxNorm != Float.MAX_VALUE,
        ): IOType.D2 = with(scope) { adapt(weight, dw.batchAverage(), enableClip) }
    }

    @Serializable
    abstract class D3(private val _maxNorm: Float = Float.MAX_VALUE, private val _stepUnit: Int = 1) {
        private var _step: Int = 0
        protected val step: Int get() = _step / _stepUnit
        protected abstract fun IOScope.adapt(weight: IOType.D3, dw: IOType.D3): IOType.D3

        context(scope: IOScope)
        fun adapt(weight: IOType.D3, dw: IOType.D3, enableClip: Boolean = _maxNorm != Float.MAX_VALUE): IOType.D3 {
            if (enableClip) {
                with(scope) {
                    val norm = dw.pow(2).sum().sqrt()
                    val scale = _maxNorm / norm
                    val clipped = scale.where(condition = scale lt 1f, onFalse = 1f)
                    return adapt(weight, dw * clipped).also { _step++ }
                }
            }
            return with(scope) { adapt(weight, dw) }.also { _step++ }
        }

        context(scope: IOScope)
        fun adapt(
            weight: IOType.D3,
            dw: Batch<IOType.D3>,
            enableClip: Boolean = _maxNorm != Float.MAX_VALUE,
        ): IOType.D3 = with(scope) { adapt(weight, dw.batchAverage(), enableClip) }
    }

    @Serializable
    abstract class D4(private val _maxNorm: Float = Float.MAX_VALUE, private val _stepUnit: Int = 1) {
        private var _step: Int = 0
        protected val step: Int get() = _step / _stepUnit
        protected abstract fun IOScope.adapt(weight: IOType.D4, dw: IOType.D4): IOType.D4

        context(scope: IOScope)
        fun adapt(weight: IOType.D4, dw: IOType.D4, enableClip: Boolean = _maxNorm != Float.MAX_VALUE): IOType.D4 {
            if (enableClip) {
                with(scope) {
                    val norm = dw.pow(2).sum().sqrt()
                    val scale = _maxNorm / norm
                    val clipped = scale.where(condition = scale lt 1f, onFalse = 1f)
                    return adapt(weight, dw * clipped).also { _step++ }
                }
            }
            return with(scope) { adapt(weight, dw) }.also { _step++ }
        }

        context(scope: IOScope)
        fun adapt(
            weight: IOType.D4,
            dw: Batch<IOType.D4>,
            enableClip: Boolean = _maxNorm != Float.MAX_VALUE,
        ): IOType.D4 = with(scope) { adapt(weight, dw.batchAverage(), enableClip) }
    }
}
