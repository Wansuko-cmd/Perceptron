package com.wsr.knist.batch.elementwise.compare

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.compare.EQUALS_ABSOLUTE_TOLERANCE
import com.wsr.knist.core.elementwise.compare.EQUALS_RELATIVE_TOLERANCE
import kotlin.jvm.JvmName

@JvmName("infixBatchD4sEqFloat")
infix fun Batch<IOType.D4>.eq(other: Float): Batch<IOType.D4> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun Batch<IOType.D4>.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D4> {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("infixBatchD4sEqD4s")
infix fun Batch<IOType.D4>.eq(other: Batch<IOType.D4>): Batch<IOType.D4> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@JvmName("batchD4sEqD4s")
fun Batch<IOType.D4>.eq(
    other: Batch<IOType.D4>,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D4> {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sGtFloat")
infix fun Batch<IOType.D4>.gt(other: Float): Batch<IOType.D4> {
    val result = Backend.greaterThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sGtD4s")
infix fun Batch<IOType.D4>.gt(other: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.greaterThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sLtFloat")
infix fun Batch<IOType.D4>.lt(other: Float): Batch<IOType.D4> {
    val result = Backend.lessThan(value, other)
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("batchD4sLtD4s")
infix fun Batch<IOType.D4>.lt(other: Batch<IOType.D4>): Batch<IOType.D4> {
    val result = Backend.lessThan(value, other.value)
    return Batch(size = size, shape = shape, value = result)
}
