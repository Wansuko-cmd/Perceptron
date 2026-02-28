package com.wsr.batch.compare.equals

import com.wsr.Backend
import com.wsr.batch.Batch
import com.wsr.core.IOType
import com.wsr.core.compare.equals.EQUALS_ABSOLUTE_TOLERANCE
import com.wsr.core.compare.equals.EQUALS_RELATIVE_TOLERANCE
import kotlin.jvm.JvmName

@JvmName("infixBatchD2sEqFloat")
infix fun Batch<IOType.D2>.eq(other: Float): Batch<IOType.D2> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

fun Batch<IOType.D2>.eq(
    other: Float,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D2> {
    val result = Backend.equals(
        x = value,
        y = other,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch(size = size, shape = shape, value = result)
}

@JvmName("infixBatchD2sEqD2s")
infix fun Batch<IOType.D2>.eq(other: Batch<IOType.D2>): Batch<IOType.D2> = eq(
    other = other,
    absoluteTolerance = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance = EQUALS_RELATIVE_TOLERANCE,
)

@JvmName("batchD2sEqD2s")
fun Batch<IOType.D2>.eq(
    other: Batch<IOType.D2>,
    absoluteTolerance: Float = EQUALS_ABSOLUTE_TOLERANCE,
    relativeTolerance: Float = EQUALS_RELATIVE_TOLERANCE,
): Batch<IOType.D2> {
    val result = Backend.equals(
        x = value,
        y = other.value,
        absoluteTolerance = absoluteTolerance,
        relativeTolerance = relativeTolerance,
    )
    return Batch(size = size, shape = shape, value = result)
}
