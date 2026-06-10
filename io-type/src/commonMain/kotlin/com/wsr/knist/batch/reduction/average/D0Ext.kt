package com.wsr.knist.batch.reduction.average

import com.wsr.knist.Backend
import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import kotlin.jvm.JvmName
import com.wsr.knist.scope.ScopeOp

@JvmName("batchD0sBatchAverage")
@ScopeOp
fun Batch<IOType.D0>.batchAverage(): IOType.D0 = IOType.D0(value = Backend.average(value))
