package com.wsr.knist.network.process

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType

data class Context(val input: Batch<IOType>)
