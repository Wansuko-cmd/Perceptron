package com.wsr.network.process

import com.wsr.batch.Batch
import com.wsr.core.IOType

data class Context(val input: Batch<IOType>)
