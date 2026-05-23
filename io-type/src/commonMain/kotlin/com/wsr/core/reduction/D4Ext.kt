package com.wsr.core.reduction

import com.wsr.Backend
import com.wsr.core.IOType

fun IOType.D4.sum(): IOType.D0 = IOType.D0(Backend.sum(value))
