@file:Suppress("UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.skip

import com.wsr.knist.network.GraphBuilder
import com.wsr.knist.network.GraphScope
import com.wsr.knist.network.join.add.add

fun GraphBuilder.Node.D3.skip(builder: GraphBuilder.Node.D3.() -> GraphBuilder.Node.D3): GraphBuilder.Node.D3 {
    val branch = builder()
    return if (this == branch) this else GraphScope.add(this, branch)
}
