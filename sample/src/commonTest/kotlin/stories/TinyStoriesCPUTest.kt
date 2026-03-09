@file:Suppress("NonAsciiCharacters", "RemoveRedundantBackticks")

package stories

import com.wsr.Backend
import com.wsr.cpu.cpu
import kotlin.test.Test


class TinyStoriesCPUTest {
    @Test
    fun run() {
        Backend.set(cpu)
        evaluateModel()
    }
}
