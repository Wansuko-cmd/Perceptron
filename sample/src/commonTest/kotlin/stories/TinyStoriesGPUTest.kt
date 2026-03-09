@file:Suppress("NonAsciiCharacters", "RemoveRedundantBackticks")

package stories

import com.wsr.Backend
import com.wsr.gpu.gpu
import kotlin.test.Test

class TinyStoriesGPUTest {
    @Test
    fun run() {
        Backend.set(gpu)
        evaluateModel()
    }
}
