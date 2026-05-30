@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network

import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.converter.linear.LinearD1
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.output.mean.meanSquare
import com.wsr.knist.network.process.compute.affine.affine
import kotlin.test.Test
import kotlin.test.assertTrue
import okio.Buffer

class NetworkSerializerTest {

    private fun createNetwork(): Network<IOType.D1, IOType.D1> = NetworkBuilder.inputD1(
        converter = LinearD1(3),
        optimizer = Sgd(scheduler = Scheduler.Fix(rate = 0.01f)),
        initializer = Fixed(0.5f),
    )
        .affine(neuron = 4)
        .affine(neuron = 2)
        .meanSquare()

    private val input = IOType.d1(1f, 2f, 3f)

    @Test
    fun `JSON=文字列にシリアライズして復元できる`() = networkTestRule {
        val original = createNetwork()
        val restored = Network.fromJson<IOType.D1, IOType.D1>(original.toJson())

        assertContentEquals(expected = original.expect(input), actual = restored.expect(input))
    }

    @Test
    fun `JSON=BufferedSinkにシリアライズして復元できる`() = networkTestRule {
        val original = createNetwork()
        val buffer = Buffer()
        original.toJson(buffer)
        val restored = Network.fromJson<IOType.D1, IOType.D1>(buffer)

        assertContentEquals(expected = original.expect(input), actual = restored.expect(input))
    }

    @Test
    fun `CBOR=ByteArrayにシリアライズして復元できる`() = networkTestRule {
        val original = createNetwork()
        val restored = Network.fromCbor<IOType.D1, IOType.D1>(original.toCbor())

        assertContentEquals(expected = original.expect(input), actual = restored.expect(input))
    }

    @Test
    fun `CBOR=BufferedSinkにシリアライズして復元できる`() = networkTestRule {
        val original = createNetwork()
        val buffer = Buffer()
        original.toCbor(buffer)
        val restored = Network.fromCbor<IOType.D1, IOType.D1>(buffer)

        assertContentEquals(expected = original.expect(input), actual = restored.expect(input))
    }

    @Test
    fun `CBOR=JSONよりバイトサイズが小さい`() = networkTestRule {
        val network = createNetwork()
        val jsonSize = network.toJson().encodeToByteArray().size
        val cborSize = network.toCbor().size

        assertTrue(
            actual = cborSize < jsonSize,
            message = "CBOR ($cborSize bytes) should be smaller than JSON ($jsonSize bytes)",
        )
    }
}
