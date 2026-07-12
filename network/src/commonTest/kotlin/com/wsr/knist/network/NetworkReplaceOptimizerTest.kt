@file:Suppress("NonAsciiCharacters")

package com.wsr.knist.network

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.network.converter.raw.RawD1
import com.wsr.knist.network.initializer.Fixed
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.freeze.Freeze
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.output.mean.meanSquare
import com.wsr.knist.network.process.compute.affine.affine
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

class NetworkReplaceOptimizerTest {

    private fun createNetwork(): Network<Batch<IOType.D1>, Batch<IOType.D1>> = NetworkBuilder.inputD1(
        converter = RawD1(3),
        optimizer = Sgd(scheduler = Scheduler.Fix(rate = 0.01f)),
        initializer = Fixed(0.5f),
    )
        .affine(neuron = 4, id = "first")
        .affine(neuron = 2, id = "last")
        .meanSquare()

    private val input = Batch(1) { IOType.d1(1f, 2f, 3f) }
    private val label = Batch(1) { IOType.d1(0f, 1f) }

    @Test
    fun `replace=Freezeで凍結した層は学習しても出力が変わらない`() = networkTestRule {
        val frozen = createNetwork().replace(condition = { true }, optimizer = Freeze)

        runTest {
            val before = frozen.expect(input)
            frozen.train(input = input, label = label)
            val after = frozen.expect(input)

            assertSameOutput(before, after)
        }
    }

    @Test
    fun `replace=凍結した層より前段にdeltaが流れて学習が継続する`() = networkTestRule {
        val frozen = createNetwork().replace(condition = { it.id == "last" }, optimizer = Freeze)

        runTest {
            val before = frozen.expect(input)
            frozen.train(input = input, label = label)
            val after = frozen.expect(input)

            // 後段が凍結されていても前段（first）は学習するため、出力は変化する
            assertDifferentOutput(before, after)
        }
    }

    @Test
    fun `replace=通常のOptimizerに差し替えると学習が再開する`() = networkTestRule {
        val frozen = createNetwork().replace(condition = { true }, optimizer = Freeze)
        val unfrozen = frozen.replace(condition = { true }, optimizer = Sgd(scheduler = Scheduler.Fix(rate = 0.01f)))

        runTest {
            val before = unfrozen.expect(input)
            unfrozen.train(input = input, label = label)
            val after = unfrozen.expect(input)

            assertDifferentOutput(before, after)
        }
    }

    @Test
    fun `replace=元のNetworkは変更されない`() = networkTestRule {
        val original = createNetwork()
        original.replace(condition = { true }, optimizer = Freeze)

        runTest {
            val before = original.expect(input)
            original.train(input = input, label = label)
            val after = original.expect(input)

            assertDifferentOutput(before, after)
        }
    }

    @Test
    fun `replace=凍結したNetworkはシリアライズしても凍結が維持される`() = networkTestRule {
        val frozen = createNetwork().replace(condition = { true }, optimizer = Freeze)
        val restored = Network.fromJson<Batch<IOType.D1>, Batch<IOType.D1>>(frozen.toJson())

        runTest {
            val before = restored.expect(input)
            restored.train(input = input, label = label)
            val after = restored.expect(input)

            assertSameOutput(before, after)
        }
    }

    private fun assertSameOutput(expected: Batch<IOType.D1>, actual: Batch<IOType.D1>) {
        val e = expected.value.toFloatArray()
        val a = actual.value.toFloatArray()
        assertTrue(
            e.size == a.size && e.indices.all { e[it] == a[it] },
            "expected ${e.toList()}, actual ${a.toList()}",
        )
    }

    private fun assertDifferentOutput(before: Batch<IOType.D1>, after: Batch<IOType.D1>) {
        val b = before.value.toFloatArray()
        val a = after.value.toFloatArray()
        assertTrue(
            b.indices.any { abs(b[it] - a[it]) > 1e-6f },
            "output did not change: ${b.toList()}",
        )
    }
}
