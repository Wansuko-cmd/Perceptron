@file:Suppress("NonAsciiCharacters", "UNCHECKED_CAST")

package com.wsr.knist.network.process.compute.attention

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOType
import com.wsr.knist.core.d1
import com.wsr.knist.core.d2
import com.wsr.knist.core.get
import com.wsr.knist.network.assertContentEquals
import com.wsr.knist.network.networkScopeTestRule
import com.wsr.knist.network.optimizer.Scheduler
import com.wsr.knist.network.optimizer.sgd.Sgd
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.compute.attention.bias.AttentionBiasD2
import kotlin.test.Test

class AttentionD2Test {
    private val outputI = 3
    private val outputJ = 4
    private val numOfHeads = 2
    private val dim = outputJ / numOfHeads

    val target
        get() = AttentionD2(
            outputI = outputI,
            outputJ = outputJ,
            numOfHeads = numOfHeads,
            dim = dim,
            biases = listOf(AttentionBiasD2.Causal(outputI)),
            weightQ = IOType.d2(4, numOfHeads * dim) { i, j -> -i * 1f + j * 0.1f },
            weightK = IOType.d2(4, numOfHeads * dim) { i, j -> -i * 2f + j * 0.2f },
            weightV = IOType.d2(4, numOfHeads * dim) { i, j -> -i * 3f + j * 0.3f },
            weightO = IOType.d2(numOfHeads * dim, outputJ) { i, j -> -i * 4f + j * 0.4f },
            optimizerQ = Sgd(Scheduler.Fix(0.01f)).d2(outputJ, numOfHeads * dim),
            optimizerK = Sgd(Scheduler.Fix(0.01f)).d2(outputJ, numOfHeads * dim),
            optimizerV = Sgd(Scheduler.Fix(0.01f)).d2(outputJ, numOfHeads * dim),
            optimizerO = Sgd(Scheduler.Fix(0.01f)).d2(numOfHeads * dim, outputJ),
        )

    val input
        get() = Batch.of(
            IOType.d2(
                IOType.d1(4) { it * 0.2f },
                IOType.d1(4) { it * 0.3f },
                IOType.d1(4) { it * 0.4f },
            ),
            IOType.d2(
                IOType.d1(4) { it * -0.3f },
                IOType.d1(4) { it * -0.2f },
                IOType.d1(4) { it * -0.1f },
            ),
        )

    @Test
    fun `expect=注目度を計算`() = networkScopeTestRule {
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(181.4400f, 168.8640f, 156.2880f, 143.7120f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(272.1598f, 253.2958f, 234.4318f, 215.5678f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(362.8800f, 337.7280f, 312.5760f, 287.4240f),
            actual = actual[0][2],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-272.1600f, -253.2960f, -234.4320f, -215.5680f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-272.1488f, -253.2853f, -234.4218f, -215.5583f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-271.1732f, -252.3620f, -233.5508f, -214.7395f),
            actual = actual[1][2],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=逆伝播を行い勾配を返す`() = networkScopeTestRule {
        val actual = with(target) {
            _train(input = input, context = Context(input), calcDelta = { it })
        } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(-10267.7650f, 32172.5840f, 74612.9300f, 117053.2700f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-15402.5690f, 48261.6800f, 111925.9200f, 175590.1900f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-20536.1020f, 64346.9300f, 149229.9700f, 234113.0000f),
            actual = actual[0][2],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(46719.8320f, -146507.8400f, -339735.5300f, -532963.2000f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-547.6068f, 1833.6252f, 4214.8574f, 6596.0900f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(735.2773f, -2390.929f, -5517.1353f, -8643.3410f),
            actual = actual[1][2],
            absoluteTolerance = 1e-4f,
        )
    }

    @Test
    fun `train=重みを更新する`() = networkScopeTestRule {
        val target = target
        with(target) { _train(input = input, context = Context(input), calcDelta = { it }) }
        val actual = with(target) { _expect(input = input, context = Context(input)) } as Batch<IOType.D2>

        assertContentEquals(
            expected = IOType.d1(44581.2730f, 41340.8200f, 38100.3670f, 34859.9060f),
            actual = actual[0][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(63399.9730f, 58793.3200f, 54186.6640f, 49580.0040f),
            actual = actual[0][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(86385.9140f, 80108.1640f, 73830.4100f, 67552.6500f),
            actual = actual[0][2],
            absoluteTolerance = 1e-4f,
        )

        assertContentEquals(
            expected = IOType.d1(-66871.9100f, -62011.2300f, -57150.5470f, -52289.8630f),
            actual = actual[1][0],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-61691.9340f, -57210.2500f, -52728.5620f, -48246.8700f),
            actual = actual[1][1],
            absoluteTolerance = 1e-4f,
        )
        assertContentEquals(
            expected = IOType.d1(-53664.8900f, -49770.4300f, -45875.9700f, -41981.5080f),
            actual = actual[1][2],
            absoluteTolerance = 1e-4f,
        )
    }
}
