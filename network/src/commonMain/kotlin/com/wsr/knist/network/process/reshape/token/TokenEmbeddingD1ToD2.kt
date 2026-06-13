package com.wsr.knist.network.process.reshape.token

import com.wsr.knist.batch.Batch
import com.wsr.knist.batch.d1
import com.wsr.knist.batch.index.gather
import com.wsr.knist.batch.index.scatterAdd
import com.wsr.knist.core.IOType
import com.wsr.knist.core.elementwise.operation.div.div
import com.wsr.knist.network.NetworkBuilder
import com.wsr.knist.network.initializer.WeightInitializer
import com.wsr.knist.network.optimizer.Optimizer
import com.wsr.knist.network.process.Context
import com.wsr.knist.network.process.reshape.Reshape
import kotlinx.serialization.Serializable

@Serializable
class TokenEmbeddingD1ToD2 internal constructor(
    override val outputX: Int,
    override val outputY: Int,
    private val vocabSize: Int,
    private val optimizer: Optimizer.D2,
    private var weight: IOType.D2,
) : Reshape.D1ToD2() {

    override fun expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D2> = input.gather(other = weight)

    override fun train(
        input: Batch<IOType.D1>,
        context: Context,
        calcDelta: (Batch<IOType.D2>) -> Batch<IOType.D2>,
    ): Batch<IOType.D1> {
        val output = input.gather(other = weight)
        val delta = calcDelta(output)

        val dw = delta.scatterAdd(other = input, n = vocabSize)
        weight = optimizer.adapt(weight = weight, dw = dw / input.size.toFloat())

        // Embedding層は離散的なので、入力への勾配は意味を持たない
        // しかし型の整合性のため、ダミーのD1を返す
        return Batch.d1(input.size, input.shape)
    }
}

fun <T> NetworkBuilder.D1<T>.tokenEmbedding(
    vocabSize: Int,
    tokenSize: Int,
    optimizer: Optimizer = this.optimizer,
    initializer: WeightInitializer = this.initializer,
): NetworkBuilder.D2<T> = addReshape(
    reshape = TokenEmbeddingD1ToD2(
        outputX = inputSize,
        outputY = tokenSize,
        vocabSize = vocabSize,
        optimizer = optimizer.d2(vocabSize, tokenSize),
        weight = initializer.d2(
            input = listOf(vocabSize),
            output = listOf(tokenSize),
            x = vocabSize,
            y = tokenSize,
        ),
    ),
)
