package com.wsr.knist.network.process

import com.wsr.knist.batch.Batch
import com.wsr.knist.core.IOScope
import com.wsr.knist.core.IOType
import kotlinx.serialization.Serializable

@Suppress("UNCHECKED_CAST")
sealed interface Reshape : Process {
    @Serializable
    abstract class D1ToD2 : Reshape {
        abstract val inputI: Int

        abstract val outputI: Int
        abstract val outputJ: Int

        protected abstract fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D2>

        protected abstract fun IOScope.train(
            input: Batch<IOType.D1>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
        ): Batch<IOType.D1>

        final override fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType> = expect(
            input = input as Batch<IOType.D1>,
            context = context,
        )

        final override fun IOScope._train(
            input: Batch<IOType>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
        ): Batch<IOType> = train(
            input = input as Batch<IOType.D1>,
            context = context,
            calcDelta = calcDelta as IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
        )
    }

    @Serializable
    abstract class D1ToD3 : Reshape {
        abstract val inputI: Int

        abstract val outputI: Int
        abstract val outputJ: Int
        abstract val outputK: Int

        protected abstract fun IOScope.expect(input: Batch<IOType.D1>, context: Context): Batch<IOType.D3>

        protected abstract fun IOScope.train(
            input: Batch<IOType.D1>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
        ): Batch<IOType.D1>

        final override fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType> =
            expect(input = input as Batch<IOType.D1>, context = context)

        final override fun IOScope._train(
            input: Batch<IOType>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
        ): Batch<IOType> = train(
            input = input as Batch<IOType.D1>,
            context = context,
            calcDelta = calcDelta as IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
        )
    }

    @Serializable
    abstract class D2ToD1 : Reshape {
        abstract val inputI: Int
        abstract val inputJ: Int

        abstract val outputI: Int

        protected abstract fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D1>

        protected abstract fun IOScope.train(
            input: Batch<IOType.D2>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
        ): Batch<IOType.D2>

        final override fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType> = expect(
            input = input as Batch<IOType.D2>,
            context = context,
        )

        final override fun IOScope._train(
            input: Batch<IOType>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
        ): Batch<IOType> = train(
            input = input as Batch<IOType.D2>,
            context = context,
            calcDelta = calcDelta as IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
        )
    }

    @Serializable
    abstract class D2ToD3 : Reshape {
        abstract val inputI: Int
        abstract val inputJ: Int

        abstract val outputI: Int
        abstract val outputJ: Int
        abstract val outputK: Int

        protected abstract fun IOScope.expect(input: Batch<IOType.D2>, context: Context): Batch<IOType.D3>

        protected abstract fun IOScope.train(
            input: Batch<IOType.D2>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
        ): Batch<IOType.D2>

        final override fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType> =
            expect(input = input as Batch<IOType.D2>, context = context)

        final override fun IOScope._train(
            input: Batch<IOType>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
        ): Batch<IOType> = train(
            input = input as Batch<IOType.D2>,
            context = context,
            calcDelta = calcDelta as IOScope.(Batch<IOType.D3>) -> Batch<IOType.D3>,
        )
    }

    @Serializable
    abstract class D3ToD1 : Reshape {
        abstract val inputI: Int
        abstract val inputJ: Int
        abstract val inputK: Int

        abstract val outputI: Int

        protected abstract fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D1>

        protected abstract fun IOScope.train(
            input: Batch<IOType.D3>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
        ): Batch<IOType.D3>

        final override fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType> =
            expect(input = input as Batch<IOType.D3>, context = context)

        final override fun IOScope._train(
            input: Batch<IOType>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
        ): Batch<IOType> = train(
            input = input as Batch<IOType.D3>,
            context = context,
            calcDelta = calcDelta as IOScope.(Batch<IOType.D1>) -> Batch<IOType.D1>,
        )
    }

    @Serializable
    abstract class D3ToD2 : Reshape {
        abstract val inputI: Int
        abstract val inputJ: Int
        abstract val inputK: Int

        abstract val outputI: Int
        abstract val outputJ: Int

        protected abstract fun IOScope.expect(input: Batch<IOType.D3>, context: Context): Batch<IOType.D2>

        protected abstract fun IOScope.train(
            input: Batch<IOType.D3>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
        ): Batch<IOType.D3>

        final override fun IOScope._expect(input: Batch<IOType>, context: Context): Batch<IOType> =
            expect(input = input as Batch<IOType.D3>, context = context)

        final override fun IOScope._train(
            input: Batch<IOType>,
            context: Context,
            calcDelta: IOScope.(Batch<IOType>) -> Batch<IOType>,
        ): Batch<IOType> = train(
            input = input as Batch<IOType.D3>,
            context = context,
            calcDelta = calcDelta as IOScope.(Batch<IOType.D2>) -> Batch<IOType.D2>,
        )
    }
}
