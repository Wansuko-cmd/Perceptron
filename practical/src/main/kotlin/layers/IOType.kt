package layers

sealed interface IOType {
    fun asIOType0d(): IOType0d

    data class IOType0d(val value: Array<Double>) : IOType {
        override inline fun asIOType0d(): IOType0d = this
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as IOType0d

            if (!value.contentEquals(other.value)) return false

            return true
        }

        override fun hashCode(): Int {
            return value.contentHashCode()
        }
    }

    data class IOType1d(val value: Array<Array<Double>>) : IOType {

        override fun asIOType0d(): IOType0d = IOType0d(value.flatten().toTypedArray())

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as IOType1d

            if (!value.contentDeepEquals(other.value)) return false

            return true
        }

        override fun hashCode(): Int {
            return value.contentDeepHashCode()
        }
    }
}
