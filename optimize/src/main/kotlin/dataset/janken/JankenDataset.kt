package dataset.janken

import java.io.File
import kotlin.math.absoluteValue

data class JankenDataset(
    val data: List<Double>,
    val label: Int,
) {
    companion object {
        private const val PATH = "janken.txt"
        private fun create(data: List<Double>, label: Int) =
            JankenDataset(
                data = data.map {
                    when {
                        it.absoluteValue > 1000 -> it / 1000.0
                        it.absoluteValue > 100 -> it / 100.0
                        it.absoluteValue > 10 -> it / 10.0
                        it.absoluteValue < 0.1 -> it * 10.0
                        else -> it
                    }
                },
                label = label,
            )

        fun load(): List<JankenDataset> {
            val file = File(PATH)
            return file.readText()
                .split('\n')
                .filter { it.isNotBlank() }
                .map { it.split('|').filter { it.isNotBlank() } }
                .map { (data, label) -> create(data.split(',').filter { it.isNotBlank() }.map { it.toDouble() }, label.toInt()) }
        }
    }
}
