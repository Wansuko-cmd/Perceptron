package dataset.mnist

import dataset.resource
import okio.BufferedSource
import okio.FileSystem
import okio.GzipSource
import okio.buffer

data class MnistDataset(val pixels: List<Float>, val label: Int) {
    companion object {
        fun read(imagePath: String, labelPath: String): List<MnistDataset> = try {
            gzipBuffer(imagePath).use { imgIn ->
                gzipBuffer(labelPath).use { lblIn ->
                    imgIn.skip(4)
                    lblIn.skip(4)

                    val numOfImages = imgIn.readInt()
                    val height = imgIn.readInt()
                    val width = imgIn.readInt()
                    val numOfLabels = lblIn.readInt()

                    val numOfPixel = height * width

                    List(numOfImages) {
                        MnistDataset(
                            pixels = List(numOfPixel) { imgIn.readUInt().toFloat() },
                            label = lblIn.readUInt(),
                        )
                    }
                }
            }
        } catch (e: Exception) {
            println("resource/mnistにMNISTデータセットを入れてください")
            throw e
        }

        private fun gzipBuffer(path: String): BufferedSource {
            val source = FileSystem.SYSTEM.resource(path)
            return GzipSource(source).buffer()
        }

        private fun BufferedSource.readUInt() = readByte().toInt() and 0xFF
    }
}
