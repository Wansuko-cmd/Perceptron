package dataset.stories

import com.wsr.network.Network
import com.wsr.network.NetworkBuilder
import com.wsr.network.converter.word.WordD2
import com.wsr.network.converter.word.wordsD1
import com.wsr.network.initializer.Xavier
import com.wsr.network.optimizer.Scheduler
import com.wsr.network.optimizer.adam.AdamW
import com.wsr.network.output.softmax.softmaxWithLoss
import com.wsr.network.process.compute.affine.affine
import com.wsr.network.process.compute.attention.attention
import com.wsr.network.process.compute.bias.d2.bias
import com.wsr.network.process.compute.dropout.dropout
import com.wsr.network.process.compute.function.relu.swish
import com.wsr.network.process.compute.norm.layer.d2.layerNorm
import com.wsr.network.process.compute.position.roPE
import com.wsr.network.process.compute.scale.d2.scale
import com.wsr.network.process.compute.skip.skip
import com.wsr.network.process.reshape.token.tokenEmbedding
import java.io.File
import kotlin.random.Random
import kotlin.random.nextInt

private const val TRAIN_PATH = "src/main/resources/stories/TinyStories-train.txt"

private const val VOCAB_SIZE = 3000
private const val EMBEDDING_DIM = 256
const val MAX_LENGTH = 128
private const val NUM_LAYERS = 4
private const val NUM_HEADS = 4
private const val FFN_DIM = EMBEDDING_DIM * 4

private const val BATCH_SIZE = 64
private const val NUM_OF_STORIES = 1000

private const val PAD_INDEX = 0
private const val UNK_INDEX = 1

fun createTinyStoriesModel(seed: Int? = null): Network<List<String>, List<String>> {
    println("単語リスト生成開始")
    val words: List<String> = createWordList(TRAIN_PATH, VOCAB_SIZE)

    // ニューラルネットワークを構築
    val network = NetworkBuilder.wordsD1(
        maxLength = MAX_LENGTH,
        words = words,
        unknownIndex = UNK_INDEX,
        paddingIndex = PAD_INDEX,
        optimizer = AdamW(
            scheduler = Scheduler.CosineAnnealing(
                minRate = 0.0005f,
                maxRate = 0.001f,
                stepSize = NUM_OF_STORIES,
                warmUp = 200,
                initialRate = 0f,
            ),
        ),
        initializer = Xavier(seed = seed),
    )
        .tokenEmbedding(
            vocabSize = words.size,
            tokenSize = EMBEDDING_DIM,
        )
        .roPE()
        .repeat(NUM_LAYERS) {
            this
                .skip {
                    this
                        .layerNorm(axis = 1).scale(axis = 1).bias(axis = 1)
                        .attention(numOfHeads = NUM_HEADS, maskValue = PAD_INDEX, isCausal = true)
                        .dropout(ratio = 0.9f)
                }
                .skip {
                    this
                        .layerNorm(axis = 1).scale(axis = 1).bias(axis = 1)
                        .affine(neuron = FFN_DIM).bias(axis = 1).swish()
                        .affine(neuron = EMBEDDING_DIM).bias(axis = 1)
                        .dropout(ratio = 0.9f)
                }
        }
        .layerNorm(axis = 1).scale(axis = 1).bias(axis = 1)
        .affine(neuron = words.size)
        .softmaxWithLoss(
            converter = {
                WordD2(
                    words = words,
                    length = MAX_LENGTH,
                    unknownIndex = UNK_INDEX,
                )
            },
            maskValue = PAD_INDEX,
        )

    println("学習開始")
    File(TRAIN_PATH)
        .useLines { lines ->
            lines
                .generateStories()
                .flatMap { tokenize(it).toData() }
                // バッチサイズ
                .chunked(BATCH_SIZE)
                // 学習バッチ数
                .take(NUM_OF_STORIES)
                .forEachIndexed { lineIndex, trainData ->
                    val inputs = trainData.map { it.first }
                    val labels = trainData.map { it.second }
                    val random = Random.nextInt(inputs.indices)
                    println(
                        """
                            ---------------------------
                            train line: $lineIndex
                            入力例: ${inputs[random]}
                            ラベル: ${labels[random]}
                        """.trimIndent(),
                    )

                    val loss = network.train(inputs, labels)
                    println(
                        """
                            loss: $loss
                            ---------------------------
                        """.trimIndent(),
                    )
                }
        }

    println("出力確認")
    val story = network.createStories(beginning = "One day, a sheep named Bob was very happy.", maxLength = 300)
    println(story)

    return network
}

private fun Network<List<String>, List<String>>.createStories(beginning: String, maxLength: Int): String {
    val text = tokenize(beginning).take(MAX_LENGTH).toMutableList()
    repeat(maxLength) {
        val input = text.takeLast(MAX_LENGTH)
        if (input.last() == "<EOS>") return@repeat
        val expect = this.expect(input)[input.lastIndex]
        text.add(expect)
    }
    return text.joinToString(" ")
        .replace(Regex(" ([!?.]) "), ".\n")
        .replace(Regex(" (,)"), "$1")
}
