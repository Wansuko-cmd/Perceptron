# knist

## 概要

Kotlin製のニューラルネットワークライブラリです

元々はKotlinでMNIST識別をしたくて作ったためknistとしています

※ 現在開発中のため、予告なく破壊的変更が加わる可能性が非常に高いです。あらかじめご了承ください。

## ドキュメント

- [docs/llms.txt](docs/llms.txt) : ドキュメントの索引
- [docs/llms-usage.txt](docs/llms-usage.txt) : 利用者向けの完全リファレンス（導入・モデル構築・学習・推論・保存）
- [docs/llms-custom.txt](docs/llms-custom.txt) : 拡張実装者向けリファレンス（カスタム層・出力層・オプティマイザ等の自作）

> **AIエージェント（Claude Code / Copilot 等）にこのライブラリを使わせる場合**は、まず `docs/llms-usage.txt` を読み込ませてください。カスタム層を実装させる場合のみ `docs/llms-custom.txt` も追加で読み込ませてください。

## コード例

1. モデルを定義

```kotlin
val network: Network.Src1.Sink1<List<List<Float>>, List<Int>> = Network.create(
    port = port(PixelConverter(28, 28)),
    optimizer = AdamW(Scheduler.Fix(0.001f)),
    initializer = He(seed = seed),
) { input ->
    input.reshapeToD1()
        .layerNorm()
        .affine(neuron = 256).bias().reLU()
        .affine(neuron = 128).bias().reLU()
        .affine(neuron = 10)
        .softmaxWithLoss(converter = { LabelConverter(inputI) })
}
```

2. `train`関数を用いて学習（`train` / `loss` / `expect` は suspend 関数のため、コルーチンスコープ内から呼びます）

```kotlin
runBlocking {
    repeat(epoch) { epoch ->
        println("epoch: $epoch")
        train.chunked(256).forEach { data ->
            network.train(
                input = data.map { it.pixels },
                label = data.map { it.label },
            )
        }
    }
}
```

3. `expect`関数を用いて推測（入力・出力ともにバッチ単位）

```kotlin
val expected: List<Int> = network.expect(input = data.map { it.pixels })
```

sampleにてMNIST(3層NN)とTinyStories(Transformer)の例を載せています

- [MNIST](sample/src/commonMain/kotlin/dataset/mnist/MnistUtils.kt)
- [TinyStories](sample/src/commonMain/kotlin/dataset/stories/TinyStoriesUtils.kt)

## 対応プラットフォーム

- **JVM / Android**: JitPack経由で導入できます
- **Kotlin/Native**（macOS / Linux / Windows）: JitPackでは配布していません。リポジトリをクローンして `publishToMavenLocal` することで利用できます
- JS / wasm は非対応です

## ライブラリ構成

### Network

ニューラルネットワーク関連の処理を定義

- 処理層
- 変換層
- 重み初期化処理
- 最適化処理
- JSONへのシリアライズ・デシリアライズ
- etc...

### IOType

行列演算の処理を定義

主に形状管理を行う

### Buffer

1次元配列でのデータ保持、および計算処理を定義

## バックエンド

演算の実行バックエンドを切り替えられます。デフォルトはCPU（Rustネイティブ実装）です。

- **CPU**: Rustネイティブ実装（デフォルト）
- **GPU**: WebGPU（wgpu）実装。稀に不安定になることがあります
- **KotlinBackend**: 純Kotlin実装（フォールバック用）

```kotlin
import com.wsr.knist.Backend
import com.wsr.knist.base.KotlinBackend
import com.wsr.knist.gpu.loadGPUBackend

Backend.set(loadGPUBackend(KotlinBackend))
```

詳細は[docs/llms-usage.txt](docs/llms-usage.txt)を参照してください。

## 導入方法

[JitPack](https://jitpack.io/#Wansuko-cmd/knist/) から導入できます。

**settings.gradle.kts** にリポジトリを追加:

```kotlin
dependencyResolutionManagement {
    repositories {
        maven { url = uri("https://jitpack.io") }
    }
}
```

**build.gradle.kts** に依存を追加:

```kotlin
dependencies {
    implementation("com.github.Wansuko-cmd.knist:network:<version>")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:<version>")  // train / loss / expect が suspend 関数のため必須
}
```

> **注意**: `com.github.Wansuko-cmd:knist` ではなく `com.github.Wansuko-cmd.knist:network` を使ってください。
