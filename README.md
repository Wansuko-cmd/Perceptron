# knist

## 概要

Kotlin製のニューラルネットワークライブラリです

元々はKotlinでMNIST識別をしたくて作ったためknistとしています

## コード例

1. モデルを定義

```kotlin
val network: Network<List<Float>, Int> = NetworkBuilder
    .inputPx(
        x = 28,
        y = 28,
        optimizer = AdamW(scheduler = Scheduler.Fix(0.001f)),
        initializer = He(seed = seed),
    )
    .reshapeToD1()
    .layerNorm()
    .affine(neuron = 256).bias().reLU()
    .affine(neuron = 128).bias().reLU()
    .affine(neuron = 10)
    .softmaxWithLoss(converter = { LabelConverter(inputSize) })
```

2. `train`関数を用いて学習

```kotlin
repeat(epoch) { epoch ->
    println("epoch: $epoch")
    train.chunked(256).forEach { data ->
        network.train(
            input = data.map { it.pixels },
            label = data.map { it.label },
        )
    }
}
```

3. `expect`関数を用いて推測

```kotlin
val expect = network.expect(input = data.pixels)
```

sampleにてMNIST(3層NN)とTinyStories(Transformer)の例を載せています

- [MNIST](sample/src/main/kotlin/dataset/mnist/MnistUtils.kt)
- [TinyStories](sample/src/main/kotlin/dataset/storeis/TinyStoriesUtils.kt)

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

## 試したい場合

[Jitpack](https://jitpack.io/#Wansuko-cmd/knist/)より導入可能です

※ 現在開発中のため、予告なく破壊的変更が加わる可能性が非常に高いです。あらかじめご了承ください
