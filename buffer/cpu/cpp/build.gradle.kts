// パスの定義
val cmakeSourceDir = projectDir
val cmakeBuildDir = projectDir.resolve("build")        // ビルド用一時ディレクトリ

// CMake 設定（cmake configure 相当）
val cmakeJvmConfigure by tasks.registering(Exec::class) {
    group = "build"
    description = "JVM環境向けにCMakeをセットアップ"
    workingDir = cmakeSourceDir
    doFirst {
        cmakeBuildDir.mkdirs()
    }
    commandLine = listOf(
        "cmake",
        "-S", cmakeSourceDir.absolutePath,
        "-B", cmakeBuildDir.absolutePath,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DINCLUDE_JNI=ON",
        "-DAS_SHARED=ON",
    )
}

// CMake ビルド（make 相当）
val cmakeJvmBuild by tasks.registering(Exec::class) {
    group = "build"
    description = "JVM環境向けにビルド"
    dependsOn(cmakeJvmConfigure)
    workingDir = cmakeSourceDir
    commandLine = listOf(
        "cmake",
        "--build", cmakeBuildDir.absolutePath,
        "--config", "Release",
    )
}

// Cleanタスク
tasks.register<Delete>("clean") {
    group = "build"
    description = "ビルド成果物の削除"
    delete(fileTree(projectDir.resolve("build")) {
        exclude("**/openblas/**")
    })
}
