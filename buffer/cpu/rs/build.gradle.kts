val sourceDir = projectDir
val targetDir = projectDir.resolve("target")

/**
 * JVM
 **/
val cargoJvmBuild by tasks.registering(Exec::class) {
    group = "build"
    description = "JVM環境向けにビルド"
    workingDir = sourceDir
    commandLine = listOf("cargo", "build", "--release", "--features", "jni")
}

/**
 * Native
 **/
val cargoNativeBuild by tasks.registering(Exec::class) {
    group = "build"
    description = "Native環境向けにビルド"
    workingDir = sourceDir
    commandLine = listOf("cargo", "build", "--release", "--features", "native")
}

tasks.register<Delete>("clean") {
    group = "build"
    description = "ビルド成果物の削除"
    val target = fileTree(targetDir) {

    }
    delete(target)
}
