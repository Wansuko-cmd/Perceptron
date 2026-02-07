import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform

val sourceDir = projectDir
val targetDir = projectDir.resolve("target")

/**
 * JVM
 **/
val cargoJvmBuild by tasks.registering(Exec::class) {
    group = "build"
    description = "JVM環境向けにビルド"
    workingDir = sourceDir

    val hostOs = DefaultNativePlatform.getCurrentOperatingSystem()
    val hostArch = DefaultNativePlatform.getCurrentArchitecture()
    val target = when {
        hostArch.isAmd64 -> "x86-64-v3"
        hostOs.isMacOsX && hostArch.isArm64 -> "apple-m1"
        hostArch.isArm64 -> "generic -C target-feature=+neon"
        else -> "native"
    }
    environment("RUSTFLAGS", "-C target-cpu=$target")

    commandLine = listOf("cargo", "build", "--release")
}

tasks.register<Delete>("clean") {
    group = "build"
    description = "ビルド成果物の削除"
    val target = fileTree(targetDir) {

    }
    delete(target)
}
