import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform

val cmakeSourceDir = projectDir
val cmakeBuildDir = projectDir.resolve("build")

/**
 * JVM
 **/
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

/**
 * Native
 **/
val cmakeNativeConfigure by tasks.registering(Exec::class) {
    group = "build"
    description = "Native環境向けにCMakeをセットアップ"
    workingDir = cmakeSourceDir
    doFirst {
        cmakeBuildDir.mkdirs()
    }
    val hostOs = DefaultNativePlatform.getCurrentOperatingSystem()
    commandLine = buildList {
        add("cmake")
        addAll(listOf("-S", cmakeSourceDir.absolutePath))
        addAll(listOf("-B", cmakeBuildDir.absolutePath))
        if (hostOs.isWindows) addAll(listOf("-G", "Ninja"))
        add("-DCMAKE_BUILD_TYPE=Release")
        add("-DINCLUDE_JNI=OFF")
        add("-DAS_SHARED=OFF")
    }
}

val cmakeNativeBuild by tasks.registering(Exec::class) {
    group = "build"
    description = "Native環境向けにビルド"
    dependsOn(cmakeNativeConfigure)
    workingDir = cmakeSourceDir
    commandLine = listOf(
        "cmake",
        "--build", cmakeBuildDir.absolutePath,
        "--config", "Release",
        "--verbose",
    )
}

tasks.register<Delete>("clean") {
    group = "build"
    description = "ビルド成果物の削除"
    val target = fileTree(cmakeBuildDir) {
        exclude("**/openblas/**")
    }
    delete(target)
}
