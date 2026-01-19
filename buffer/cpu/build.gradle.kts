plugins {
    kotlin("multiplatform")
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm()

    val hostOs = System.getProperty("os.name")
    val hostArch = System.getProperty("os.arch")
    val hostTarget = when {
        hostOs == "Mac OS X" && hostArch == "x86_64" -> macosX64()
        hostOs == "Mac OS X" && hostArch == "aarch64" -> macosArm64()
        hostOs == "Linux" && hostArch == "x86_64" -> linuxX64()
        hostOs == "Linux" && hostArch == "aarch64" -> linuxArm64()
        hostOs.startsWith("Windows") -> mingwX64()
        else -> throw GradleException("Host OS is not supported in Kotlin/Native.")
    }
    hostTarget.compilations.getByName("main") {
        cinterops {
            val lib by creating {
                val headersDir = "$projectDir/cpp/src/cpu"
                headers(
                    "$headersDir/collection_fun.h",
                    "$headersDir/collection_fun.h",
                    "$headersDir/mat_mul_fun.h",
                    "$headersDir/math_fun.h",
                    "$headersDir/operation_fun.h",
                    "$headersDir/transpose_fun.h",
                )
                defFile(project.file("src/nativeMain/lib.def"))
                extraOpts("-libraryPath", "$projectDir/src/nativeMain/resources/cpu/${target.name}")
                extraOpts("-staticLibrary", if (hostOs.startsWith("Windows")) "cpu.lib" else "libcpu.a")
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(projects.buffer.base)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
    }

    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }
}

publishing {
    publications {
        create<MavenPublication>(project.name) {
            groupId = libs.versions.lib.group.id.get()
            artifactId = "knist"
            version = libs.versions.lib.version.get()
        }
    }
}
