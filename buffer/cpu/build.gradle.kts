import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform

plugins {
    kotlin("multiplatform")
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm()

    val hostOs = DefaultNativePlatform.getCurrentOperatingSystem()
    val hostArch = DefaultNativePlatform.getCurrentArchitecture()
    val hostTarget = when {
        hostOs.isMacOsX && hostArch.isAmd64 -> macosX64()
        hostOs.isMacOsX && hostArch.isArm64 -> macosArm64()
        hostOs.isLinux && hostArch.isAmd64 -> linuxX64()
        hostOs.isLinux && hostArch.isArm64 -> linuxArm64()
        hostOs.isWindows && hostArch.isAmd64 -> mingwX64()
        else -> throw GradleException("$hostOs:$hostArch is not supported in Kotlin/Native.")
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
                if (hostOs.isWindows || hostOs.isLinux) {
                    val path = System.getenv("OPENBLAS_HOME")
                    linkerOpts("-L/usr/lib/x86_64-linux-gnu", "-lopenblas")
                }
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
