@file:OptIn(ExperimentalKotlinGradlePluginApi::class)

import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi

plugins {
    kotlin("multiplatform")
    alias(libs.plugins.serialization)
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm {
        mainRun { mainClass = "MainKt" }
        testRuns.named("test") {
            executionTask.configure {
                minHeapSize = "256M"
                maxHeapSize = "${1024 * 12}M"
                jvmArgs("-XX:MaxMetaspaceSize=1024M")
            }
        }
    }

    val hostOs = System.getProperty("os.name")
    val hostArch = System.getProperty("os.arch")
    val hostTarget = when {
        hostOs == "Mac OS X" && hostArch == "x86_64" -> macosX64()
        hostOs == "Mac OS X" && hostArch == "aarch64" -> macosArm64()
        hostOs == "Linux" && hostArch == "x86_64" -> linuxX64()
        hostOs == "Linux" && hostArch == "aarch64" -> linuxArm64()
        hostOs.startsWith("Windows") -> mingwX64()
        else -> throw GradleException("$hostOs:$hostArch is not supported in Kotlin/Native.")
    }
    hostTarget.binaries {
        executable {
            entryPoint = "main"
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(projects.network)

                implementation(libs.serialization)
                implementation(libs.okio)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
    }
}
