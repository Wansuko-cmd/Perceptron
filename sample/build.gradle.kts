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

    listOf(
        mingwX64(),
        linuxX64(),
        linuxArm64(),
        macosX64(),
        macosArm64(),
        iosX64(),
        iosArm64(),
        iosSimulatorArm64(),
    ).forEach { arch ->
        arch.binaries {
            executable {
                entryPoint = "main"
            }
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
