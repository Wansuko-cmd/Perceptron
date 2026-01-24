import org.gradle.kotlin.dsl.assign
import org.gradle.kotlin.dsl.withType
import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform
import org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeTest

plugins {
    kotlin("multiplatform")
    alias(libs.plugins.serialization)
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
    hostTarget.binaries.all {
        if (hostOs.isWindows || hostOs.isLinux) {
            val path = System.getenv("OPENBLAS_HOME")
            linkerOpts("-L$path")
        }
    }
    tasks.withType<KotlinNativeTest>().configureEach {
        failOnNoDiscoveredTests = false
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(projects.ioType)

                implementation(libs.serialization)
                implementation(libs.serialization.okio)

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

publishing {
    publications {
        create<MavenPublication>(project.name) {
            groupId = libs.versions.lib.group.id.get()
            artifactId = "knist"
            version = libs.versions.lib.version.get()
            from(components["kotlin"])
        }
    }
}
