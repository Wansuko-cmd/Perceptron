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
    hostTarget.binaries.all {
        if (hostOs.isWindows || hostOs.isLinux) {
            val path = System.getenv("OPENBLAS_HOME")
            linkerOpts("$path")
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(projects.buffer.base)
                api(projects.buffer.cpu)
                api(projects.buffer.gpu)
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
        }
    }
}
