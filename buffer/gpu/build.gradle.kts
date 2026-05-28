import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform

plugins {
    kotlin("multiplatform")
    alias(libs.plugins.android.kmp.library)
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm()

    android {
        compileSdk = 35
        minSdk = 28
        namespace = "com.wsr.gpu"
    }

    val hostOs = DefaultNativePlatform.getCurrentOperatingSystem()
    val hostArch = DefaultNativePlatform.getCurrentArchitecture()
    when {
        hostOs.isMacOsX && hostArch.isAmd64 -> macosX64()
        hostOs.isMacOsX && hostArch.isArm64 -> macosArm64()
        hostOs.isLinux && hostArch.isAmd64 -> linuxX64()
        hostOs.isLinux && hostArch.isArm64 -> linuxArm64()
        hostOs.isWindows && hostArch.isAmd64 -> mingwX64()
        else -> throw GradleException("$hostOs:$hostArch is not supported in Kotlin/Native.")
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

        val jvmAndroidMain by creating {
            dependsOn(commonMain)
        }
        jvmMain {
            dependsOn(jvmAndroidMain)
        }
        androidMain {
            dependsOn(jvmAndroidMain)
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
