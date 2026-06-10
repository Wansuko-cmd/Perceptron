import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform

plugins {
    kotlin("multiplatform")
    alias(libs.plugins.serialization)
    alias(libs.plugins.android.kmp.library)
    alias(libs.plugins.ksp)
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm()

    android {
        compileSdk = libs.versions.android.compile.sdk.get().toInt()
        minSdk = libs.versions.android.min.sdk.get().toInt()
        namespace = "com.wsr.knist.iotype"
        withHostTestBuilder { sourceSetTreeName = "test" }
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
            kotlin.srcDir("build/generated/ksp/metadata/commonMain/kotlin")
            dependencies {
                api(projects.buffer)

                implementation(libs.serialization)
                implementation(projects.ioType.scopeAnnotation)
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

dependencies {
    add("kspCommonMainMetadata", projects.ioType.scopeProcessor)
}

afterEvaluate {
    publishing {
        publications {
            withType<MavenPublication>().configureEach {
                groupId = libs.versions.lib.group.id.get()
                version = libs.versions.lib.version.get()
            }
        }
    }
}
