plugins {
    kotlin("multiplatform")
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm()

    listOf(
        mingwX64(),
        linuxX64(),
        linuxArm64(),
        macosX64(),
        macosArm64(),
    ).forEach { arch ->
        arch.compilations.getByName("main") {
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
                }
            }
        }
    }

    iosX64()
    iosArm64()
    iosSimulatorArm64()

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
