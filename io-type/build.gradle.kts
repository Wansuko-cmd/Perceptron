plugins {
    kotlin("multiplatform")
    alias(libs.plugins.serialization)
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm()

    mingwX64()

    linuxX64()
    linuxArm64()

    macosX64()
    macosArm64()

    iosX64()
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(projects.buffer)

                implementation(libs.coroutine)
                implementation(libs.serialization)
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
            groupId =
                libs.versions.lib.group.id
                    .get()
            artifactId = "knist"
            version =
                libs.versions.lib.version
                    .get()
        }
    }
}
