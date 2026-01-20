plugins {
    kotlin("multiplatform")
    alias(libs.plugins.serialization)
}

kotlin {
    applyDefaultHierarchyTemplate()
    jvm()

    val hostOs = System.getProperty("os.name")
    val hostArch = System.getProperty("os.arch")
    when {
        hostOs == "Mac OS X" && hostArch == "x86_64" -> macosX64()
        hostOs == "Mac OS X" && hostArch == "aarch64" -> macosArm64()
        hostOs == "Linux" && hostArch == "x86_64" -> linuxX64()
        hostOs == "Linux" && hostArch == "aarch64" -> linuxArm64()
        hostOs.startsWith("Windows") -> mingwX64()
        else -> throw GradleException("$hostOs:$hostArch is not supported in Kotlin/Native.")
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
