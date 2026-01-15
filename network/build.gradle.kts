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
