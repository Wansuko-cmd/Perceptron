plugins {
    kotlin("jvm")
    alias(libs.plugins.serialization)
    application
}

dependencies {
    implementation(projects.network)

    implementation(libs.serialization)
    implementation(libs.okio)

    testImplementation(kotlin("test"))
}

application {
    mainClass = "MainKt"
}

tasks.test {
    minHeapSize = "256M"
    maxHeapSize = "${1024 * 12}M"
    jvmArgs = listOf("-XX:MaxMetaspaceSize=1024M")
    useJUnit()
}
