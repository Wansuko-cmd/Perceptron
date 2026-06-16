plugins {
    kotlin("jvm")
}

dependencies {
    compileOnly(libs.ksp.api)
    implementation(projects.ioType.scopeAnnotation)
}
