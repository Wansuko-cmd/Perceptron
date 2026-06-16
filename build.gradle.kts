plugins {
    alias(libs.plugins.ktlint) apply false
    id("maven-publish")
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.kmp.library) apply false
}

subprojects {
    apply(plugin = "maven-publish")

    // ktlintプラグインはKotlinプラグインが適用されているプロジェクトのみに適用
    pluginManager.withPlugin("org.jetbrains.kotlin.jvm") {
        apply(plugin = libs.plugins.ktlint.get().pluginId)
        configure<org.jlleitschuh.gradle.ktlint.KtlintExtension> {
            version.set(libs.versions.ktlint.lib.get())
        }
    }
    pluginManager.withPlugin("org.jetbrains.kotlin.multiplatform") {
        apply(plugin = libs.plugins.ktlint.get().pluginId)
        configure<org.jlleitschuh.gradle.ktlint.KtlintExtension> {
            version.set(libs.versions.ktlint.lib.get())
        }
    }
}

tasks.register<Delete>(name = "clean") {
    delete(rootProject.layout.buildDirectory)
}
