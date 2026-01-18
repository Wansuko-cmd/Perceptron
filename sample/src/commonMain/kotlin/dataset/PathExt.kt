package dataset

import okio.FileSystem
import okio.Path.Companion.toPath

expect val resourcePath: String

fun FileSystem.resource(path: String) = source("$resourcePath$path".toPath())
