package com.wsr.knist.scope.processor

import com.google.devtools.ksp.processing.CodeGenerator
import com.google.devtools.ksp.processing.Dependencies
import com.google.devtools.ksp.processing.KSPLogger
import com.google.devtools.ksp.processing.Resolver
import com.google.devtools.ksp.processing.SymbolProcessor
import com.google.devtools.ksp.symbol.KSAnnotated
import com.google.devtools.ksp.symbol.KSFunctionDeclaration
import com.google.devtools.ksp.symbol.KSType
import com.google.devtools.ksp.symbol.Modifier
import com.google.devtools.ksp.symbol.Variance
import com.google.devtools.ksp.validate

class ScopeOpProcessor(private val codeGenerator: CodeGenerator, private val logger: KSPLogger) : SymbolProcessor {

    private val collectedFunctions = mutableListOf<KSFunctionDeclaration>()
    private var generated = false

    override fun process(resolver: Resolver): List<KSAnnotated> {
        val (valid, deferred) = resolver
            .getSymbolsWithAnnotation("com.wsr.knist.scope.ScopeOp")
            .filterIsInstance<KSFunctionDeclaration>()
            .toList()
            .partition { it.validate() }

        collectedFunctions.addAll(valid)

        // Deferred symbols will be re-presented in the next round; wait until all are resolved.
        if (deferred.isNotEmpty()) return deferred

        if (!generated && collectedFunctions.isNotEmpty()) {
            generated = true
            val file = codeGenerator.createNewFile(
                dependencies = Dependencies(
                    aggregating = true,
                    *collectedFunctions.mapNotNull { it.containingFile }.toTypedArray(),
                ),
                packageName = "com.wsr.knist.core",
                fileName = "IOScope",
            )
            file.bufferedWriter().use { writer ->
                writer.write(buildIOScopeContent(collectedFunctions))
            }
        }

        return emptyList()
    }

    private fun buildIOScopeContent(functions: List<KSFunctionDeclaration>): String {
        val indexed = functions.mapIndexedNotNull { i, func ->
            generateMember(func, i)?.let { i to it }
        }

        val sb = StringBuilder()
        sb.appendLine("package com.wsr.knist.core")
        sb.appendLine()
        sb.appendLine("import com.wsr.knist.base.BufferScope")
        // Import alias per function — avoids calling extension functions via FQN (not valid Kotlin)
        // and avoids infinite recursion from IOScope member shadowing top-level extension
        indexed.forEach { (i, data) ->
            sb.appendLine("import ${data.fqn} as _impl$i")
        }
        sb.appendLine()
        sb.appendLine("@Suppress(\"NOTHING_TO_INLINE\")")
        sb.appendLine("class IOScope(val bufferScope: BufferScope) {")
        sb.appendLine()

        indexed.forEach { (_, data) ->
            sb.appendLine(data.member)
        }

        sb.appendLine("}")
        return sb.toString()
    }

    private data class MemberData(val fqn: String, val member: String)

    private fun generateMember(func: KSFunctionDeclaration, index: Int): MemberData? {
        val receiver = func.extensionReceiver ?: run {
            logger.warn("@ScopeOp on ${func.qualifiedName?.asString()} has no extension receiver, skipping")
            return null
        }

        val receiverTypeName = receiver.resolve().toFqnString()
        val returnType = func.returnType?.resolve() ?: return null
        val returnTypeName = returnType.toFqnString()

        val returnDeclFqn = returnType.declaration.qualifiedName?.asString() ?: return null
        if (!returnDeclFqn.startsWith("com.wsr.knist.core.IOType") &&
            !returnDeclFqn.startsWith("com.wsr.knist.batch.Batch")
        ) {
            logger.warn("@ScopeOp on ${func.qualifiedName?.asString()} returns $returnDeclFqn, skipping")
            return null
        }

        val funcName = func.simpleName.asString()
        val fqn = func.qualifiedName?.asString() ?: return null
        val modifiers = buildModifiers(func)
        val params = func.parameters

        val paramsDecl = params.joinToString(", ") { p ->
            val name = p.name!!.asString()
            val type = p.type.resolve().toFqnString()
            "$name: $type"
        }

        val paramCall = params.joinToString(", ") { p ->
            p.name!!.asString()
        }

        val member = buildString {
            // @JvmName needed to avoid platform declaration clashes when Batch<T> generics are erased
            appendLine("    @kotlin.jvm.JvmName(\"_ioScopeImpl$index\")")
            appendLine("    $modifiers fun $receiverTypeName.$funcName($paramsDecl): $returnTypeName {")
            appendLine("        val result = this._impl$index($paramCall)")
            appendLine("        bufferScope.register(result.value)")
            appendLine("        return result")
            append("    }")
        }
        return MemberData(fqn = fqn, member = member)
    }

    private fun buildModifiers(func: KSFunctionDeclaration): String {
        val mods = mutableListOf("inline")
        if (Modifier.OPERATOR in func.modifiers) mods.add("operator")
        if (Modifier.INFIX in func.modifiers) mods.add("infix")
        if (Modifier.SUSPEND in func.modifiers) mods.add("suspend")
        return mods.joinToString(" ")
    }

    private fun KSType.toFqnString(): String {
        val fqn = declaration.qualifiedName?.asString()
            ?: declaration.simpleName.asString()
        return if (arguments.isEmpty()) {
            fqn + if (isMarkedNullable) "?" else ""
        } else {
            val argStr = arguments.joinToString(", ") { arg ->
                when (arg.variance) {
                    Variance.STAR -> "*"
                    else -> arg.type?.resolve()?.toFqnString() ?: "*"
                }
            }
            "$fqn<$argStr>" + if (isMarkedNullable) "?" else ""
        }
    }
}
