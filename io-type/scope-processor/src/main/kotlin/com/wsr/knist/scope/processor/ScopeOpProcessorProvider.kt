package com.wsr.knist.scope.processor

import com.google.devtools.ksp.processing.SymbolProcessor
import com.google.devtools.ksp.processing.SymbolProcessorEnvironment
import com.google.devtools.ksp.processing.SymbolProcessorProvider

class ScopeOpProcessorProvider : SymbolProcessorProvider {
    override fun create(environment: SymbolProcessorEnvironment): SymbolProcessor = ScopeOpProcessor(
        codeGenerator = environment.codeGenerator,
        logger = environment.logger,
    )
}
