package com.wsr.knist.scope

@Target(AnnotationTarget.VALUE_PARAMETER)
@Retention(AnnotationRetention.SOURCE)
annotation class ScopeOpDefault(val value: String)
