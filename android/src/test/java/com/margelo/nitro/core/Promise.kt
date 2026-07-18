package com.margelo.nitro.core

import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip

@Keep
@DoNotStrip
class Promise<T> {
    companion object {
        @JvmStatic
        fun <T> parallel(block: () -> T): Promise<T> {
            val promise = Promise<T>()
            try {
                val result = block()
                promise.resolve(result)
            } catch (e: Throwable) {
                promise.reject(e)
            }
            return promise
        }
    }

    var result: T? = null
        private set
    var error: Throwable? = null
        private set
    var isCompleted = false
        private set
    private val callbacks = mutableListOf<(T?, Throwable?) -> Unit>()

    fun resolve(value: T) {
        synchronized(this) {
            result = value
            isCompleted = true
            callbacks.forEach { it(value, null) }
        }
    }

    fun reject(exception: Throwable) {
        synchronized(this) {
            error = exception
            isCompleted = true
            callbacks.forEach { it(null, exception) }
        }
    }

    // Mirror the real Nitro Promise continuation API (then/catch return `this`
    // for chaining) so production code compiled against react-native-nitro-modules
    // links against this stub under Robolectric.
    fun then(listener: (result: T) -> Unit): Promise<T> {
        synchronized(this) {
            if (isCompleted) {
                if (error == null) {
                    @Suppress("UNCHECKED_CAST")
                    listener(result as T)
                }
            } else {
                callbacks.add { value, err ->
                    if (err == null) {
                        @Suppress("UNCHECKED_CAST")
                        listener(value as T)
                    }
                }
            }
        }
        return this
    }

    fun catch(listener: (throwable: Throwable) -> Unit): Promise<T> {
        synchronized(this) {
            if (isCompleted) {
                error?.let(listener)
            } else {
                callbacks.add { _, err -> if (err != null) listener(err) }
            }
        }
        return this
    }
}
