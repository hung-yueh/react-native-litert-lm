package com.margelo.nitro.dev.litert.litertlm

import java.util.TreeSet
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Engine teardowns that close(), unload() or a memory release started, on
 * any instance, and that haven't finished. A load waits for the ones started
 * before it was called: models are gigabytes, and one loaded right after
 * another was closed shouldn't sit beside it in memory. A load never waits
 * for a teardown started after it was called, so the waits can't form a cycle.
 */
internal object Teardowns {
    private val lock = ReentrantLock()
    private val finished = lock.newCondition()
    private var issued = 0L
    private val pending = TreeSet<Long>()

    fun begin(): Long = lock.withLock {
        issued += 1
        pending += issued
        issued
    }

    fun end(ticket: Long) = lock.withLock {
        pending -= ticket
        finished.signalAll()
    }

    /** The latest teardown so far — what a load called now waits for. */
    fun mark(): Long = lock.withLock { issued }

    /** Block until every teardown up to [ticket] has finished. Never call it on the JS thread. */
    fun await(ticket: Long) = lock.withLock {
        while (pending.isNotEmpty() && pending.first() <= ticket) finished.await()
    }
}
