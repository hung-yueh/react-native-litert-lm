package com.margelo.nitro.dev.litert.litertlm.product

import android.app.ActivityManager
import android.content.Context
import android.os.Build

import java.io.File

/**
 * Hardware facts that decide the starting configuration before (or instead
 * of) autotuning. Everything here is cheap to read.
 */
data class DeviceProfile(
  val totalRamBytes: Long,
  val lowRamDevice: Boolean,
  /** Cores whose max clock is at least 75% of the fastest core's (excludes the little cluster on 1+3+4 and 4+4 layouts with a big gap). */
  val performanceCores: Int,
  val allCores: Int,
  val socModel: String,
) {
  val ramGb get() = totalRamBytes / 1e9

  /** Products to keep prefilled: each costs one full KV cache. */
  val suggestedCachedProducts: Int
    get() = when {
      lowRamDevice || ramGb < 5 -> 1
      ramGb < 7 -> 2
      ramGb < 10 -> 3
      else -> 4
    }

  /**
   * XNNPACK runs best on the big cores only. Google's own benchmarks for this
   * model use 4 threads; more than 4 rarely helps a memory-bound decode.
   */
  val suggestedCpuThreads get() = performanceCores.coerceIn(1, 4)

  /** CPU thread counts worth measuring on this core layout. */
  val cpuThreadCandidates: List<Int>
    get() = listOf(suggestedCpuThreads, (suggestedCpuThreads - 1).coerceAtLeast(2), allCores.coerceAtMost(6))
      .distinct().filter { it in 1..allCores }

  /**
   * Starting options. [contextTokens] should come from
   * [ProductRuntime.contextFor]; it's capped by what RAM can hold for the
   * cached products, dropping cached products before shrinking the context.
   */
  fun baseOptions(cacheDir: File, contextTokens: Int): ProductRuntime.Options {
    var cached = suggestedCachedProducts
    while (cached > 1 && ProductRuntime.maxContextForRam(totalRamBytes, cached) < contextTokens) cached--
    return ProductRuntime.Options(
      maxNumTokens = minOf(contextTokens, ProductRuntime.maxContextForRam(totalRamBytes, cached)),
      maxCachedProducts = cached,
      cpuThreads = suggestedCpuThreads,
      cacheDir = cacheDir,
    )
  }

  companion object {
    fun read(context: Context): DeviceProfile {
      val am = context.getSystemService(ActivityManager::class.java)
      val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
      val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { i ->
        runCatching {
          File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
        }.getOrNull()
      }
      val fastest = freqs.maxOrNull()
      val perf = if (fastest == null) 4 else freqs.count { it >= fastest * 0.75 }
      return DeviceProfile(
        totalRamBytes = mem.totalMem,
        lowRamDevice = am.isLowRamDevice,
        performanceCores = perf,
        allCores = Runtime.getRuntime().availableProcessors(),
        socModel = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE,
      )
    }
  }
}
