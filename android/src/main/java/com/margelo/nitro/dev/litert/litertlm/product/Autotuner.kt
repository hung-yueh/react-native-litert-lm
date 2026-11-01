package com.margelo.nitro.dev.litert.litertlm.product

import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.SpeculativeDecodingMode
import java.io.File
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.toList

/**
 * Finds the fastest engine setup on *this* device and remembers it.
 *
 * Mid-range phones vary too much for one default: some GPUs lose to the CPU,
 * the useful CPU thread count depends on the core layout, and whether MTP
 * pays off depends on the backend. So each candidate is measured on a
 * realistic request (cached system prompt + short user turn) and scored by
 * the latency a user would see: time-to-first-token + time to produce
 * [expectedOutputChars].
 *
 * Run once per device + model file, ideally right after the model download
 * while the user isn't waiting (it loads one engine per candidate). The
 * result is cached in [store]; later launches read it instantly.
 */
class Autotuner(
  private val modelPath: String,
  private val cacheDir: File?,
  private val store: KeyValueStore,
  /** Stable device identifier, e.g. Build.FINGERPRINT on Android. */
  private val deviceId: String,
) {
  interface KeyValueStore {
    fun get(key: String): String?
    fun put(key: String, value: String)
  }

  data class Candidate(val backend: ProductRuntime.BackendChoice, val cpuThreads: Int = 4) {
    override fun toString() =
      if (backend == ProductRuntime.BackendChoice.CPU) "CPU x$cpuThreads" else backend.name
  }

  data class Measurement(
    val candidate: Candidate,
    val mode: SpeculativeDecodingMode,
    val initSeconds: Double,
    val ttftSeconds: Double,
    val charsPerSecond: Double,
    val scoreSeconds: Double,
  )

  data class Choice(
    val backend: ProductRuntime.BackendChoice,
    val cpuThreads: Int,
    val speculativeDecoding: SpeculativeDecodingMode,
    val expectedLatencySeconds: Double,
  ) {
    /** Applies this choice to [base] options. */
    fun applyTo(base: ProductRuntime.Options) = base.copy(
      backend = backend, cpuThreads = cpuThreads, speculativeDecoding = speculativeDecoding,
    )

    internal fun encode() = "$backend|$cpuThreads|$speculativeDecoding|$expectedLatencySeconds"

    companion object {
      internal fun decode(s: String): Choice? = runCatching {
        val (b, t, m, l) = s.split("|")
        Choice(ProductRuntime.BackendChoice.valueOf(b), t.toInt(),
               SpeculativeDecodingMode.valueOf(m), l.toDouble())
      }.getOrNull()
    }
  }

  private val key: String = run {
    val f = File(modelPath)
    "gemma.autotune.v1|$deviceId|${f.length()}|${f.lastModified()}"
  }

  /** The stored choice for this device and model file, if tuning already ran. */
  fun cached(): Choice? = store.get(key)?.let { Choice.decode(it) }

  /**
   * Measures [candidates] (skipping any that fail to load) and stores the
   * best. [base] supplies context size, cache dir and so on.
   */
  suspend fun tune(
    candidates: List<Candidate>,
    base: ProductRuntime.Options,
    expectedOutputChars: Int = 400,
    onMeasurement: (Measurement) -> Unit = {},
  ): Choice {
    val all = mutableListOf<Measurement>()
    for (c in candidates) {
      val opts = base.copy(
        backend = c.backend, cpuThreads = c.cpuThreads, cacheDir = cacheDir,
        maxCachedProducts = 3, speculativeDecoding = SpeculativeDecodingMode.ADAPTIVE,
        sampler = base.sampler ?: GREEDY,
      )
      val rt = try {
        ProductRuntime.create(modelPath, opts)
      } catch (t: Throwable) {
        System.err.println("Autotuner: $c unavailable: ${t.message?.lineSequence()?.firstOrNull()}")
        continue
      }
      try {
        // GPU_THEN_CPU may silently land on CPU; don't credit GPU for it.
        if (c.backend != ProductRuntime.BackendChoice.CPU && rt.backend != "GPU") continue
        val modes = listOf(SpeculativeDecodingMode.OFF, SpeculativeDecodingMode.ON,
                           SpeculativeDecodingMode.ADAPTIVE)
        // Prefill every variant before any decode (also avoids the drafter bug path).
        for (m in modes) rt.registerProduct("tune-$m", CALIBRATION_SYSTEM_PROMPT, m)
        for (m in modes) rt.prewarm("tune-$m")
        rt.generateText("tune-OFF", "Say OK.", maxChars = 8) // warm-up
        for (m in modes) {
          var ttft = 0.0; var chars = 0; var decode = 0.0
          for (q in CALIBRATION_QUESTIONS) {
            val r = timeRequest(rt, "tune-$m", q, maxChars = 300)
            ttft += r.first; chars += r.second; decode += r.third
          }
          val n = CALIBRATION_QUESTIONS.size
          val rate = if (decode > 0) chars / decode else 0.0
          val score = ttft / n + if (rate > 0) expectedOutputChars / rate else Double.MAX_VALUE
          Measurement(c, m, rt.initSeconds, ttft / n, rate, score).also { all += it; onMeasurement(it) }
        }
      } finally {
        rt.close()
      }
    }
    val best = all.minByOrNull { it.scoreSeconds }
      ?: error("No candidate could run on this device")
    return Choice(best.candidate.backend, best.candidate.cpuThreads, best.mode, best.scoreSeconds)
      .also { store.put(key, it.encode()) }
  }

  /** (ttft seconds, chars after the first chunk, seconds after the first chunk) */
  private suspend fun timeRequest(rt: ProductRuntime, id: String, q: String, maxChars: Int):
    Triple<Double, Int, Double> {
    val start = System.nanoTime()
    var first = 0L; var last = 0L; var chars = 0
    rt.generate(id, q, maxChars).onEach {
      val now = System.nanoTime()
      if (first == 0L) first = now else chars += it.length
      last = now
    }.toList()
    if (first == 0L) return Triple(0.0, 0, 0.0)
    return Triple((first - start) / 1e9, chars, (last - first) / 1e9)
  }

  companion object {
    private val GREEDY = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0)

    private const val CALIBRATION_SYSTEM_PROMPT =
      "You are the help desk for a regional home-improvement store. Answer briefly and " +
      "accurately from the policies. Returns: 90 days with receipt, 30 days without for store " +
      "credit. Paint tinted to order cannot be returned. Tool rental is billed per four-hour " +
      "block; late returns are billed a full extra block. Delivery is free over 199 dollars " +
      "within 20 miles. Installation appointments can be moved up to 24 hours ahead at no cost. " +
      "Contractors with a trade account get net-30 terms and a 5 percent discount on lumber."

    private val CALIBRATION_QUESTIONS = listOf(
      "Summarize the return policy for someone who lost their receipt.",
      "Rewrite this politely: you're late on the rental, pay for another block.",
      "Explain how a contractor account changes what I pay for lumber.",
    )
  }
}
