package com.margelo.nitro.dev.litert.litertlm.product

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.InputData
import com.google.ai.edge.litertlm.ResponseCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.Session
import com.google.ai.edge.litertlm.SessionConfig
import com.google.ai.edge.litertlm.SpeculativeDecodingConfig
import com.google.ai.edge.litertlm.SpeculativeDecodingMode
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.fold
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Runs Gemma 4 E2B (LiteRT-LM) for many "products", each with its own static
 * system prompt, with the system prompt prefilled once and reused.
 *
 * How the reuse works (LiteRT-LM 0.19 internals):
 *  - Each product gets a *template* [Session] whose KV cache holds only
 *    `<bos><|turn>system\n{prompt}<turn|>\n`. Templates are never advanced.
 *  - A request clones the template. A clone shares the template's KV
 *    (copy-on-write), so cloning is cheap and only the user turn is prefilled.
 *    The clone is closed when the request finishes, which keeps the next
 *    clone on the zero-copy path (a second live clone would force a copy).
 *  - Switching to another product makes the runtime park the outgoing
 *    product's KV and swap in the new one: one KV copy, no re-prefill.
 *
 * Memory: every cached product keeps one full-size KV cache resident, so total
 * KV ~= (maxCachedProducts + 1) x maxNumTokens x [KV_BYTES_PER_TOKEN].
 * Evicted products are re-prefilled on next use.
 *
 * All engine work runs on one dedicated thread; requests are serialized
 * (LiteRT-LM executes one session at a time anyway).
 */
class ProductRuntime private constructor(
  private val engine: Engine,
  val options: Options,
  /** Backend actually in use ("GPU" or "CPU"; GPU may have fallen back). */
  val backend: String,
  /** Seconds spent in Engine.initialize() (0 when attached to an existing engine). */
  val initSeconds: Double,
  /** False when attached to an engine someone else owns and closes. */
  private val ownsEngine: Boolean = true,
) : AutoCloseable {

  enum class BackendChoice { GPU_THEN_CPU, GPU, CPU }

  data class Options(
    /**
     * Context size = KV capacity, shared by system prompt + user input + output.
     * Keep it as small as your largest request allows; see [contextFor].
     */
    val maxNumTokens: Int = 4096,
    /** Products whose prefilled system prompt stays resident (LRU beyond this). */
    val maxCachedProducts: Int = 4,
    val backend: BackendChoice = BackendChoice.GPU_THEN_CPU,
    /** CPU threads. Use the performance-core count; little cores slow XNNPACK down. */
    val cpuThreads: Int = 4,
    /**
     * Default MTP speculative decoding mode for products (overridable per
     * product). ON drafts every step; ADAPTIVE drafts only while measured
     * faster. Lossless at the default lenience. OFF also skips loading the
     * drafter (~45 MB).
     */
    val speculativeDecoding: SpeculativeDecodingMode = SpeculativeDecodingMode.ADAPTIVE,
    /** null = model defaults. Greedy (topK = 1) gives MTP the best acceptance rate. */
    val sampler: SamplerConfig? = null,
    /** Writable, persistent dir for compiled-kernel/weight caches (fast 2nd load). */
    val cacheDir: File? = null,
    /** Adds Gemma 4's <|think|> flag to system turns. Much slower; off by default. */
    val enableThinking: Boolean = false,
  ) {
    init {
      require(maxNumTokens in 256..32768) { "Gemma 4 E2B supports up to 32K context" }
      require(maxCachedProducts >= 1)
      require(cpuThreads >= 1)
    }
  }

  /** Counters for monitoring the MTP workaround in the field. */
  data class Stats(
    val requests: Int,
    val drafterFailures: Int,
    val recoveredByRetry: Int,
    val productsWithMtpOff: List<String>,
  )

  private val thread = Executors.newSingleThreadExecutor { r ->
    Thread(r, "gemma-engine").apply { isDaemon = true }
  }
  private val engineDispatcher = thread.asCoroutineDispatcher()
  private val requestLock = Mutex()

  private val requests = AtomicInteger()
  private val drafterFailures = AtomicInteger()
  private val recovered = AtomicInteger()

  private class Product(
    val id: String,
    val systemPrompt: String,
    val mode: SpeculativeDecodingMode,
  ) {
    var template: Session? = null
    /** Set after MTP failed twice in a row for this product. */
    var mtpOff = false
    val effectiveMode get() = if (mtpOff) SpeculativeDecodingMode.OFF else mode
  }

  private val products = HashMap<String, Product>()
  // Access-ordered: iteration starts at the least recently used template.
  private val resident = LinkedHashMap<String, Product>(16, 0.75f, true)

  /**
   * Registers (or replaces) a product. Cheap; prefill happens on [prewarm] or
   * first use. [speculative] overrides [Options.speculativeDecoding]; ON tends
   * to win for summarizing/rewriting products, ADAPTIVE is safer for code.
   */
  suspend fun registerProduct(
    id: String,
    systemPrompt: String,
    speculative: SpeculativeDecodingMode? = null,
  ) = withContext(engineDispatcher) {
    products.remove(id)?.let { evict(it) }
    products[id] = Product(id, systemPrompt, speculative ?: options.speculativeDecoding)
  }

  /**
   * Prefills a product's system prompt now so its first request is fast.
   * Prewarming every product right after load, before the first request, also
   * avoids the drafter failure path (see [generate]).
   */
  suspend fun prewarm(id: String) = requestLock.withLock {
    withContext(engineDispatcher) { templateFor(id); Unit }
  }

  /**
   * Streams the reply to [userText] under product [id]'s system prompt.
   *
   * [maxChars] stops generation once that much text has been produced; the
   * cheapest speed-up there is is not generating what you won't show.
   *
   * LiteRT-LM 0.19's MTP drafter intermittently fails on the first request of
   * a session created after decoding has started ("Cannot auto-resize tensor
   * kv_cache_*"), always before any output. The failure poisons that
   * template: a new clone of it then fails with "Unprocessed token is null".
   * Recovery, only while nothing has been emitted: rebuild the template with
   * MTP still on; if that fails too, rebuild it with MTP off for this one
   * product.
   */
  fun generate(id: String, userText: String, maxChars: Int? = null): Flow<String> =
    streamFrom(input = userTurn(userText), maxChars = maxChars, label = id) { attempt ->
      val p = products[id] ?: throw IllegalArgumentException("Unknown product '$id'")
      when (attempt) {
        0 -> Unit
        1 -> evict(p)                         // rebuild the template, MTP still on
        else -> { p.mtpOff = true; evict(p) } // last resort: this product only
      }
      templateFor(id).clone()
    }

  /**
   * Baseline without prefix reuse: a fresh session prefills system prompt and
   * user turn together. For A/B checks and benchmarks; prefer [generate].
   */
  fun generateUncached(systemPrompt: String, userText: String, maxChars: Int? = null): Flow<String> =
    streamFrom(systemTurn(systemPrompt) + userTurn(userText), maxChars, "uncached") { attempt ->
      val mode = if (attempt >= 2) SpeculativeDecodingMode.OFF else options.speculativeDecoding
      engine.createSession(sessionConfig(mode))
    }

  /** Non-streaming convenience wrapper around [generate]. */
  suspend fun generateText(id: String, userText: String, maxChars: Int? = null): String =
    generate(id, userText, maxChars).fold(StringBuilder()) { sb, s -> sb.append(s) }.toString()

  /** Drops a product's resident KV (e.g. on onTrimMemory). It's rebuilt on next use. */
  suspend fun evictProduct(id: String) = withContext(engineDispatcher) {
    products[id]?.let { evict(it) }
    Unit
  }

  suspend fun evictAll() = withContext(engineDispatcher) {
    resident.values.toList().forEach { evict(it) }
  }

  /** Product ids whose system prompt is currently prefilled, least recently used first. */
  suspend fun residentProducts(): List<String> =
    withContext(engineDispatcher) { resident.keys.toList() }

  /** Lets MTP be tried again for products where it was switched off. */
  suspend fun resetMtp() = withContext(engineDispatcher) {
    products.values.filter { it.mtpOff }.forEach { it.mtpOff = false; evict(it) }
  }

  suspend fun stats(): Stats = withContext(engineDispatcher) {
    Stats(requests.get(), drafterFailures.get(), recovered.get(),
          products.values.filter { it.mtpOff }.map { it.id })
  }

  /**
   * Closes the templates (and the engine, if this runtime created it).
   * Waits for an in-flight request to finish closing its session first, so
   * the engine is never closed under a live session; cancel the request's
   * collector beforehand to make that quick.
   */
  override fun close() {
    kotlinx.coroutines.runBlocking {
      requestLock.withLock {
        withContext(engineDispatcher) {
          resident.values.toList().forEach { evict(it) }
          if (ownsEngine) engine.close()
        }
      }
    }
    thread.shutdown()
    thread.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
  }

  // --- request plumbing ---------------------------------------------------

  private fun streamFrom(
    input: String,
    maxChars: Int?,
    label: String,
    open: (attempt: Int) -> Session,
  ): Flow<String> = callbackFlow {
    requestLock.lock()
    try {
      requests.incrementAndGet()
      var attempt = 0
      var failedBefore = false
      while (true) {
        var emitted = 0
        val error = runOnce({ open(attempt) }, input, maxChars) { chunk ->
          emitted++
          trySend(chunk)
        }
        if (error == null) {
          if (failedBefore) recovered.incrementAndGet()
          break
        }
        // After a drafter failure, any error before output is part of the same
        // incident (the runtime may report it differently), so keep retrying.
        if (emitted == 0 && attempt < MAX_ATTEMPT && (failedBefore || isDrafterFailure(error))) {
          drafterFailures.incrementAndGet()
          failedBefore = true
          attempt++
          log("MTP drafter failed for '$label' (retry $attempt of $MAX_ATTEMPT): " +
              error.message?.lineSequence()?.firstOrNull())
          continue
        }
        throw error
      }
      channel.close()
    } catch (e: kotlinx.coroutines.CancellationException) {
      throw e
    } catch (t: Throwable) {
      channel.close(t)
    } finally {
      requestLock.unlock()
    }
    awaitClose {}
  }

  /**
   * Runs one generation; returns its error (null on success). Called with
   * [requestLock] held. The session is closed before returning, so the next
   * clone stays on the zero-copy path.
   */
  private suspend fun runOnce(
    open: () -> Session,
    input: String,
    maxChars: Int?,
    emit: (String) -> Unit,
  ): Throwable? {
    var session: Session? = null
    try {
      session = withContext(engineDispatcher) { open() }
      injectedFailure()?.let { return it }
      val s = session
      val finished = CompletableDeferred<Unit>()
      var produced = 0
      s.generateContentStream(
        listOf(InputData.Text(input)),
        object : ResponseCallback {
          override fun onNext(response: String) {
            emit(response)
            produced += response.length
            if (maxChars != null && produced >= maxChars) s.cancelProcess()
          }

          override fun onDone() {
            finished.complete(Unit)
          }

          override fun onError(throwable: Throwable) {
            if (throwable is CancellationException) finished.complete(Unit)
            else finished.completeExceptionally(throwable)
          }
        },
      )
      try {
        finished.await()
      } catch (e: kotlinx.coroutines.CancellationException) {
        s.cancelProcess()
        throw e
      }
      return null
    } catch (e: kotlinx.coroutines.CancellationException) {
      throw e
    } catch (t: Throwable) {
      return t
    } finally {
      withContext(NonCancellable + engineDispatcher) { session?.close() }
    }
  }

  private fun isDrafterFailure(t: Throwable): Boolean {
    val msg = t.message ?: return false
    return "mtp_drafter" in msg || "Cannot auto-resize tensor kv_cache" in msg
  }

  // --- engine thread only -------------------------------------------------

  private fun templateFor(id: String): Session {
    val p = products[id] ?: throw IllegalArgumentException("Unknown product '$id'")
    p.template?.let { t ->
      resident[id] // touch for LRU order
      return t
    }
    while (resident.size >= options.maxCachedProducts) {
      evict(resident.values.first())
    }
    val session = engine.createSession(sessionConfig(p.effectiveMode))
    try {
      // The runtime prepends <bos> to a session's first prefill.
      session.runPrefill(listOf(InputData.Text(systemTurn(p.systemPrompt))))
    } catch (t: Throwable) {
      session.close()
      throw t
    }
    p.template = session
    resident[id] = p
    return session
  }

  private fun evict(p: Product) {
    p.template?.close()
    p.template = null
    resident.remove(p.id)
  }

  // A session asking for MTP on an engine loaded without it makes LiteRT-LM
  // load the drafter lazily, so per-product overrides work either way.
  private fun sessionConfig(mode: SpeculativeDecodingMode) =
    SessionConfig(
      samplerConfig = options.sampler,
      speculativeDecodingConfig = SpeculativeDecodingConfig(mode = mode),
    )

  private fun systemTurn(prompt: String): String {
    val think = if (options.enableThinking) "<|think|>" else ""
    return "<|turn>system\n$think${sanitize(prompt)}<turn|>\n"
  }

  companion object {
    /** Retries after the first attempt: rebuilt template with MTP on, then off. */
    private const val MAX_ATTEMPT = 2

    private val DEBUG = System.getenv("GEMMA_RUNTIME_DEBUG") == "1"

    /** Test hook: number of drafter failures to simulate (JVM property). */
    private val injectRemaining =
      AtomicInteger(System.getProperty("gemma.test.injectDrafterFailures")?.toIntOrNull() ?: 0)

    private fun injectedFailure(): Throwable? =
      if (injectRemaining.getAndUpdate { if (it > 0) it - 1 else 0 } > 0)
        RuntimeException("[injected] llm_litert_mtp_drafter.cc: Cannot auto-resize tensor kv_cache_v_3")
      else null

    private fun log(msg: String) = System.err.println("ProductRuntime: $msg")

    @Suppress("unused")
    private inline fun debug(msg: () -> String) {
      if (DEBUG) System.err.println("[ProductRuntime ${System.nanoTime() / 1_000_000}] ${msg()}")
    }

    /**
     * KV bytes per token for gemma-4-E2B-it.litertlm: INT8 K+V over 15 shared
     * KV layers (12 x 256 + 3 x 512 channels each). From the file's decode
     * signature; use litertlm_kv_audit.py to check another build.
     */
    const val KV_BYTES_PER_TOKEN = 9216L

    /**
     * Runs products on an engine that already exists (e.g. the one the
     * plugin's conversation API uses), so both share one ~0.8 GB engine.
     * The engine must have been created with a speculative-decoding mode
     * other than OFF for MTP to be available to products, and the owner must
     * close this runtime before closing the engine.
     */
    fun attach(engine: Engine, options: Options, backend: String): ProductRuntime =
      ProductRuntime(engine, options, backend, 0.0, ownsEngine = false)

    /**
     * Loads the engine. Blocking and slow (seconds) on first run; call once,
     * off the main thread, and keep the instance for the app's lifetime.
     */
    fun create(modelPath: String, options: Options = Options()): ProductRuntime {
      val cache = options.cacheDir?.apply { mkdirs() }?.absolutePath
      fun build(backend: Backend, name: String): ProductRuntime {
        val t0 = System.nanoTime()
        val engine = Engine(
          EngineConfig(
            modelPath = modelPath,
            backend = backend,
            visionBackend = null, // text only: skips ~330 MB of vision/audio encoders
            audioBackend = null,
            maxNumTokens = options.maxNumTokens,
            cacheDir = cache,
            // ON/ADAPTIVE loads the drafter once; sessions pick their own mode.
            speculativeDecodingConfig = SpeculativeDecodingConfig(
              mode = if (options.speculativeDecoding == SpeculativeDecodingMode.OFF)
                SpeculativeDecodingMode.OFF else SpeculativeDecodingMode.ADAPTIVE
            ),
          )
        )
        engine.initialize()
        return ProductRuntime(engine, options, name, (System.nanoTime() - t0) / 1e9)
      }

      val cpu = { build(Backend.CPU(threadCount = options.cpuThreads), "CPU") }
      return when (options.backend) {
        BackendChoice.CPU -> cpu()
        BackendChoice.GPU -> build(Backend.GPU(), "GPU")
        BackendChoice.GPU_THEN_CPU -> try {
          build(Backend.GPU(), "GPU")
        } catch (t: Throwable) {
          log("GPU init failed, falling back to CPU: ${t.message?.lineSequence()?.firstOrNull()}")
          cpu()
        }
      }
    }

    /** User turn plus the model-turn opener; generation continues from here. */
    fun userTurn(text: String) = "<|turn>user\n${sanitize(text)}<turn|>\n<|turn>model\n"

    /**
     * Session text is tokenized raw, so a literal "<|turn>" in user input
     * would become a real control token. Break up Gemma's control-token
     * delimiters (<|x>, <x|>, <bos>) with a zero-width space.
     */
    fun sanitize(text: String): String =
      text.replace("<|", "<​|").replace("|>", "|​>").replace("<bos>", "<​bos>")

    /**
     * Smallest context that fits the largest request you'll make.
     *
     * Size the context to the workload, not to the phone: LiteRT-LM allocates
     * the KV cache at maxNumTokens and every decode step attends over all of
     * it, so an oversized context slows every token. Measured on this model
     * (CPU): 4K -> 16K made decode 2.4x slower and time-to-first-token 2.6x
     * slower for the same short requests.
     *
     * Character counts are converted at 3 chars/token, a safety margin over
     * Gemma's ~4 for English. Rounded up to a multiple of 1024.
     */
    fun contextFor(longestSystemPromptChars: Int, maxUserChars: Int, maxOutputChars: Int): Int {
      val tokens = (longestSystemPromptChars + maxUserChars + maxOutputChars) / 3 + 64
      return (((tokens + 1023) / 1024) * 1024).coerceIn(1024, 32768)
    }

    /**
     * Largest context whose KV for all cached products fits in [kvFraction]
     * of RAM. An upper bound only; use [contextFor] to pick the size.
     */
    fun maxContextForRam(totalRamBytes: Long, maxCachedProducts: Int, kvFraction: Double = 0.20): Int {
      val perToken = KV_BYTES_PER_TOKEN * (maxCachedProducts + 1)
      val n = (totalRamBytes * kvFraction / perToken).toLong()
      return ((n / 1024) * 1024).toInt().coerceIn(1024, 32768)
    }
  }
}
