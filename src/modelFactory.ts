import { NitroModules } from "react-native-nitro-modules";
import {
  LiteRTLM,
  LLMConfig,
  Message,
  MultimodalPart,
  ExecuteOptions,
} from "./specs/LiteRTLM.nitro";
import { createMemoryTracker, MemoryTracker } from "./memoryTracker";
import { ModelRegistry, type ModelDownloadOptions } from "./modelRegistry";
import {
  isLegacyInferenceMethod,
  routeLegacyInference,
  TokenCallback,
} from "./inferenceRouting";
import {
  estimateMemory,
  estimateKvCacheBytesPerToken,
  forecastMemory,
  resolveMaxContextTokens,
  evaluateBudget,
  type MemoryEstimate,
  type MemoryForecast,
  type MemoryBudget,
  type BudgetLevel,
} from "./core/memoryEstimator";
import { LiteRTLMError, MemoryError } from "./core/errors";
import { toLiteRTLMError, type NativePhase } from "./core/nativeError";
import { legacyMultimodalHint, preflightLoad, type ModelInfo } from "./core/modelInfo";
import { warnDeprecated } from "./core/deprecations";
import { inspectModel, tryGetDeviceInfo } from "./modelInspector";
import {
  createStreamEventParser,
  DEFAULT_CHANNELS,
  type StreamEvent,
  type StreamChannel,
} from "./core/streamEvents";

/** Callback receiving typed streaming events from `executeWithEvents`. */
export type StreamEventCallback = (event: StreamEvent) => void;

/** Options for {@link LiteRTLMInstance.createConversation}. */
export interface ConversationOptions {
  /** Override the session-level `systemPrompt` for this conversation. */
  systemPrompt?: string;
  /**
   * Turn speculative decoding (the bundle's multi-token drafter, e.g. Gemma 4's
   * MTP) on or off for this conversation; unset follows
   * `LLMConfig.enableSpeculativeDecoding`. `true` loads the drafter on first use.
   */
  speculativeDecoding?: boolean;
}

/**
 * A logical conversation sharing the loaded engine with other conversations.
 *
 * One engine holds one native context at a time: switching between
 * conversations replays the target's transcript into a fresh context
 * (re-prefill on the next message — expect seconds of extra latency on long
 * transcripts). Multimodal turns replay as text placeholders (`[Image]`,
 * `[Audio]`); images/audio are not re-encoded on switch.
 */
export interface ConversationHandle {
  /** Stable identifier for this conversation. */
  readonly id: string;
  /** Run inference in this conversation (switches native context if needed). */
  execute(
    parts: MultimodalPart[],
    onToken?: TokenCallback,
    options?: ExecuteOptions,
  ): Promise<string>;
  /** Typed-event variant of {@link execute}. */
  executeWithEvents(
    parts: MultimodalPart[],
    onEvent: StreamEventCallback,
    options?: ExecuteOptions,
  ): Promise<string>;
  /** This conversation's transcript (live when active, snapshot otherwise). */
  getHistory(): Message[];
  /** Drop this conversation's transcript; further calls on the handle reject. */
  release(): Promise<void>;
}

/**
 * Extended LiteRT-LM instance with memory controls and typed event streaming,
 * plus an augmented loadModel that accepts a download progress callback and
 * runs an automatic pre-flight memory check.
 *
 * Inference calls (`execute()`, `executeWithEvents()`, the deprecated
 * `sendMessage*()` methods and conversation handles) run one at a time, in
 * the order they were made. A call made while another is generating waits
 * its turn, and reads its parts (buffers included) only then. A waiting
 * message is cancelled like the one generating, rejecting with
 * `INFERENCE_CANCELLED`: `resetConversation()` cancels the default
 * conversation's waiting messages, and `loadModel()` (once it replaces the
 * engine), `unload()` and `close()` cancel all of them.
 */
export type LiteRTLMInstance = Omit<LiteRTLM, "loadModel"> & {
  memoryTracker?: MemoryTracker;
  /**
   * The third argument is a download progress callback, or the options of
   * `ModelRegistry.resolveModel()` (`onProgress`, `headers`, `sha256`).
   */
  loadModel: (
    pathOrUrl: string,
    config?: LLMConfig,
    download?: ((progress: number) => void) | ModelDownloadOptions,
  ) => Promise<void>;
  /**
   * Pre-flight memory estimate for the most recently loaded (or a given)
   * model path. Returns null if the model file cannot be found.
   */
  estimateMemory: (modelPath?: string, config?: LLMConfig) => MemoryEstimate | null;
  /**
   * Forecast context-window / KV-cache growth for the active conversation.
   * Returns null if no model is loaded.
   */
  getMemoryForecast: () => MemoryForecast | null;
  /**
   * Execute inference receiving typed streaming events (token / toolCall /
   * thinking) instead of a raw token stream. Completed tool calls arrive as
   * `toolCall` events on both platforms; `streamToolCalls: true` streams them
   * token by token (iOS). Answer them with `toolResultPart()`.
   */
  executeWithEvents: (
    parts: MultimodalPart[],
    onEvent: StreamEventCallback,
    options?: ExecuteOptions,
  ) => Promise<string>;
  /**
   * What the loaded bundle declares about itself (type, modalities, context,
   * backends, sampler defaults…), read during `loadModel()`. Null before a
   * load, or when the bundle's metadata could not be read.
   */
  getModelInfo: () => ModelInfo | null;
  /**
   * Create a logical conversation sharing the loaded engine. Multiple
   * conversations coexist without loading the model twice; switching between
   * them replays the target's transcript (re-prefill cost on the next
   * message). All conversations share one queue: one message runs at a time.
   * A manual `resetConversation()` restarts only the default conversation:
   * the others keep their turns and their waiting messages.
   */
  createConversation: (options?: ConversationOptions) => ConversationHandle;
};

/** Options for {@link createLLM}. */
export interface CreateLLMOptions {
  /** Enable automatic memory tracking (default: false) */
  enableMemoryTracking?: boolean;
  /** Maximum number of memory snapshots to retain (ring buffer; default: 256) */
  maxMemorySnapshots?: number;
  /**
   * Memory budget thresholds evaluated on every recorded snapshot.
   * `onBudgetExceeded` fires on level transitions (ok→warn, warn→critical, …).
   */
  memoryBudget?: MemoryBudget & {
    onBudgetExceeded?: (level: BudgetLevel) => void;
  };
  /**
   * Channel markers for typed event streaming (tool calls, thinking).
   * Defaults to `<tool_call>…</tool_call>` and `<thinking>…</thinking>`.
   */
  streamChannels?: StreamChannel[];
}

/**
 * Creates a new LiteRT-LM inference engine instance with memory controls:
 * automatic pre-flight estimation in `loadModel` (rejects with `MemoryError`
 * when the model likely doesn't fit), context/KV forecasting, budget
 * monitoring, and typed event streaming.
 */
export function createLLM(options?: CreateLLMOptions): LiteRTLMInstance {
  const native = NitroModules.createHybridObject<LiteRTLM>("LiteRTLM");

  const enableTracking = options?.enableMemoryTracking ?? false;
  const tracker = enableTracking
    ? createMemoryTracker(options?.maxMemorySnapshots ?? 256)
    : undefined;
  const budget = options?.memoryBudget;
  const channels = options?.streamChannels ?? DEFAULT_CHANNELS;

  // Cached facts about the currently loaded model (for estimate/forecast)
  let loadedModelPath: string | undefined;
  let loadedModelFileSize = -1;
  let loadedConfig: LLMConfig | undefined;
  let loadedModelInfo: ModelInfo | null = null;
  let lastBudgetLevel: BudgetLevel = "ok";
  // Moved by unload() and close(): a load still downloading or reading its
  // bundle when either is called stops before the engine loads anything
  let teardowns = 0;
  let closed = false;
  // Native rejects a message sent before a reset, load, unload or close that
  // hasn't finished, whether it is generating or waiting to start. Messages
  // waiting in the inference queue (below) haven't reached native yet, so the
  // queue applies the same rule with two counters: a message notes them when
  // it is sent and is cancelled, not run, if one has moved by its turn.
  /** Moved as loadModel() replaces the engine, and by unload() and close() */
  let engineEpoch = 0;
  /** Moved by resetConversation(): the default conversation starts over */
  let defaultResets = 0;

  const closedError = () =>
    new LiteRTLMError("MODEL_NOT_LOADED", "This instance is closed; create a new one with createLLM().");

  const checkBudget = (residentBytes: number, availableBytes: number) => {
    if (!budget?.onBudgetExceeded) return;
    const level = evaluateBudget(residentBytes, availableBytes, budget);
    if (level !== lastBudgetLevel && level !== "ok") {
      budget.onBudgetExceeded(level);
    }
    lastBudgetLevel = level;
  };

  const recordMemorySnapshot = () => {
    if (!tracker && !budget?.onBudgetExceeded) return;
    try {
      const usage = native.getMemoryUsage();
      tracker?.record({
        timestamp: Date.now(),
        nativeHeapBytes: usage.nativeHeapBytes,
        residentBytes: usage.residentBytes,
        availableMemoryBytes: usage.availableMemoryBytes,
      });
      checkBudget(usage.residentBytes, usage.availableMemoryBytes);
    } catch {
      // Non-critical
    }
  };

  const runEstimate = (
    modelPath?: string,
    config?: LLMConfig,
  ): MemoryEstimate | null => {
    const path = modelPath ?? loadedModelPath;
    if (!path) return null;
    const fileSize =
      path === loadedModelPath && loadedModelFileSize >= 0
        ? loadedModelFileSize
        : ModelRegistry.getFileSizeBytes(path);
    if (fileSize < 0) return null;
    const usage = native.getMemoryUsage();
    return estimateMemory({
      modelFileSizeBytes: fileSize,
      availableMemoryBytes: usage.availableMemoryBytes,
      config: config ?? loadedConfig,
    });
  };

  /**
   * Type a native rejection. Engine out-of-memory becomes a MemoryError that
   * carries the pre-flight estimate for the same model and config, so
   * `isMemoryError()` covers both "refused up front" and "ran out mid-way".
   */
  const toTypedError = (
    e: unknown,
    phase: NativePhase,
    estimateFor?: { modelPath?: string; config?: LLMConfig },
  ): LiteRTLMError => {
    const error = toLiteRTLMError(e, phase);
    if (error.code !== "OUT_OF_MEMORY") return error;
    let estimate: MemoryEstimate | null = null;
    try {
      estimate = runEstimate(estimateFor?.modelPath, estimateFor?.config);
    } catch {
      // No estimate available (no file, no memory reading) — keep the plain error.
    }
    if (!estimate) return error;
    return new MemoryError(error.message, estimate, {
      code: "OUT_OF_MEMORY",
      nativeStatus: error.nativeStatus,
      cause: error.cause,
    });
  };

  const augmentedLoadModel = async (
    pathOrUrl: string,
    config?: LLMConfig,
    download?: ((progress: number) => void) | ModelDownloadOptions,
  ) => {
    if (config?.maxTokens !== undefined) {
      warnDeprecated("LLMConfig.maxTokens", "maxContextTokens and maxOutputTokens");
    }
    if (closed) throw closedError();
    const teardownsAtStart = teardowns;
    const modelPath = await ModelRegistry.resolveModel(
      pathOrUrl,
      typeof download === "object" ? download : { onProgress: download },
    );

    // Bundle pre-flight (LiteRT-LM 0.18 model info, header only): refuse the
    // wrong model type or a bundle needing a newer runtime, pick the backend
    // the bundle prefers when none was given, clamp the context to the
    // bundle's maximum. A missing file fails here; unreadable metadata (older
    // bundles) only skips this step — the engine still gets the final say.
    let info: ModelInfo | null = null;
    try {
      info = await inspectModel(modelPath);
    } catch (e) {
      if (e instanceof LiteRTLMError && e.code === "MODEL_NOT_FOUND") throw e;
      console.warn(
        `[react-native-litert-lm] Could not read the metadata of ${modelPath} ` +
          `(${e instanceof Error ? e.message : String(e)}); loading without the bundle pre-flight.`,
      );
    }
    // With `multimodal` unset, native code loads the encoders the bundle
    // declares — or, for a bundle whose metadata can't be read, falls back to
    // 0.7's file-name default. Only that fallback is deprecated.
    if (config?.multimodal === undefined && !info && legacyMultimodalHint(modelPath)) {
      warnDeprecated(
        "Enabling vision/audio from the model's file name",
        "multimodal: true in the loadModel() config",
      );
    }
    let effectiveConfig = config;
    if (info) {
      const preflight = preflightLoad(info, config, { device: tryGetDeviceInfo() });
      if (preflight.changed) effectiveConfig = preflight.config;
      for (const note of preflight.notes) console.warn(`[react-native-litert-lm] ${note}`);
    }

    // Pre-flight memory check: refuse loads that would likely OOM-kill the
    // app. `forceLoad: true` opts out.
    const fileSize = ModelRegistry.getFileSizeBytes(modelPath);
    if (fileSize >= 0 && !effectiveConfig?.forceLoad) {
      const usage = native.getMemoryUsage();
      const estimate = estimateMemory({
        modelFileSizeBytes: fileSize,
        availableMemoryBytes: usage.availableMemoryBytes,
        config: effectiveConfig,
      });
      if (estimate.verdict === "critical") {
        throw new MemoryError(
          `Refusing to load model (${Math.round(fileSize / (1024 * 1024))} MB): ` +
            estimate.recommendation +
            " Pass { forceLoad: true } to override.",
          estimate,
        );
      }
    }

    // A download another instance shares keeps running after this one is
    // closed (useModel closes on unmount): the model must not load into it
    if (teardowns !== teardownsAtStart) {
      throw new LiteRTLMError(
        "MODEL_NOT_LOADED",
        `The instance was ${closed ? "closed" : "unloaded"} before ${modelPath} finished loading; nothing was loaded.`,
      );
    }
    // Replacing the engine cancels the message generating (native does that)
    // and the messages waiting behind it (the queue does)
    engineEpoch++;
    const result = await native.loadModel(modelPath, effectiveConfig).catch((e: unknown) => {
      throw toTypedError(e, "load", { modelPath, config: effectiveConfig });
    });

    loadedModelPath = modelPath;
    loadedModelFileSize = fileSize;
    loadedConfig = effectiveConfig;
    loadedModelInfo = info;
    resetConversationManagerState();

    if (tracker) {
      tracker.reset();
    }
    recordMemorySnapshot();

    return result;
  };

  const getForecast = (): MemoryForecast | null => {
    if (loadedModelFileSize < 0) return null;
    const used = native.getContextTokenCount();
    if (used < 0) return null;
    return forecastMemory({
      contextTokensUsed: used,
      maxContextTokens: resolveMaxContextTokens(loadedConfig),
      kvCacheBytesPerToken: estimateKvCacheBytesPerToken(loadedModelFileSize),
    });
  };

  /** Single JS inference path — always calls native execute(). */
  const runExecute = (
    parts: MultimodalPart[],
    onToken?: TokenCallback,
    options?: ExecuteOptions,
  ): Promise<string> => {
    const processedParts = parts.map((part) => {
      if (part.path?.startsWith("file://")) {
        return { ...part, path: part.path.substring(7) };
      }
      return part;
    });

    const rethrow = (e: unknown): never => {
      throw toTypedError(e, "execute");
    };
    if (onToken) {
      const wrapped: TokenCallback = (token, done) => {
        onToken(token, done);
        if (done) recordMemorySnapshot();
      };
      return native.execute(processedParts, wrapped, options).catch(rethrow);
    }
    return native
      .execute(processedParts, undefined, options)
      .then((result: string) => {
        recordMemorySnapshot();
        return result;
      }, rethrow);
  };

  /** Typed-event streaming: raw token stream → token/toolCall/thinking events. */
  const runExecuteWithEvents = (
    parts: MultimodalPart[],
    onEvent: StreamEventCallback,
    options?: ExecuteOptions,
  ): Promise<string> => {
    const parser = createStreamEventParser(channels);
    const onToken: TokenCallback = (token, done) => {
      for (const event of parser.push(token)) {
        onEvent(event);
      }
      if (done) {
        for (const event of parser.finish()) {
          onEvent(event);
        }
      }
    };
    return runExecute(parts, onToken, options);
  };

  // ── Inference queue and multi-conversation manager ─────────────────────
  // An instance runs one message at a time, in call order. The engine has a
  // single native conversation, and a message sent while another was still
  // generating ran alongside it: on Android the first call resolved with the
  // second's reply, and a message sent during another's prefill failed with
  // "Session is not prefilled yet.".
  //
  // One native context exists at a time; logical conversations are switched
  // by snapshotting the outgoing native history and replaying the incoming
  // transcript via resetConversation (re-prefill on the next message).
  // `null` identifies the default conversation (top-level execute calls).
  let conversationsInUse = false;
  let conversationSeq = 0;
  /**
   * The conversation whose transcript the native context holds; `undefined`
   * while native resets, and after a reset that failed. Then none does: the
   * next switch replays a saved transcript and saves nothing native left.
   */
  let activeConversationId: string | null | undefined = null;
  const savedTranscripts = new Map<string | null, Message[]>();
  /** The default conversation's options, set by the last manual reset */
  let defaultConversationOptions: ConversationOptions | undefined;
  /** Settles when the last task queued so far has settled */
  let queueTail: Promise<void> = Promise.resolve();
  /** Tasks queued or running; 0 while the queue is idle */
  let queueLength = 0;

  /**
   * A per-conversation speculative-decoding request the loaded bundle can
   * honor: without a drafter the engine would fail the first message.
   */
  const supportedSpeculativeDecoding = (requested: boolean | undefined): boolean | undefined => {
    if (requested === true && loadedModelInfo?.llm && !loadedModelInfo.llm.speculativeDecoding) {
      console.warn(
        "[react-native-litert-lm] speculativeDecoding: this bundle has no multi-token-prediction drafter — ignoring it.",
      );
      return undefined;
    }
    return requested;
  };

  /**
   * Give conversation `id` the native context: save the outgoing
   * conversation's transcript, then replay `id`'s with its options (the
   * default conversation's come from the last manual reset). If native
   * fails, the error is typed and no conversation holds the context.
   */
  const switchTo = (id: string | null, options?: ConversationOptions) => {
    if (activeConversationId === id) return;
    if (activeConversationId !== undefined) {
      savedTranscripts.set(activeConversationId, native.getHistory());
    }
    const transcript = savedTranscripts.get(id) ?? [];
    const switchOptions = id === null ? defaultConversationOptions : options;
    // Android replaces its transcript before it can fail, so until native
    // returns, what it holds is no conversation's transcript
    activeConversationId = undefined;
    try {
      native.resetConversation(
        transcript.length > 0 ? JSON.stringify(transcript) : undefined,
        switchOptions?.systemPrompt,
        supportedSpeculativeDecoding(switchOptions?.speculativeDecoding),
      );
    } catch (e) {
      throw toLiteRTLMError(e, "load");
    }
    activeConversationId = id;
  };

  /**
   * `turns` up to the model's last reply: without the user or tool turns of
   * a message that got no answer. Android writes those when a message
   * starts and keeps them when it is cancelled; iOS writes a message's
   * turns only once it completes.
   */
  const answeredTurns = (turns: Message[]): Message[] => {
    let end = turns.length;
    while (end > 0 && (turns[end - 1].role === "user" || turns[end - 1].role === "tool")) end--;
    return end === turns.length ? turns : turns.slice(0, end);
  };

  /**
   * Run `fn` once every task queued before it has settled. On an idle queue
   * it starts at once, in the caller's tick, as calls did before the queue
   * existed: native execute() reads the message, buffers included, before it
   * returns. A task that has to wait reads its arguments when its turn comes.
   */
  const enqueue = <T>(fn: () => Promise<T>): Promise<T> => {
    const previous = queueTail;
    const idle = queueLength === 0;
    let release!: () => void;
    queueTail = new Promise<void>((resolve) => {
      release = resolve;
    });
    queueLength++;
    let task: Promise<T>;
    if (idle) {
      try {
        task = Promise.resolve(fn());
      } catch (e) {
        task = Promise.reject(e);
      }
    } else {
      task = previous.then(fn);
    }
    const settled = () => {
      queueLength--;
      release();
    };
    task.then(settled, settled);
    return task;
  };

  /**
   * Queue an inference call in conversation `id` (null: the default one).
   * When its turn comes, it rejects with INFERENCE_CANCELLED, without
   * reaching native, if the engine was replaced since it was sent or, in
   * the default conversation, the conversation was reset. Then `check`
   * runs, and once conversations are in use the native context switches to
   * `id`. On a closed instance the call rejects at once.
   */
  const withConversation = <T>(
    id: string | null,
    options: ConversationOptions | undefined,
    fn: () => Promise<T>,
    check?: () => void,
  ): Promise<T> => {
    if (closed) return Promise.reject(closedError());
    const sentEngine = engineEpoch;
    const sentResets = defaultResets;
    return enqueue(() => {
      if (sentEngine !== engineEpoch || (id === null && sentResets !== defaultResets)) {
        throw new LiteRTLMError(
          "INFERENCE_CANCELLED",
          "The conversation was reset or the model unloaded before this message ran.",
        );
      }
      check?.();
      if (conversationsInUse) switchTo(id, options);
      return fn();
    });
  };

  const createConversation = (
    options?: ConversationOptions,
  ): ConversationHandle => {
    conversationsInUse = true;
    const id = `conversation_${++conversationSeq}`;
    let released = false;

    const guard = () => {
      if (released) {
        throw new Error(`LiteRTLM: conversation '${id}' has been released.`);
      }
    };

    return {
      id,
      // `guard` runs before the switch: a released handle never takes the
      // native context
      execute: (parts, onToken, options_) =>
        withConversation(id, options, () => runExecute(parts, onToken, options_), guard),
      executeWithEvents: (parts, onEvent, options_) =>
        withConversation(id, options, () => runExecuteWithEvents(parts, onEvent, options_), guard),
      getHistory: () => {
        guard();
        return activeConversationId === id
          ? native.getHistory()
          : (savedTranscripts.get(id) ?? []);
      },
      release: () =>
        enqueue(async () => {
          if (released) return;
          released = true;
          try {
            // If this conversation holds the native context, hand it back to
            // the default conversation (replaying its transcript, if any). A
            // closed instance has no context to hand back.
            if (activeConversationId === id && !closed) switchTo(null);
          } catch {
            // Native couldn't (no model loaded, or an unload under way): no
            // conversation holds the context now, and the next message
            // replays its own. The handle is released all the same.
          } finally {
            savedTranscripts.delete(id);
          }
        }),
    };
  };

  const resetConversationManagerState = () => {
    savedTranscripts.clear();
    activeConversationId = null;
    defaultConversationOptions = undefined;
  };

  // unload(), close() and resetConversation() go straight to native, never
  // through the queue: they must reach the message that is generating.
  const unloadWrapped = async (): Promise<void> => {
    teardowns++;
    engineEpoch++;
    await native.unload();
    loadedModelPath = undefined;
    loadedModelFileSize = -1;
    loadedConfig = undefined;
    loadedModelInfo = null;
    resetConversationManagerState();
    recordMemorySnapshot();
  };

  const closeWrapped = (): void => {
    teardowns++;
    engineEpoch++;
    closed = true;
    native.close();
  };

  return new Proxy(native, {
    get(target, prop, receiver) {
      if (typeof prop !== "string") {
        return Reflect.get(target, prop, receiver);
      }

      if (prop === "memoryTracker") {
        return tracker;
      }
      if (prop === "loadModel") {
        return augmentedLoadModel;
      }
      if (prop === "estimateMemory") {
        return runEstimate;
      }
      if (prop === "getMemoryForecast") {
        return getForecast;
      }
      if (prop === "getModelInfo") {
        return () => loadedModelInfo;
      }
      if (prop === "executeWithEvents") {
        return (
          parts: MultimodalPart[],
          onEvent: StreamEventCallback,
          options?: ExecuteOptions,
        ) =>
          withConversation(null, undefined, () =>
            runExecuteWithEvents(parts, onEvent, options),
          );
      }
      if (prop === "unload") {
        return unloadWrapped;
      }
      if (prop === "close") {
        return closeWrapped;
      }
      if (prop === "createConversation") {
        return createConversation;
      }
      if (prop === "execute") {
        return (
          parts: MultimodalPart[],
          onToken?: TokenCallback,
          options?: ExecuteOptions,
        ) =>
          withConversation(null, undefined, () =>
            runExecute(parts, onToken, options),
          );
      }
      if (prop === "resetConversation") {
        return (historyJson?: string, systemPrompt?: string, speculativeDecoding?: boolean) => {
          // A manual reset starts the default conversation over and gives it
          // the live context. Native cancels the message generating; the
          // default conversation's messages waiting behind it are cancelled
          // too. Other conversations' waiting messages still run.
          defaultResets++;
          // A conversation handle holding the context keeps its turns for its
          // next message, without the unanswered one this reset cancels
          if (activeConversationId !== null && activeConversationId !== undefined) {
            savedTranscripts.set(activeConversationId, answeredTurns(native.getHistory()));
          }
          // Switching back to the default conversation replays it with this
          // reset's options. If native fails, no conversation holds the
          // context, and the default conversation starts over empty.
          const supported = supportedSpeculativeDecoding(speculativeDecoding);
          savedTranscripts.delete(null);
          defaultConversationOptions = { systemPrompt, speculativeDecoding: supported };
          activeConversationId = undefined;
          let result: void;
          try {
            result = target.resetConversation(historyJson, systemPrompt, supported);
          } catch (e) {
            throw toLiteRTLMError(e, "load");
          }
          activeConversationId = null;
          recordMemorySnapshot();
          return result;
        };
      }

      if (isLegacyInferenceMethod(prop)) {
        return (...args: unknown[]) => {
          warnDeprecated(`llm.${prop}()`, "llm.execute()");
          const route = routeLegacyInference(prop, args)!;
          const promise = withConversation(null, undefined, () =>
            runExecute(route.parts, route.onToken),
          );
          return prop.endsWith("Async") ? promise.then(() => {}) : promise;
        };
      }

      const original = target[prop as keyof LiteRTLM];
      if (typeof original === "function") {
        return original.bind(target);
      }
      return original;
    },
  }) as unknown as LiteRTLMInstance;
}
