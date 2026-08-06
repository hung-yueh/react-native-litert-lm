import { NitroModules } from "react-native-nitro-modules";
import {
  LiteRTLM,
  LLMConfig,
  Message,
  MultimodalPart,
  ExecuteOptions,
} from "./specs/LiteRTLM.nitro";
import { createMemoryTracker, MemoryTracker } from "./memoryTracker";
import { ModelRegistry } from "./modelRegistry";
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
} from "./memoryEstimator";
import { MemoryError } from "./errors";
import {
  createStreamEventParser,
  DEFAULT_CHANNELS,
  type StreamEvent,
  type StreamChannel,
} from "./streamEvents";

/** Callback receiving typed streaming events from `executeWithEvents`. */
export type StreamEventCallback = (event: StreamEvent) => void;

/** Options for {@link LiteRTLMInstance.createConversation}. */
export interface ConversationOptions {
  /** Override the session-level `systemPrompt` for this conversation. */
  systemPrompt?: string;
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
 */
export type LiteRTLMInstance = Omit<LiteRTLM, "loadModel"> & {
  memoryTracker?: MemoryTracker;
  loadModel: (
    pathOrUrl: string,
    config?: LLMConfig,
    onDownloadProgress?: (progress: number) => void,
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
   * thinking) instead of a raw token stream. Requires `streamToolCalls: true`
   * in the load config for tool-call events to appear.
   */
  executeWithEvents: (
    parts: MultimodalPart[],
    onEvent: StreamEventCallback,
    options?: ExecuteOptions,
  ) => Promise<string>;
  /**
   * Create a logical conversation sharing the loaded engine. Multiple
   * conversations coexist without loading the model twice; switching between
   * them replays the target's transcript (re-prefill cost on the next
   * message). Inference calls are serialized once conversations are in use.
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
  let lastBudgetLevel: BudgetLevel = "ok";

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

  const augmentedLoadModel = async (
    pathOrUrl: string,
    config?: LLMConfig,
    onDownloadProgress?: (progress: number) => void,
  ) => {
    const modelPath = await ModelRegistry.resolveModel(pathOrUrl, {
      onProgress: onDownloadProgress,
    });

    // Pre-flight memory check: refuse loads that would likely OOM-kill the
    // app. `forceLoad: true` opts out.
    const fileSize = ModelRegistry.getFileSizeBytes(modelPath);
    if (fileSize >= 0 && !config?.forceLoad) {
      const usage = native.getMemoryUsage();
      const estimate = estimateMemory({
        modelFileSizeBytes: fileSize,
        availableMemoryBytes: usage.availableMemoryBytes,
        config,
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

    const result = await native.loadModel(modelPath, config);

    loadedModelPath = modelPath;
    loadedModelFileSize = fileSize;
    loadedConfig = config;
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

    if (onToken) {
      const wrapped: TokenCallback = (token, done) => {
        onToken(token, done);
        if (done) recordMemorySnapshot();
      };
      return native.execute(processedParts, wrapped, options);
    }
    return native
      .execute(processedParts, undefined, options)
      .then((result: string) => {
        recordMemorySnapshot();
        return result;
      });
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

  // ── Multi-conversation manager ─────────────────────────────────────────
  // One native context exists at a time; logical conversations are switched
  // by snapshotting the outgoing native history and replaying the incoming
  // transcript via resetConversation (re-prefill on the next message).
  // `null` identifies the default conversation (top-level execute calls).
  let conversationsInUse = false;
  let conversationSeq = 0;
  let activeConversationId: string | null = null;
  const savedTranscripts = new Map<string | null, Message[]>();
  let switchQueue: Promise<void> = Promise.resolve();

  const switchTo = (id: string | null, systemPrompt?: string) => {
    if (activeConversationId === id) return;
    savedTranscripts.set(activeConversationId, native.getHistory());
    const transcript = savedTranscripts.get(id) ?? [];
    native.resetConversation(
      transcript.length > 0 ? JSON.stringify(transcript) : undefined,
      systemPrompt,
    );
    activeConversationId = id;
  };

  /** Serialize `fn` behind every pending conversation operation. */
  const enqueue = <T>(fn: () => Promise<T>): Promise<T> => {
    const task = switchQueue.then(fn);
    switchQueue = task.then(
      () => {},
      () => {},
    );
    return task;
  };

  /**
   * Run `fn` with conversation `id` active. A no-op passthrough until the
   * first createConversation() — after that, all inference serializes through
   * one queue so a context switch can never interrupt a generation.
   */
  const withConversation = <T>(
    id: string | null,
    systemPrompt: string | undefined,
    fn: () => Promise<T>,
  ): Promise<T> => {
    if (!conversationsInUse) return fn();
    return enqueue(() => {
      switchTo(id, systemPrompt);
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
      execute: (parts, onToken, options_) =>
        withConversation(id, options?.systemPrompt, () => {
          guard();
          return runExecute(parts, onToken, options_);
        }),
      executeWithEvents: (parts, onEvent, options_) =>
        withConversation(id, options?.systemPrompt, () => {
          guard();
          return runExecuteWithEvents(parts, onEvent, options_);
        }),
      getHistory: () => {
        guard();
        return activeConversationId === id
          ? native.getHistory()
          : (savedTranscripts.get(id) ?? []);
      },
      release: () =>
        enqueue(async () => {
          if (released) return;
          // If this conversation holds the native context, hand it back to the
          // default conversation (replaying its transcript, if any).
          if (activeConversationId === id) {
            switchTo(null);
          }
          savedTranscripts.delete(id);
          released = true;
        }),
    };
  };

  const resetConversationManagerState = () => {
    savedTranscripts.clear();
    activeConversationId = null;
  };

  const unloadWrapped = async (): Promise<void> => {
    await native.unload();
    loadedModelPath = undefined;
    loadedModelFileSize = -1;
    loadedConfig = undefined;
    resetConversationManagerState();
    recordMemorySnapshot();
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
        return (historyJson?: string, systemPrompt?: string) => {
          // A manual reset clears the live context — ownership returns to the
          // default conversation.
          const result = target.resetConversation(historyJson, systemPrompt);
          activeConversationId = null;
          savedTranscripts.delete(null);
          recordMemorySnapshot();
          return result;
        };
      }

      if (isLegacyInferenceMethod(prop)) {
        return (...args: unknown[]) => {
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
