import { NitroModules } from "react-native-nitro-modules";
import {
  LiteRTLM,
  LLMConfig,
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

  const unloadWrapped = async (): Promise<void> => {
    await native.unload();
    loadedModelPath = undefined;
    loadedModelFileSize = -1;
    loadedConfig = undefined;
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
        return runExecuteWithEvents;
      }
      if (prop === "unload") {
        return unloadWrapped;
      }
      if (prop === "execute") {
        return (
          parts: MultimodalPart[],
          onToken?: TokenCallback,
          options?: ExecuteOptions,
        ) => runExecute(parts, onToken, options);
      }
      if (prop === "resetConversation") {
        return () => {
          const result = target.resetConversation();
          recordMemorySnapshot();
          return result;
        };
      }

      if (isLegacyInferenceMethod(prop)) {
        return (...args: unknown[]) => {
          const route = routeLegacyInference(prop, args)!;
          const promise = runExecute(route.parts, route.onToken);
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
