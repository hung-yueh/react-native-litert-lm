import { NitroModules } from "react-native-nitro-modules";
import { LiteRTLM, LLMConfig } from "./specs/LiteRTLM.nitro";
import { createMemoryTracker, MemoryTracker } from "./memoryTracker";

/**
 * Extended LiteRT-LM instance with optional memory tracking and
 * augmented loadModel that accepts a download progress callback.
 */
export type LiteRTLMInstance = Omit<LiteRTLM, "loadModel"> & {
  memoryTracker?: MemoryTracker;
  loadModel: (
    pathOrUrl: string,
    config?: LLMConfig,
    onDownloadProgress?: (progress: number) => void,
  ) => Promise<void>;
};

/**
 * Creates a new LiteRT-LM inference engine instance.
 *
 * Optionally creates a native-backed memory tracker using
 * `NitroModules.createNativeArrayBuffer()` (v0.35+) for efficient
 * zero-copy memory usage tracking.
 *
 * @param options.enableMemoryTracking Enable automatic memory tracking (default: false)
 * @param options.maxMemorySnapshots Maximum number of memory snapshots to store (default: 256)
 */
export function createLLM(options?: {
  enableMemoryTracking?: boolean;
  maxMemorySnapshots?: number;
}): LiteRTLMInstance {
  const native = NitroModules.createHybridObject<LiteRTLM>("LiteRTLM");

  const enableTracking = options?.enableMemoryTracking ?? false;
  const tracker = enableTracking
    ? createMemoryTracker(options?.maxMemorySnapshots ?? 256)
    : undefined;

  /**
   * Record a real memory snapshot using OS-level APIs via getMemoryUsage().
   */
  const recordMemorySnapshot = () => {
    if (!tracker) return;
    try {
      const usage = native.getMemoryUsage();
      tracker.record({
        timestamp: Date.now(),
        nativeHeapBytes: usage.nativeHeapBytes,
        residentBytes: usage.residentBytes,
        availableMemoryBytes: usage.availableMemoryBytes,
      });
    } catch {
      // Ignore errors during memory tracking - it's non-critical
    }
  };

  const augmentedLoadModel = async (
    pathOrUrl: string,
    config?: LLMConfig,
    onDownloadProgress?: (progress: number) => void,
  ) => {
    let modelPath = pathOrUrl;

    // Check if it's a URL — enforce HTTPS for model downloads
    if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
      if (pathOrUrl.startsWith("http://")) {
        throw new Error(
          "Insecure HTTP URLs are not allowed for model downloads. " +
            "Use HTTPS instead: " +
            pathOrUrl.replace("http://", "https://"),
        );
      }

      // Extract filename from URL, stripping query parameters
      const urlWithoutQuery = pathOrUrl.split("?")[0];
      const fileName = urlWithoutQuery.split("/").pop();
      if (!fileName) {
        throw new Error(`Invalid model URL: ${pathOrUrl}`);
      }

      console.log(`Checking model at ${pathOrUrl}...`);
      modelPath = await native.downloadModel(
        pathOrUrl,
        fileName,
        (progress) => {
          onDownloadProgress?.(progress);
        },
      );
      console.log(`Model downloaded to: ${modelPath}`);
    }

    const result = await native.loadModel(modelPath, config);

    // Record initial memory snapshot after model load
    if (tracker) {
      tracker.reset();
      recordMemorySnapshot();
    }

    return result;
  };

  const SNAPSHOT_TRIGGERS = new Set([
    "sendMessage",
    "sendMessageWithImage",
    "sendMessageWithAudio",
    "resetConversation",
  ]);

  return new Proxy(native, {
    get(target, prop, receiver) {
      if (prop === "memoryTracker") {
        return tracker;
      }
      if (prop === "loadModel") {
        return augmentedLoadModel;
      }

      const original = Reflect.get(target, prop, receiver);
      if (typeof original !== "function") {
        return original;
      }

      if (prop === "sendMessageAsync") {
        return (message: string, onToken: (token: string, done: boolean) => void) => {
          return original.call(target, message, (token: string, done: boolean) => {
            onToken(token, done);
            if (done) {
              recordMemorySnapshot();
            }
          });
        };
      }

      if (prop === "sendMessageWithImageAsync") {
        return (message: string, imagePath: string, onToken: (token: string, done: boolean) => void) => {
          return original.call(target, message, imagePath, (token: string, done: boolean) => {
            onToken(token, done);
            if (done) {
              recordMemorySnapshot();
            }
          });
        };
      }

      if (prop === "sendMessageWithAudioAsync") {
        return (message: string, audioPath: string, onToken: (token: string, done: boolean) => void) => {
          return original.call(target, message, audioPath, (token: string, done: boolean) => {
            onToken(token, done);
            if (done) {
              recordMemorySnapshot();
            }
          });
        };
      }

      if (SNAPSHOT_TRIGGERS.has(prop as string)) {
        return async (...args: any[]) => {
          const result = await original.apply(target, args);
          recordMemorySnapshot();
          return result;
        };
      }

      return original.bind(target);
    },
  }) as unknown as LiteRTLMInstance;
}

