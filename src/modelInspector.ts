import { NitroModules } from "react-native-nitro-modules";
import type { ModelInspector } from "./specs/ModelInspector.nitro";
import { ModelRegistry } from "./modelRegistry";
import { LiteRTLMError } from "./core/errors";
import { toLiteRTLMError } from "./core/nativeError";
import {
  normalizeModelInfo,
  recommendBackend as recommendBackendFor,
  type BackendRecommendation,
  type DeviceInfo,
  type ModelInfo,
  type RecommendBackendOptions,
} from "./core/modelInfo";

let inspector: ModelInspector | null = null;
const native = (): ModelInspector =>
  (inspector ??= NitroModules.createHybridObject<ModelInspector>("ModelInspector"));

let cachedDevice: DeviceInfo | null = null;

/**
 * Facts about this device that decide which bundle variant and backend fit:
 * platform, OS, hardware model, SoC (Android 12+), total memory, whether a
 * GPU driver is present, and whether it is a simulator/emulator. Cached.
 */
export function getDeviceInfo(): DeviceInfo {
  if (!cachedDevice) {
    const d = native().getDeviceInfo();
    cachedDevice = { ...d, platform: d.platform === "android" ? "android" : "ios" };
  }
  return cachedDevice;
}

/** {@link getDeviceInfo}, or undefined if the native module is unavailable. */
export function tryGetDeviceInfo(): DeviceInfo | undefined {
  try {
    return getDeviceInfo();
  } catch {
    return undefined;
  }
}

/**
 * Read a `.litertlm` bundle's metadata — model type, input modalities,
 * context limit, declared backends, NPU target, sampler defaults, minimum
 * runtime — without creating an engine. Typically well under a second from
 * internal storage, but not free: for multimodal bundles the engine also reads
 * the vision encoder's sections (hundreds of MB, briefly in memory). Readings
 * are cached per file version, so the `loadModel()` that follows doesn't repeat it.
 *
 * Accepts a local path, a `file://` URI, or the URL of a model that is
 * already downloaded. It never downloads; call `ModelRegistry.resolveModel()`
 * first for remote models.
 *
 * @example
 * ```ts
 * const info = await inspectModel(localPath);
 * if (!info.modalities.vision) hideImagePicker();
 * const { backend, reason } = recommendBackend(info);
 * ```
 */
export async function inspectModel(pathOrUrl: string): Promise<ModelInfo> {
  const path = localPathFor(pathOrUrl);
  try {
    return normalizeModelInfo(path, await native().inspect(path));
  } catch (e) {
    throw toLiteRTLMError(e, "inspect");
  }
}

function localPathFor(pathOrUrl: string): string {
  const clean = pathOrUrl.startsWith("file://") ? pathOrUrl.slice(7) : pathOrUrl;
  if (/^https?:\/\//.test(clean)) {
    if (!ModelRegistry.isCached(clean)) {
      throw new LiteRTLMError(
        "MODEL_NOT_FOUND",
        `${clean} is not downloaded yet. inspectModel() reads local files only — ` +
          "download it with ModelRegistry.resolveModel() first.",
      );
    }
    return ModelRegistry.getFilePath(clean);
  }
  return clean;
}

/**
 * Pick the backend for a bundle on this device: the bundle's own preference
 * order (`info.backends.text`), minus what this device can't run — no GPU
 * driver, an NPU build for another SoC, NPU on iOS. Bundles that declare
 * nothing get CPU. Pass `options.device` to evaluate another device.
 *
 * @example
 * ```ts
 * const { backend, compatible, reason } = recommendBackend(await inspectModel(path));
 * if (!compatible) showError(reason);
 * await llm.loadModel(path, { backend });
 * ```
 */
export function recommendBackend(
  info: ModelInfo,
  options: RecommendBackendOptions = {},
): BackendRecommendation {
  return recommendBackendFor(info, { ...options, device: options.device ?? tryGetDeviceInfo() });
}
