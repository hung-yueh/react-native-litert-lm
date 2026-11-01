import { NitroModules } from "react-native-nitro-modules";
import type {
  MetricModel,
  MetricEvaluationInput,
  MetricEvaluationResult,
  MetricModelConfig,
} from "./specs/MetricModel.nitro";

export type {
  MetricModel,
  MetricEvaluationInput,
  MetricEvaluationResult,
  MetricModelConfig,
};

export const METRIC_DIMENSION_COUNT = 13;

/**
 * Canonical index mapping for the 13 classification dimensions.
 * Index in tensor output -> Metric adjective.
 */
export const METRIC_AXES: readonly string[] = [
  "Warmth",                  // 0
  "Sentiment",               // 1
  "Directness",              // 2
  "Passive-Aggressive",      // 3
  "Chemistry",               // 4
  "Professional Impression", // 5
  "Urgency",                 // 6
  "Formality",               // 7
  "Tension",                 // 8
  "Empathy",                 // 9
  "Clarity",                 // 10
  "Assertiveness",           // 11
  "Playfulness",              // 12
] as const;

export type MetricAxis = (typeof METRIC_AXES)[number];

export const DEFAULT_MAX_SEQUENCE_LENGTH = 128;
export const PAD_TOKEN_ID = 0;
export const BOS_TOKEN_ID = 1;
export const EOS_TOKEN_ID = 2;
export const UNK_TOKEN_ID = 3;

/**
 * Rescales an int8 value in [-128, 128] to a floating-point score [0.0, 1.0].
 * -128 -> 0.0
 * 0 -> 0.5
 * 128 -> 1.0
 */
export function rescaleInt8ToFloat(val: number): number {
  const clamped = Math.max(-128, Math.min(128, val));
  return Math.max(0, Math.min(1, (clamped + 128) / 256));
}

/**
 * Deterministically maps a string token to a token ID in [4, vocabSize - 1].
 */
export function hashMetricToken(token: string, vocabSize = 30000): number {
  let hash = 5381;
  for (let i = 0; i < token.length; i++) {
    hash = (hash * 33) ^ token.charCodeAt(i);
  }
  const positive = Math.abs(hash);
  return (positive % (vocabSize - 4)) + 4;
}

/**
 * Deterministic hash-based subword tokenizer for metric classification models.
 */
export function tokenizeMetricText(
  text: string,
  maxLength = DEFAULT_MAX_SEQUENCE_LENGTH,
  vocabSize = 30000,
): { tokenIds: number[]; attentionMask: number[]; maxLength: number } {
  const effectiveMax = Math.max(4, maxLength);
  const cleaned = (text || "").trim().toLowerCase();

  if (!cleaned) {
    const tokenIds = new Array(effectiveMax).fill(PAD_TOKEN_ID);
    const attentionMask = new Array(effectiveMax).fill(0);
    if (effectiveMax >= 2) {
      tokenIds[0] = BOS_TOKEN_ID;
      tokenIds[1] = EOS_TOKEN_ID;
      attentionMask[0] = 1;
      attentionMask[1] = 1;
    }
    return { tokenIds, attentionMask, maxLength: effectiveMax };
  }

  const words = cleaned.match(/\w+|[^\s\w]+/g) || [];
  const maxWords = effectiveMax - 2;
  const boundedWords = words.slice(0, maxWords);

  const tokenIds: number[] = [BOS_TOKEN_ID];
  const attentionMask: number[] = [1];

  for (const word of boundedWords) {
    tokenIds.push(hashMetricToken(word, vocabSize));
    attentionMask.push(1);
  }

  tokenIds.push(EOS_TOKEN_ID);
  attentionMask.push(1);

  while (tokenIds.length < effectiveMax) {
    tokenIds.push(PAD_TOKEN_ID);
    attentionMask.push(0);
  }

  return { tokenIds, attentionMask, maxLength: effectiveMax };
}

export const defaultMetricTokenizer = {
  tokenize: tokenizeMetricText,
};

export interface MetricModelInstance {
  /**
   * Load a metric classification model (.tflite).
   * @param modelPath Absolute file path to the model file.
   * @param config Optional thread and sequence length settings.
   */
  loadModel(modelPath: string, config?: MetricModelConfig): Promise<void>;

  /**
   * Evaluates input language along the 13 classification axes.
   * Accepts either raw text string or a MetricEvaluationInput object.
   */
  evaluate(input: MetricEvaluationInput | string): Promise<MetricEvaluationResult>;

  /**
   * Semantic alias for evaluate().
   */
  classify(textOrInput: string | MetricEvaluationInput): Promise<MetricEvaluationResult>;

  /**
   * Checks whether the model is loaded and ready.
   */
  isReady(): boolean;

  /**
   * Unloads the model and releases native memory while keeping the instance reusable.
   */
  unload(): Promise<void>;

  /**
   * Closes the instance and frees all native resources.
   */
  close(): void;
}

/**
 * Creates a new Metric Model classifier instance.
 *
 * @example
 * ```typescript
 * import { createMetricModel } from 'react-native-litert-lm';
 *
 * const metricModel = createMetricModel();
 * await metricModel.loadModel('/path/to/metrics_model_int8.tflite');
 *
 * // Classify raw text
 * const result = await metricModel.classify('Thanks so much, really appreciate your help!');
 * console.log('Warmth:', result.scores['Warmth']); // e.g. 0.92
 * console.log('All values (0-1):', result.values);
 *
 * metricModel.close();
 * ```
 */
export function createMetricModel(): MetricModelInstance {
  const native = NitroModules.createHybridObject<MetricModel>("MetricModel");

  const runEvaluate = async (
    input: MetricEvaluationInput | string,
  ): Promise<MetricEvaluationResult> => {
    let prepared: MetricEvaluationInput;
    if (typeof input === "string") {
      prepared = { text: input };
    } else {
      prepared = input;
    }

    if (prepared.text && (!prepared.tokenIds || prepared.tokenIds.length === 0)) {
      const tokenized = tokenizeMetricText(prepared.text, prepared.maxLength);
      prepared = {
        ...prepared,
        tokenIds: tokenized.tokenIds,
        attentionMask: tokenized.attentionMask,
        maxLength: tokenized.maxLength,
      };
    }

    return native.evaluate(prepared);
  };

  return {
    loadModel: (modelPath: string, config?: MetricModelConfig) => {
      const cleanPath = modelPath.startsWith("file://")
        ? modelPath.substring(7)
        : modelPath;
      return native.loadModel(cleanPath, config);
    },
    evaluate: runEvaluate,
    classify: runEvaluate,
    isReady: () => native.isReady(),
    unload: () => native.unload(),
    close: () => native.close(),
  };
}
