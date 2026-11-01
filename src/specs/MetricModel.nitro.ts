import type { HybridObject } from "react-native-nitro-modules";

export interface MetricEvaluationInput {
  /**
   * Raw text to evaluate. When provided (without tokenIds),
   * the text is tokenized into token IDs and attention mask.
   */
  text?: string;
  /**
   * Explicit token IDs (e.g. hashed subwords or BERT tokens).
   */
  tokenIds?: number[];
  /**
   * Attention mask corresponding to tokenIds (1 for active tokens, 0 for padding).
   */
  attentionMask?: number[];
  /**
   * Maximum sequence length for padding/truncating (defaults to model's expected shape, e.g. 128 or 64).
   */
  maxLength?: number;
}

export interface MetricEvaluationResult {
  /**
   * Map of axis names (e.g., "Warmth", "Sentiment", "Directness", etc.)
   * to rescaled 0.0 - 1.0 floating point scores.
   */
  scores: Record<string, number>;
  /**
   * Array of 13 normalized [0.0, 1.0] float scores in canonical axis order.
   */
  values: number[];
  /**
   * Raw quantized int8 outputs [-128, 128] emitted by the model head.
   */
  rawValues: number[];
  /**
   * Evaluation latency in milliseconds.
   */
  latencyMs: number;
}

export interface MetricModelConfig {
  /**
   * Maximum sequence length (defaults to 128).
   */
  maxSequenceLength?: number;
  /**
   * Number of CPU threads for inference (defaults to 2).
   */
  numThreads?: number;
}

export interface MetricModel extends HybridObject<{
  ios: "swift";
  android: "kotlin";
}> {
  /**
   * Load a metric model (.tflite).
   * @param modelPath Absolute file path to the .tflite model.
   * @param config Optional configuration options.
   */
  loadModel(modelPath: string, config?: MetricModelConfig): Promise<void>;

  /**
   * Classify language along the 13 metric axes.
   * Quantized outputs in [-128, 128] are converted to [0.0, 1.0] floats.
   */
  evaluate(input: MetricEvaluationInput): Promise<MetricEvaluationResult>;

  /**
   * Check if the metric model is loaded and ready.
   */
  isReady(): boolean;

  /**
   * Unload the model and free native resources.
   */
  unload(): Promise<void>;

  /**
   * Release native resources and close the instance.
   */
  close(): void;
}
