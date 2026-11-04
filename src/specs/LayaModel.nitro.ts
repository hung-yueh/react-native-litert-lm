import type { HybridObject } from "react-native-nitro-modules";

export interface LayaModelConfig {
  /**
   * Execution accelerator: "gpu" (with explicit FP32 precision) or "cpu" (multi-threaded).
   * Defaults to "gpu" with fallback to "cpu".
   */
  accelerator?: string;
  /**
   * Number of CPU threads when running on CPU (defaults to 4).
   */
  numThreads?: number;
  /**
   * Sequence token window length (defaults to 256).
   */
  window?: number;
  /**
   * File name of the main model graph (defaults to "laya_ml_s256_embeds_wfp16.tflite").
   */
  mainGraphName?: string;
  /**
   * File name of the action head graph (defaults to "laya_ml_act_head_fp32.tflite").
   */
  actGraphName?: string;
  /**
   * File name of the calibration file (defaults to "laya_ml_calibration.json").
   */
  calibrationFile?: string;
}

export interface LayaDecisionRequest {
  /**
   * State or context text (e.g. support email, conversation, raw text, or serialized JSON).
   */
  state?: string;
  /**
   * Question type: "choice", "score", or "noul".
   */
  type: string;
  /**
   * Question instructions (e.g. "What does the customer need?").
   */
  instructions: string;
  /**
   * JSON-serialized criteria mapping or array (e.g. '{"refund": "money returned", "help": "technical help"}').
   */
  criteria?: string;
  /**
   * Optional string options list (convenience alternative to criteria for choice questions).
   */
  options?: string[];
}

export interface LayaDecisionResponse {
  /**
   * Result type: "choice", "score", or "noul".
   */
  type: string;
  /**
   * For "choice": the selected label (argmax).
   */
  choice?: string;
  /**
   * For "score": expectation score (sum of i * p[i]).
   */
  score?: number;
  /**
   * For "noul": continuous probability of the affirmative statement (p[1]).
   */
  noul?: number;
  /**
   * For "noul": boolean decision (noul >= 0.5).
   */
  decision?: boolean;
  /**
   * Calibrated probability distribution for each option / level.
   */
  probabilities: Record<string, number>;
  /**
   * Normalized entropy confidence [0.0, 1.0].
   */
  confidence: number;
  /**
   * Action head escalation / action probability [0.0, 1.0].
   */
  actProbability: number;
  /**
   * Execution latency in milliseconds.
   */
  latencyMs: number;
}

export interface LayaModel extends HybridObject<{
  ios: "swift";
  android: "kotlin";
}> {
  /**
   * Load Laya model assets from a directory containing the model files:
   * main graph, act graph, token_embeddings_fp16.bin, tokenizer.json, and calibration.
   *
   * @param modelDir Absolute file path to the directory containing model files.
   * @param config Optional model configuration.
   */
  loadModel(modelDir: string, config?: LayaModelConfig): Promise<void>;

  /**
   * Execute a System-1 decision query.
   *
   * @param request The decision request containing state, type, instructions, and criteria.
   */
  decide(request: LayaDecisionRequest): Promise<LayaDecisionResponse>;

  /**
   * Checks whether the Laya model graphs and embeddings are loaded and ready.
   */
  isReady(): boolean;

  /**
   * Unloads the model graphs and memory-mapped table, freeing native memory.
   */
  unload(): Promise<void>;

  /**
   * Closes the instance and frees all native resources.
   */
  close(): void;
}
