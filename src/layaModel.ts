import { NitroModules } from "react-native-nitro-modules";
import type {
  LayaModel,
  LayaModelConfig,
  LayaDecisionRequest,
  LayaDecisionResponse,
} from "./specs/LayaModel.nitro";

export type {
  LayaModel,
  LayaModelConfig,
  LayaDecisionRequest,
  LayaDecisionResponse,
};

export interface LayaChoiceResult {
  type: "choice";
  /**
   * The predicted category label (argmax).
   */
  choice: string;
  /**
   * Probability distribution across all choice labels.
   */
  probabilities: Record<string, number>;
  /**
   * Normalized entropy confidence score in [0.0, 1.0].
   */
  confidence: number;
  /**
   * Escalation probability from the action head in [0.0, 1.0].
   */
  actProbability: number;
  /**
   * Total evaluation latency in milliseconds.
   */
  latencyMs: number;
}

export interface LayaScoreResult {
  type: "score";
  /**
   * Expected score expectation: sum(i * p[i]).
   */
  score: number;
  /**
   * Human-readable criteria description for each ordinal level.
   */
  legend: Record<string, string>;
  /**
   * Probability distribution across each level index ("0", "1", ...).
   */
  probabilities: Record<string, number>;
  /**
   * Normalized entropy confidence score in [0.0, 1.0].
   */
  confidence: number;
  /**
   * Escalation probability from the action head in [0.0, 1.0].
   */
  actProbability: number;
  /**
   * Total evaluation latency in milliseconds.
   */
  latencyMs: number;
}

export interface LayaBoolResult {
  type: "bool";
  /**
   * Boolean determination: true if affirmative probability >= 0.5.
   */
  decision: boolean;
  /**
   * Continuous probability of the affirmative statement (noul true probability) in [0.0, 1.0].
   */
  probability: number;
  /**
   * Confidence score in [0.0, 1.0].
   */
  confidence: number;
  /**
   * Escalation probability from the action head in [0.0, 1.0].
   */
  actProbability: number;
  /**
   * Total evaluation latency in milliseconds.
   */
  latencyMs: number;
}

export type LayaDecisionResult =
  | LayaChoiceResult
  | LayaScoreResult
  | LayaBoolResult;

export interface LayaModelInstance {
  /**
   * Load Laya model assets from a directory.
   *
   * @param modelDir Absolute directory path containing model files.
   * @param config Optional model configuration (accelerator, threads, window).
   */
  loadModel(modelDir: string, config?: LayaModelConfig): Promise<void>;

  /**
   * Categorical decision: picks one of several options.
   *
   * @param state The input text or state object (e.g. email, message, json data).
   * @param instructions The question instructions (e.g. "What does the customer need?").
   * @param options A list of label strings or a map of { label: description }.
   */
  decideChoice(
    state: string | any,
    instructions: string,
    options: string[] | Record<string, string>,
  ): Promise<LayaChoiceResult>;

  /**
   * Ordinal decision: scores the input on an ordinal scale (e.g. 0 to 4).
   *
   * @param state The input text or state object.
   * @param instructions The question instructions (e.g. "Rate the customer sentiment").
   * @param levels An ordered array of criteria descriptions (e.g. ["Terrible", "Poor", "Neutral", "Good", "Great"]).
   */
  decideScore(
    state: string | any,
    instructions: string,
    levels: string[] | Record<string, string>,
  ): Promise<LayaScoreResult>;

  /**
   * Boolean decision: evaluates whether a statement holds true (yes/no).
   *
   * @param state The input text or state object.
   * @param instructions The statement to verify (e.g. "Is the user expressing urgency?").
   * @param criteria Optional descriptions for { false: string, true: string }.
   */
  decideBool(
    state: string | any,
    instructions: string,
    criteria?: { false?: string; true?: string },
  ): Promise<LayaBoolResult>;

  /**
   * Low-level raw decision call using the native request format.
   */
  decide(request: LayaDecisionRequest): Promise<LayaDecisionResponse>;

  /**
   * Checks whether the model is loaded and ready for inference.
   */
  isReady(): boolean;

  /**
   * Unloads model graphs and memory-mapped tables, freeing native resources.
   */
  unload(): Promise<void>;

  /**
   * Closes the instance and frees all native resources.
   */
  close(): void;
}

function serializeState(state: any): string {
  if (state === undefined || state === null) {
    return "";
  }
  if (typeof state === "string") {
    return state;
  }
  return JSON.stringify(state);
}

/**
 * Creates a new Laya Multilingual LiteRT decision model instance.
 *
 * @example
 * ```typescript
 * import { createLayaModel } from 'react-native-litert-lm';
 *
 * const laya = createLayaModel();
 * await laya.loadModel('/path/to/laya_model_dir');
 *
 * // 1. Categorical choice
 * const intent = await laya.decideChoice(
 *   'I was charged twice for the same order, please issue a refund.',
 *   'What does the customer need?',
 *   { refund: 'money returned', technical_support: 'help with a technical issue' }
 * );
 * console.log(intent.choice); // "refund"
 * console.log(intent.confidence); // 0.958
 *
 * // 2. Ordinal score
 * const urgency = await laya.decideScore(
 *   'Server is down in production!',
 *   'Rate urgency level',
 *   ['Low', 'Medium', 'High', 'Critical']
 * );
 * console.log(urgency.score); // e.g. 3.0 (Critical)
 *
 * // 3. Boolean / noul
 * const needsRefund = await laya.decideBool(
 *   'I want my money back',
 *   'The customer is requesting a refund'
 * );
 * console.log(needsRefund.decision); // true
 *
 * laya.close();
 * ```
 */
export function createLayaModel(): LayaModelInstance {
  const native = NitroModules.createHybridObject<LayaModel>("LayaModel");

  return {
    loadModel: (modelDir: string, config?: LayaModelConfig) => {
      const cleanPath = modelDir.startsWith("file://")
        ? modelDir.substring(7)
        : modelDir;
      return native.loadModel(cleanPath, config);
    },

    decideChoice: async (
      state: string | any,
      instructions: string,
      options: string[] | Record<string, string>,
    ): Promise<LayaChoiceResult> => {
      let criteriaJson: string;
      let optionsList: string[] | undefined;

      if (Array.isArray(options)) {
        optionsList = options;
        const map: Record<string, null> = {};
        for (const opt of options) {
          map[opt] = null;
        }
        criteriaJson = JSON.stringify(map);
      } else {
        criteriaJson = JSON.stringify(options);
        optionsList = Object.keys(options);
      }

      const response = await native.decide({
        state: serializeState(state),
        type: "choice",
        instructions,
        criteria: criteriaJson,
        options: optionsList,
      });

      return {
        type: "choice",
        choice: response.choice ?? (optionsList[0] || ""),
        probabilities: response.probabilities,
        confidence: response.confidence,
        actProbability: response.actProbability,
        latencyMs: response.latencyMs,
      };
    },

    decideScore: async (
      state: string | any,
      instructions: string,
      levels: string[] | Record<string, string>,
    ): Promise<LayaScoreResult> => {
      let criteriaJson: string;
      const legend: Record<string, string> = {};

      if (Array.isArray(levels)) {
        criteriaJson = JSON.stringify(levels);
        levels.forEach((val, idx) => {
          legend[idx.toString()] = val;
        });
      } else {
        const sortedKeys = Object.keys(levels);
        const arrayValues = sortedKeys.map((k) => levels[k]);
        criteriaJson = JSON.stringify(arrayValues);
        sortedKeys.forEach((key, idx) => {
          legend[idx.toString()] = levels[key];
        });
      }

      const response = await native.decide({
        state: serializeState(state),
        type: "score",
        instructions,
        criteria: criteriaJson,
      });

      return {
        type: "score",
        score: response.score ?? 0,
        legend,
        probabilities: response.probabilities,
        confidence: response.confidence,
        actProbability: response.actProbability,
        latencyMs: response.latencyMs,
      };
    },

    decideBool: async (
      state: string | any,
      instructions: string,
      criteria?: { false?: string; true?: string },
    ): Promise<LayaBoolResult> => {
      const criteriaJson = criteria ? JSON.stringify(criteria) : undefined;

      const response = await native.decide({
        state: serializeState(state),
        type: "noul",
        instructions,
        criteria: criteriaJson,
      });

      const noulProb = response.noul ?? response.probabilities["true"] ?? 0;
      const decision = response.decision ?? noulProb >= 0.5;

      return {
        type: "bool",
        decision,
        probability: noulProb,
        confidence: response.confidence,
        actProbability: response.actProbability,
        latencyMs: response.latencyMs,
      };
    },

    decide: (request: LayaDecisionRequest) => native.decide(request),
    isReady: () => native.isReady(),
    unload: () => native.unload(),
    close: () => native.close(),
  };
}
