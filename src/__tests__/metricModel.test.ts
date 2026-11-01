import {
  createMetricModel,
  rescaleInt8ToFloat,
  hashMetricToken,
  tokenizeMetricText,
  METRIC_AXES,
  METRIC_DIMENSION_COUNT,
  PAD_TOKEN_ID,
  BOS_TOKEN_ID,
  EOS_TOKEN_ID,
} from "../metricModel";
import { mockMetricModel } from "../__mocks__/react-native-nitro-modules";

describe("MetricModel & 13-Dimension Classifier", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockMetricModel.isReady.mockReturnValue(true);
  });

  describe("Constants & Quantization Rescaling", () => {
    test("has 13 canonical metric dimensions matching specification", () => {
      expect(METRIC_DIMENSION_COUNT).toBe(13);
      expect(METRIC_AXES.length).toBe(13);
      expect(METRIC_AXES[0]).toBe("Warmth");
      expect(METRIC_AXES[1]).toBe("Sentiment");
      expect(METRIC_AXES[2]).toBe("Directness");
      expect(METRIC_AXES[3]).toBe("Passive-Aggressive");
      expect(METRIC_AXES[4]).toBe("Chemistry");
      expect(METRIC_AXES[5]).toBe("Professional Impression");
      expect(METRIC_AXES[6]).toBe("Urgency");
      expect(METRIC_AXES[7]).toBe("Formality");
      expect(METRIC_AXES[8]).toBe("Tension");
      expect(METRIC_AXES[9]).toBe("Empathy");
      expect(METRIC_AXES[10]).toBe("Clarity");
      expect(METRIC_AXES[11]).toBe("Assertiveness");
      expect(METRIC_AXES[12]).toBe("Playfulness");
    });

    test("rescales int8 [-128, 128] quantized values to [0.0, 1.0]", () => {
      expect(rescaleInt8ToFloat(-128)).toBeCloseTo(0.0);
      expect(rescaleInt8ToFloat(0)).toBeCloseTo(0.5);
      expect(rescaleInt8ToFloat(128)).toBeCloseTo(1.0);
      expect(rescaleInt8ToFloat(-64)).toBeCloseTo(0.25);
      expect(rescaleInt8ToFloat(64)).toBeCloseTo(0.75);

      // Clamp checks
      expect(rescaleInt8ToFloat(-200)).toBeCloseTo(0.0);
      expect(rescaleInt8ToFloat(200)).toBeCloseTo(1.0);
    });
  });

  describe("Subword Tokenization", () => {
    test("hashMetricToken produces deterministic integer token IDs", () => {
      const id1 = hashMetricToken("hello");
      const id2 = hashMetricToken("hello");
      expect(id1).toBe(id2);
      expect(id1).toBeGreaterThanOrEqual(4);
      expect(id1).toBeLessThan(30000);
    });

    test("tokenizeMetricText creates bounded sequence with BOS, EOS, and padding", () => {
      const result = tokenizeMetricText("hello world", 8);
      expect(result.tokenIds.length).toBe(8);
      expect(result.attentionMask.length).toBe(8);
      expect(result.tokenIds[0]).toBe(BOS_TOKEN_ID);
      expect(result.attentionMask[0]).toBe(1);
      expect(result.attentionMask[1]).toBe(1); // 'hello'
      expect(result.attentionMask[2]).toBe(1); // 'world'
      expect(result.tokenIds[3]).toBe(EOS_TOKEN_ID);
      expect(result.attentionMask[3]).toBe(1);
      // Padded slots
      expect(result.tokenIds[4]).toBe(PAD_TOKEN_ID);
      expect(result.attentionMask[4]).toBe(0);
      expect(result.tokenIds[7]).toBe(PAD_TOKEN_ID);
      expect(result.attentionMask[7]).toBe(0);
    });

    test("tokenizeMetricText handles empty strings safely", () => {
      const result = tokenizeMetricText("", 6);
      expect(result.tokenIds[0]).toBe(BOS_TOKEN_ID);
      expect(result.tokenIds[1]).toBe(EOS_TOKEN_ID);
      expect(result.attentionMask[0]).toBe(1);
      expect(result.attentionMask[1]).toBe(1);
      expect(result.tokenIds[2]).toBe(PAD_TOKEN_ID);
      expect(result.attentionMask[2]).toBe(0);
    });
  });

  describe("MetricModelInstance Lifecycle & Evaluation", () => {
    test("loads model and strips file:// prefix", async () => {
      const model = createMetricModel();
      await model.loadModel("file:///data/local/tmp/metrics_model_int8.tflite", {
        numThreads: 4,
        maxSequenceLength: 64,
      });

      expect(mockMetricModel.loadModel).toHaveBeenCalledWith(
        "/data/local/tmp/metrics_model_int8.tflite",
        { numThreads: 4, maxSequenceLength: 64 },
      );
    });

    test("evaluates raw text with auto-tokenization", async () => {
      const model = createMetricModel();
      const result = await model.evaluate("Thanks for your quick response!");

      expect(mockMetricModel.evaluate).toHaveBeenCalledWith(
        expect.objectContaining({
          text: "Thanks for your quick response!",
          tokenIds: expect.any(Array),
          attentionMask: expect.any(Array),
        }),
      );
      expect(result.scores.Warmth).toBe(0.5);
      expect(result.values.length).toBe(13);
      expect(result.rawValues.length).toBe(13);
      expect(result.latencyMs).toBeGreaterThan(0);
    });

    test("classify() acts as semantic alias for evaluate()", async () => {
      const model = createMetricModel();
      const result = await model.classify("Sounds good.");
      expect(mockMetricModel.evaluate).toHaveBeenCalled();
      expect(result.scores).toBeDefined();
    });

    test("passes pre-tokenized tokenIds and attentionMask without re-tokenizing", async () => {
      const model = createMetricModel();
      const customTokenIds = [1, 42, 99, 2, 0, 0];
      const customMask = [1, 1, 1, 1, 0, 0];

      await model.evaluate({
        tokenIds: customTokenIds,
        attentionMask: customMask,
      });

      expect(mockMetricModel.evaluate).toHaveBeenCalledWith({
        tokenIds: customTokenIds,
        attentionMask: customMask,
      });
    });

    test("isReady, unload, and close delegate cleanly", async () => {
      const model = createMetricModel();
      expect(model.isReady()).toBe(true);
      expect(mockMetricModel.isReady).toHaveBeenCalled();

      await model.unload();
      expect(mockMetricModel.unload).toHaveBeenCalled();

      model.close();
      expect(mockMetricModel.close).toHaveBeenCalled();
    });
  });
});
