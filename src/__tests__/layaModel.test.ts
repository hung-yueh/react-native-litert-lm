import { createLayaModel } from "../layaModel";
import { mockLayaModel } from "../__mocks__/react-native-nitro-modules";

describe("LayaModel & Multilingual System-1 Decision Model", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLayaModel.isReady.mockReturnValue(true);
  });

  describe("Lifecycle and Configuration", () => {
    test("loads model and strips file:// URI prefix", async () => {
      const model = createLayaModel();
      await model.loadModel("file:///data/local/tmp/laya_model_dir", {
        accelerator: "gpu",
        numThreads: 4,
        window: 256,
      });

      expect(mockLayaModel.loadModel).toHaveBeenCalledWith(
        "/data/local/tmp/laya_model_dir",
        {
          accelerator: "gpu",
          numThreads: 4,
          window: 256,
        },
      );
    });

    test("checks readiness, unloads, and closes cleanly", async () => {
      const model = createLayaModel();
      expect(model.isReady()).toBe(true);

      await model.unload();
      expect(mockLayaModel.unload).toHaveBeenCalledTimes(1);

      model.close();
      expect(mockLayaModel.close).toHaveBeenCalledTimes(1);
    });
  });

  describe("Categorical Decision (decideChoice)", () => {
    test("evaluates choice with string array of options", async () => {
      const model = createLayaModel();
      const result = await model.decideChoice(
        "I was billed twice for my subscription this month. Please help me get a refund.",
        "What is the customer's intent?",
        ["refund", "tech_support", "cancellation"],
      );

      expect(mockLayaModel.decide).toHaveBeenCalledWith({
        state: "I was billed twice for my subscription this month. Please help me get a refund.",
        type: "choice",
        instructions: "What is the customer's intent?",
        criteria: JSON.stringify({ refund: null, tech_support: null, cancellation: null }),
        options: ["refund", "tech_support", "cancellation"],
      });

      expect(result.type).toBe("choice");
      expect(result.choice).toBe("refund");
      expect(result.probabilities).toBeDefined();
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.actProbability).toBe(1.0);
      expect(result.latencyMs).toBeGreaterThan(0);
    });

    test("evaluates choice with dictionary mapping label to description", async () => {
      const model = createLayaModel();
      const criteria = {
        refund: "money returned or charge reversed",
        tech_support: "troubleshooting a technical failure",
      };

      const result = await model.decideChoice(
        "My app keeps crashing on launch",
        "Select support category",
        criteria,
      );

      expect(mockLayaModel.decide).toHaveBeenCalledWith({
        state: "My app keeps crashing on launch",
        type: "choice",
        instructions: "Select support category",
        criteria: JSON.stringify(criteria),
        options: ["refund", "tech_support"],
      });

      expect(result.type).toBe("choice");
      expect(result.choice).toBe("refund");
      expect(result.probabilities).toBeDefined();
    });

    test("supports Japanese and multilingual text in choice decisions", async () => {
      const model = createLayaModel();
      const result = await model.decideChoice(
        "同じ支払いが二重に請求されました。返金をお願いします。",
        "What does the customer need?",
        { refund: "money returned", help: "technical help" },
      );

      expect(result.type).toBe("choice");
      expect(mockLayaModel.decide).toHaveBeenCalledWith(
        expect.objectContaining({
          state: "同じ支払いが二重に請求されました。返金をお願いします。",
          type: "choice",
        }),
      );
    });
  });

  describe("Ordinal Scoring (decideScore)", () => {
    test("evaluates ordinal score with an array of levels", async () => {
      const model = createLayaModel();
      const levels = ["Low", "Medium", "High", "Critical"];

      const result = await model.decideScore(
        "Database connection pool exhausted in production cluster",
        "Rate severity level",
        levels,
      );

      expect(mockLayaModel.decide).toHaveBeenCalledWith({
        state: "Database connection pool exhausted in production cluster",
        type: "score",
        instructions: "Rate severity level",
        criteria: JSON.stringify(levels),
      });

      expect(result.type).toBe("score");
      expect(result.score).toBe(2.0);
      expect(result.legend).toEqual({
        "0": "Low",
        "1": "Medium",
        "2": "High",
        "3": "Critical",
      });
      expect(result.probabilities["2"]).toBe(0.4);
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.actProbability).toBe(1.0);
    });

    test("evaluates ordinal score with a dictionary of level descriptions", async () => {
      const model = createLayaModel();
      const levelsDict = {
        "0": "Completely dissatisfied",
        "1": "Somewhat dissatisfied",
        "2": "Neutral",
        "3": "Satisfied",
      };

      const result = await model.decideScore(
        "The delivery was on time and package was intact",
        "Rate customer satisfaction",
        levelsDict,
      );

      expect(result.type).toBe("score");
      expect(result.legend["0"]).toBe("Completely dissatisfied");
      expect(result.legend["3"]).toBe("Satisfied");
    });
  });

  describe("Boolean Hypothesis Verification (decideBool)", () => {
    test("evaluates boolean statement with default false/true criteria", async () => {
      const model = createLayaModel();
      const result = await model.decideBool(
        "We need this fixed immediately, our business is halted!",
        "Is the customer expressing high urgency?",
      );

      expect(mockLayaModel.decide).toHaveBeenCalledWith({
        state: "We need this fixed immediately, our business is halted!",
        type: "noul",
        instructions: "Is the customer expressing high urgency?",
        criteria: undefined,
      });

      expect(result.type).toBe("bool");
      expect(result.decision).toBe(true);
      expect(result.probability).toBe(0.95);
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.actProbability).toBe(1.0);
    });

    test("evaluates boolean statement with custom criteria descriptions", async () => {
      const model = createLayaModel();
      const customCriteria = {
        false: "the customer is calm and not requesting a supervisor",
        true: "the customer is demanding escalation to management",
      };

      const result = await model.decideBool(
        "Let me speak to your manager right now!",
        "Does the customer demand escalation?",
        customCriteria,
      );

      expect(mockLayaModel.decide).toHaveBeenCalledWith({
        state: "Let me speak to your manager right now!",
        type: "noul",
        instructions: "Does the customer demand escalation?",
        criteria: JSON.stringify(customCriteria),
      });

      expect(result.type).toBe("bool");
      expect(result.decision).toBe(true);
    });
  });

  describe("Structured State Serialization & Raw Decision", () => {
    test("serializes object / JSON state representations", async () => {
      const model = createLayaModel();
      const stateObject = {
        user: "alice",
        orderId: 12345,
        messages: [
          { role: "customer", text: "Order not received" },
          { role: "agent", text: "Tracking says delivered yesterday" },
        ],
      };

      await model.decideChoice(
        stateObject,
        "Classify incident status",
        ["resolved", "in_progress", "disputed"],
      );

      expect(mockLayaModel.decide).toHaveBeenCalledWith(
        expect.objectContaining({
          state: JSON.stringify(stateObject),
        }),
      );
    });

    test("handles empty or null state safely", async () => {
      const model = createLayaModel();
      await model.decideChoice(
        null,
        "Pick default option",
        ["opt1", "opt2"],
      );

      expect(mockLayaModel.decide).toHaveBeenCalledWith(
        expect.objectContaining({
          state: "",
        }),
      );
    });

    test("passes raw request through decide() directly", async () => {
      const model = createLayaModel();
      const rawResponse = await model.decide({
        state: "Raw test state",
        type: "choice",
        instructions: "Raw test question",
        criteria: '{"a": null, "b": null}',
        options: ["a", "b"],
      });

      expect(rawResponse.type).toBe("choice");
      expect(rawResponse.confidence).toBe(0.95);
    });
  });
});
