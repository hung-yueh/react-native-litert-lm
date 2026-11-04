# Feature Plan: Laya Multilingual LiteRT System-1 Decision Model

## 1. Feature Overview
- **Description**: Integrate the Laya Multilingual LiteRT model family (`litert-community/Laya-Multilingual-LiteRT`) into `react-native-litert-lm`. Provide high-performance, on-device System-1 decision making (sub-60ms) across 100+ languages for the three standard decision types: `choice` (categorical selection), `score` (ordinal evaluation), and `bool`/`noul` (binary hypothesis verification).
- **Goals**:
  - Provide a Nitro HybridObject specification (`LayaModel.nitro.ts`) and register it in `nitro.json`.
  - Implement the Android native runtime in Kotlin (`HybridLayaModel.kt`) leveraging Google LiteRT (`com.google.ai.edge.litert:litert`), supporting `CompiledModel` with GPU (`FP32` precision) and CPU fallback, memory-mapped token table gather (`LayaEmbeddings`), pure Kotlin BPE tokenization (`LayaTokenizer`), prompt construction (`LayaPromptBuilder`), act head execution (`laya_ml_act_head_fp32.tflite`), and calibrated softmax decoding (`LayaDecoder`).
  - Implement an iOS native fallback / stub (`HybridLayaModel.swift`) conforming to `HybridLayaModelSpec`.
  - Provide high-level TypeScript ergonomics (`src/layaModel.ts` and `src/index.ts` export) including `createLayaModel`, helper methods (`decideChoice`, `decideScore`, `decideBool`, `decide`), automatic JSON schema serialization, temperature calibration, and structured decision outputs.
  - Provide comprehensive unit tests, mock support in `src/__mocks__/react-native-nitro-modules.ts`, and documentation.
- **Scope Boundaries**:
  - Focuses on the multilingual LiteRT checkpoint (`laya_ml_s256_embeds_wfp16.tflite` with window $N=256$, with support for $N=512$ if configured).
  - Uses the memory-mapped `token_embeddings_fp16.bin` host lookup for fast GPU/CPU execution.
  - Supports GPU acceleration via LiteRT OpenCL/GPU delegate with explicit FP32 precision, and CPU fallback.
  - Handles the 3 standard decision outputs: `choice`, `score`, and `bool`/`noul`, plus the action head escalation probability (`act_probability`).
- **User Value**:
  - Enables sub-60ms System-1 reactive decision making on-device in React Native / Expo apps in English, Japanese, and 100+ other languages without cloud round-trips or heavy LLM autoregressive decoding.

## 2. Architecture & Design
- **Pattern(s) Used**:
  - Nitro HybridObject pattern for zero-overhead C++/JNI bindings between React Native JS and Android Kotlin / iOS Swift.
  - Two-stage graph pipeline:
    1. Main Encoder Graph: `laya_ml_s256_embeds_wfp16.tflite` takes gathered `inputs_embeds [1,N,768]`, `attention_mask [1,N]`, and `qtype_onehot [1,3]`. Emits `token_logits [1,N]` and `pooled_cls [1,768]`.
    2. Action Head Graph: `laya_ml_act_head_fp32.tflite` takes `pooled_cls [1,768]` and uncalibrated softmax distribution features `feats [1,4]` (`top1`, `top1 - top2`, `entropy`, `k / 255.0`). Emits `act_logits [1,2]`.
  - Host-side memory-mapped embedding table:
    - Reads row-major binary16 `token_embeddings_fp16.bin` (393.2 MB) directly without copying the table into JVM heap, expanding float16 entries to float32 into the model's `inputs_embeds` buffer. Real PAD row gathered for padding tokens (id 0).
  - High-level TypeScript layer:
    - Offers clean developer ergonomics: `decideChoice(state, question, options)`, `decideScore(state, question, levels)`, `decideBool(state, question, criteria?)`.
- **Component Structure**:
  - `src/specs/LayaModel.nitro.ts`: Nitro interface definition for `LayaModel`.
  - `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridLayaModel.kt`: Android Kotlin native engine.
  - `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/laya/`:
    - `LayaTokenizer.kt`: BPE tokenizer for 256,000 vocab `tokenizer.json`.
    - `LayaEmbeddings.kt`: Memory-mapped `token_embeddings_fp16.bin` reader and float16->float32 gather.
    - `LayaPromptBuilder.kt`: Prompt framing, option truncation/squeeze, marker recording.
    - `LayaDecoder.kt`: Softmax, NumPy-order sum, calibration temperatures, argmax/expectation/confidence.
    - `LayaCalibration.kt`: Temperature buckets (`2`, `3-5`, `6-10`, `11+`).
  - `ios/HybridLayaModel.swift`: iOS Swift stub implementation.
  - `src/layaModel.ts`: High-level TypeScript client with `createLayaModel`.
  - `src/__mocks__/react-native-nitro-modules.ts`: Mock for Jest tests.
  - `src/__tests__/layaModel.test.ts`: Unit test suite.
- **Data Models**:
  - `LayaModelConfig`: `{ modelDir: string; mainGraphName?: string; actGraphName?: string; accelerator?: 'gpu' | 'cpu'; numThreads?: number; window?: number; calibrationFile?: string }`
  - `LayaDecisionRequest`: `{ state?: string; type: 'choice' | 'score' | 'noul'; instructions: string; criteria?: Record<string, string> | string[] | Record<string, any>; options?: string[] }`
  - `LayaDecisionResult`: Union of `LayaChoiceResult`, `LayaScoreResult`, `LayaBoolResult`.
  - `LayaChoiceResult`: `{ type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number; actProbability: number; latencyMs: number }`
  - `LayaScoreResult`: `{ type: 'score'; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number; actProbability: number; latencyMs: number }`
  - `LayaBoolResult`: `{ type: 'bool'; decision: boolean; probability: number; confidence: number; actProbability: number; latencyMs: number }`

## 3. Technical Implementation
- **Tech Stack**:
  - Kotlin, Google LiteRT `CompiledModel` (`com.google.ai.edge.litert:litert:2.1.6`), React Native Nitro Modules, TypeScript, Jest.
- **Key Algorithms/Pipelines**:
  1. Sequence Builder:
     - Formats `<CLS> <type> question: <instructions> <SEP> <MASK> opt0 <MASK> opt1 ... <SEP> <state> <SEP>`.
     - Truncates options to 48 tokens max; if option budget < 16, squeezes options proportionally.
     - Maps PAD tokens with id 0.
  2. Embeddings Gather:
     - Memory-maps binary `[256000, 768]` float16 table.
     - Expands gathered token rows to float32 into `inputs_embeds [1, 256, 768]`.
  3. LiteRT Execution:
     - Compiles `laya_ml_s256_embeds_wfp16.tflite` with GPU (FP32 precision) or CPU (4 threads).
     - Feeds `inputs_embeds`, `attention_mask`, `qtype_onehot`.
     - Reads `token_logits` and `pooled_cls`.
     - Calculates `feats [1, 4]` from raw marker logits.
     - Runs `laya_ml_act_head_fp32.tflite` to obtain `act_logits [1, 2]`.
  4. Calibration & Decoding:
     - Temperature bucket lookup from `laya_ml_calibration.json`.
     - Calibrated softmax and Python-compatible rounding (4 decimal places).

## 4. Acceptance Criteria
- [x] Nitro specification `LayaModel.nitro.ts` created and registered in `nitro.json`.
- [x] Nitrogen code generation succeeds (`npx nitrogen`), generating Kotlin/Swift/C++ bridges.
- [x] Android Kotlin native implementation `HybridLayaModel.kt` executes main graph and act head with LiteRT `CompiledModel`.
- [x] iOS Swift bridge `HybridLayaModel.swift` compiles cleanly and satisfies `HybridLayaModelSpec`.
- [x] TypeScript public API `src/layaModel.ts` and exports in `src/index.ts` provide `createLayaModel()`, `decideChoice()`, `decideScore()`, `decideBool()`, and `decide()`.
- [x] Jest mocks updated in `src/__mocks__/react-native-nitro-modules.ts`.
- [x] Unit tests in `src/__tests__/layaModel.test.ts` verify all decision types, prompt construction, calibration, and error states.
- [x] Existing test suite (`npm test`) passes with 100% success.

## 5. Task Breakdown
- **T1**: Nitro Spec & Configuration — `src/specs/LayaModel.nitro.ts` & `nitro.json`
- **T2**: Nitrogen Autolinking & Bridge Generation — Run `npx nitrogen`
- **T3**: Android Kotlin Native Runtime & Components — `HybridLayaModel.kt` & `laya/` components
- **T4**: iOS Swift Native Bridge Implementation — `HybridLayaModel.swift`
- **T5**: TypeScript Interface & Ergonomics — `src/layaModel.ts` (`createLayaModel`, helper methods)
- **T6**: Public Exports & Mock Integration — `src/index.ts` and `src/__mocks__/react-native-nitro-modules.ts`
- **T7**: Unit Testing & Verification — `src/__tests__/layaModel.test.ts` and `npm test`

## 6. Risk Assessment
- **Risk 1**: Model directory assets missing or partially downloaded.
  - *Mitigation*: Validate file existence and file sizes during `loadModel()`, providing descriptive actionable error messages.
- **Risk 2**: GPU acceleration failure on unsupported devices.
  - *Mitigation*: Gracefully fall back to CPU with multi-threading (4 threads) if GPU compilation throws.
- **Risk 3**: Large token sequences exceeding window length $N=256$.
  - *Mitigation*: Follow upstream right-truncation rule for state text; validate that all option markers survive before inference.

## 7. Critique & Revisions
- **Compilation Summary**:
  - Total leaf tasks: 7
  - Max tree depth: 1
  - Concurrency waves: 4
  - Compiled At: 2026-11-03T16:15:00-06:00
  - Cycles: None detected
- Validated against official `convaiinnovations/laya` 0.3.4 host specification and Hugging Face LiteRT model card (`litert-community/Laya-Multilingual-LiteRT`).

## 8. Decision Points for User
1. **Asset Management**: Laya requires a bundle directory with `laya_ml_s256_embeds_wfp16.tflite`, `laya_ml_act_head_fp32.tflite`, `token_embeddings_fp16.bin`, `tokenizer.json`, and `laya_ml_calibration.json`. `loadModel(modelDir)` accepts a local directory containing these files.
2. **Decision Types**: The 3 standard System-1 types (`choice`, `score`, `bool`) are surfaced with dedicated TypeScript helper methods alongside the general `decide()` call.
