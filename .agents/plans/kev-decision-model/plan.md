# Feature Plan: Kev LiteRT System-1 Decision Model

## 1. Feature Overview
- **Description**: Add full native and TypeScript support for Kev-0.8B (and Kev-4B), an on-device System-1 decision-making model family converted to LiteRT (`com.google.ai.edge.litert`). Expose a high-performance public interface callable from Expo / React Native apps.
- **Goals**:
  - Provide a Nitro HybridObject specification (`KevModel.nitro.ts`) and register it in `nitro.json`.
  - Implement the Android native runtime in Kotlin (`HybridKevModel.kt`) supporting both `KevRowGraph` (standalone row prefill) and `KevPairGraph` (shared-state prefill + question branch step) using LiteRT's `CompiledModel`, GPU acceleration (`FP16_WITH_FP32_ACCUM`), and tensor buffer recycling.
  - Implement an iOS native fallback / stub (`HybridKevModel.swift`) that conforms to `HybridKevModelSpec`.
  - Provide TypeScript ergonomics (`src/kevModel.ts` and `src/index.ts` export) including tokenizer handling, delimiter token injection (PAD 248044, STATE 248060, QUESTION 248061, OPTION_START 248049, OPTION_END 248050, DECIDE 248062), pointer head dot-product calculation with temperature scaling ($T=2.3510958$), softmax, and structured decision outputs (`choice`, `noul`, `score`).
  - Add comprehensive unit tests, mock support, and documentation.
- **Scope Boundaries**:
  - Focuses on Kev-0.8B LiteRT models (`*_rowprefill_*.tflite` and `*_sharedstate_*.tflite`).
  - Supports GPU acceleration via LiteRT OpenCL / GPU delegate where available with graceful CPU/fallback options.
  - Handles the 3 standard decision contracts: `noul` (yes/no), `choice` (categorical selection with confidence), and `score` (ordinal expectation with legend).
- **User Value**:
  - Enables sub-50ms System-1 reactive decision making in Expo/React Native apps without round-tripping to cloud LLMs or running heavy generative decoder autoregression.

## 2. Architecture & Design
- **Pattern(s) Used**:
  - HybridObject pattern via Nitro Modules for zero-overhead C++/JNI bindings to Kotlin and Swift.
  - Two-tier execution strategy:
    1. Low-level native graph executor: runs compiled LiteRT models on GPU / CPU, returning float32 hidden vectors for `decide` and `optionEnds`.
    2. High-level TypeScript layer: handles prompt framing, delimiters, subword/BPE tokenization, head math (dot-product projection, temperature, softmax, confidence), and request routing (`auto`, `row`, `pair`).
- **Component Structure**:
  - `src/specs/KevModel.nitro.ts`: HybridObject specification for `KevModel`.
  - `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridKevModel.kt`: Native Kotlin LiteRT execution layer wrapping `KevRowGraph` and `KevPairGraph`.
  - `ios/HybridKevModel.swift`: Swift implementation of `HybridKevModelSpec`.
  - `src/kevModel.ts`: High-level public Expo interface (`createKevModel`), pointer head math, delimiters, decision formatters.
  - `src/__mocks__/react-native-nitro-modules.ts`: Mock implementation for Jest.
  - `src/__tests__/kevModel.test.ts`: Complete unit test suite.
- **Data Models**:
  - `KevModelConfig`: `{ modelPath: string; pairPath?: string; length?: number; ls?: number; lq?: number; precision?: 'fp16_accum' | 'fp32'; accelerator?: 'gpu' | 'cpu'; constantTensorSharing?: boolean }`
  - `KevDecisionResult`: Union of `KevChoiceResult`, `KevNoulResult`, `KevScoreResult`.
  - `KevChoiceResult`: `{ type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number; latencyMs: number }`
  - `KevNoulResult`: `{ type: 'noul'; probability: number; decision: boolean; confidence: number; latencyMs: number }`
  - `KevScoreResult`: `{ type: 'score'; score: number; mode: number; legend: string; probabilities: number[]; confidence: number; latencyMs: number }`
- **Integration Points**:
  - Auto-linked via Nitro Modules in `nitro.json`.
  - Re-exported in `src/index.ts` for clean imports: `import { createKevModel } from 'react-native-litert-lm'`.
- **Dependencies**:
  - `com.google.ai.edge.litert:litert:2.1.6` (already present in `android/build.gradle`).

## 3. Technical Implementation
- **Tech Stack**:
  - Kotlin, LiteRT CompiledModel, Nitro Modules, TypeScript, Jest.
- **Key Algorithms/Patterns**:
  - Pointer head readout:
    $$z_k = \frac{(h_{\text{opt}_k} \cdot W_k^T + b_k) \cdot (h_{\text{decide}} \cdot W_q^T + b_q)}{16 \cdot T}$$
    where $T = 2.3510958125672174$, followed by numerical-stable softmax.
  - Confidence calculation:
    - Choice: $(p_{\max} - 1/K) / (1 - 1/K)$
    - Score: $\max(0, 1 - \mathbb{E}[|\text{level} - \text{mode}|] / D)$
  - Shared-state memory caching across multiple questions on the same state context (`runState` + sequential `readoutBranch`).
- **Performance Considerations**:
  - Zero-copy tensor buffer passing between `state_prefill` outputs and `question_step` inputs in `KevPairGraph`.
  - Direct float array readout without intermediate allocations.
- **Security Considerations**:
  - Caller text sanitize: rewrite `<|name|>` to `<¦name¦>` preventing delimiter injection.

## 4. Acceptance Criteria
- [ ] Nitro specification `KevModel.nitro.ts` created and added to `nitro.json`.
- [ ] Nitrogen code generation succeeds (`npx nitrogen`), generating Kotlin/Swift/C++ specs.
- [ ] Android Kotlin implementation `HybridKevModel.kt` correctly implements `KevRowGraph` and `KevPairGraph` with LiteRT CompiledModel.
- [ ] iOS Swift implementation `HybridKevModel.swift` compiles cleanly and satisfies `HybridKevModelSpec`.
- [ ] TypeScript public API `src/kevModel.ts` and exports in `src/index.ts` provide `createKevModel`, delimiter handling, prompt construction, and head math.
- [ ] Unit tests in `src/__tests__/kevModel.test.ts` verify delimiter injection, tokenization, pointer head calculations, routing, and decision outputs.
- [ ] Existing test suite (`npm test`) passes without regressions.

## 5. Task Breakdown
- **T1**: Nitro Spec & Configuration — `src/specs/KevModel.nitro.ts` & `nitro.json`
- **T2**: Nitrogen Autolinking & Bridge Generation — Run `npx nitrogen`
- **T3**: Android Native Implementation — `HybridKevModel.kt` with `KevRowGraph` & `KevPairGraph`
- **T4**: iOS Native Bridge Implementation — `HybridKevModel.swift`
- **T5**: TypeScript Interface & Decision Runtime — `src/kevModel.ts`, prompt rendering, delimiter sanitization, pointer head math
- **T6**: Public Exports & Mock Integration — `src/index.ts` and `src/__mocks__/react-native-nitro-modules.ts`
- **T7**: Unit Testing & Verification — `src/__tests__/kevModel.test.ts` and `npm test`

## 6. Risk Assessment
- **Risk 1**: LiteRT GPU acceleration availability on Android devices without OpenCL.
  - *Mitigation*: Support `accelerator: 'auto' | 'gpu' | 'cpu'`, falling back to CPU when GPU compilation fails.
- **Risk 2**: Model weights and pointer head parameters loading.
  - *Mitigation*: Provide default embedded pointer head weights for Kev-0.8B, plus allow custom weights in `KevModelConfig`.

## 7. Critique & Revisions
- Reviewed against LiteRT and Kev repository specification on HuggingFace. Verified tensor dimensions (d=1024 for 0.8B), layer state count (48 tensors: 18 GDN, 6 Attention), delimiter token IDs (248044, 248049, 248050, 248060, 248061, 248062).

## 8. Decision Points for User
1. Model loading strategy: Support both single row graphs (`.tflite`) and shared-state pair graphs (`.tflite`), with auto-routing based on prompt sequence length.
2. Pointer head evaluation: Implemented in TypeScript/native for instant computation without requiring an extra graph.
