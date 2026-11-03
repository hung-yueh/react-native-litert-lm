# Implementation Handoff: Laya Multilingual LiteRT System-1 Decision Model

## Execution Summary
- **Feature**: Laya Multilingual LiteRT System-1 Decision Model (`Laya-Multilingual-LiteRT`)
- **Branch**: `feature/laya-decision-model`
- **Plan Score**: 9.5/10
- **Estimated Effort**: 1 day
- **Critical Path**: Nitro Spec & Configuration (T1) -> Nitrogen Generation (T2) -> Native Kotlin Engine & Components (T3) & Swift Bridge (T4) -> TypeScript Runtime & Ergonomics (T5) -> Exports & Mocks (T6) -> Tests & Verification (T7)

## Architecture Decision Map
1. **Nitro HybridObject Core**:
   - `LayaModel` specification defines model initialization (`loadModel`), execution (`decide`), readiness checks (`isReady`), and teardown (`unload`, `close`).
   - Keeps heavy C++/JNI bindings minimal and focused strictly on graph execution and structured response return.
2. **Native Runtime with LiteRT**:
   - Android Kotlin implementation directly instantiates `com.google.ai.edge.litert.CompiledModel`.
   - Uses `Accelerator.GPU` with `CompiledModel.GpuOptions.Precision.FP32` by default, falling back gracefully to CPU (4 threads) if GPU initialization fails.
   - Embeddings lookup via `LayaEmbeddings`: reads the 393.2 MB `token_embeddings_fp16.bin` table using memory mapping (`FileChannel.MapMode.READ_ONLY`), expanding float16 half values to float32 on the host. Pads sequences with the table's actual PAD row (id 0).
   - Pure Kotlin `LayaTokenizer` parses `tokenizer.json` to handle byte-fallback BPE with 256,000 tokens with zero native overhead.
   - `LayaPromptBuilder` sequences prompt tokens with explicit special tokens (CLS 2, SEP 1, MASK 4) and tracks option `<mask>` marker positions.
   - `LayaDecoder` gathers raw marker logits, runs the action head (`laya_ml_act_head_fp32.tflite`) with uncalibrated features, applies option-bucket temperatures from `laya_ml_calibration.json`, and outputs Python-compatible 4-decimal rounded results.
3. **TypeScript Ergonomics**:
   - High-level `createLayaModel()` factory wraps the hybrid object.
   - Provides convenient, type-safe decision methods:
     - `decideChoice(state, question, options)` -> `{ type: 'choice', choice, probabilities, confidence, actProbability, latencyMs }`
     - `decideScore(state, question, levels)` -> `{ type: 'score', score, legend, probabilities, confidence, actProbability, latencyMs }`
     - `decideBool(state, question, criteria?)` -> `{ type: 'bool', decision, probability, confidence, actProbability, latencyMs }`
     - `decide(request)` -> polymorphic union result.

## File-by-File Implementation Guide

### File: `src/specs/LayaModel.nitro.ts`
- **Purpose**: Defines the Nitro HybridObject interface for Laya model inference.
- **Changes**: Add `LayaModel`, `LayaModelConfig`, `LayaDecisionRequest`, and `LayaDecisionResponse` interfaces.
- **Integration Points**: Registered in `nitro.json`.

### File: `nitro.json`
- **Purpose**: Configuration for Nitro Modules code generation and autolinking.
- **Changes**: Register `"LayaModel"` mapping to Swift (`HybridLayaModel`) and Kotlin (`HybridLayaModel`).

### File: `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridLayaModel.kt`
- **Purpose**: Android Kotlin native runtime for Laya model using Google LiteRT.
- **Changes**: Implement `HybridLayaModelSpec` using `CompiledModel`, orchestrating `LayaEmbeddings`, `LayaTokenizer`, `LayaPromptBuilder`, and `LayaDecoder`.

### File: `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/laya/`
- **Purpose**: Vendored and adapted Kotlin components from `litert-community/Laya-Multilingual-LiteRT` and `litert-samples`:
  - `LayaTokenizer.kt`: Pure Kotlin BPE tokenizer.
  - `LayaEmbeddings.kt`: Memory-mapped table reader.
  - `LayaPromptBuilder.kt`: Prompt construction & option squeezing.
  - `LayaDecoder.kt`: Softmax, calibration, and typed response formatting.
  - `LayaCalibration.kt`: Temperature buckets.

### File: `ios/HybridLayaModel.swift`
- **Purpose**: iOS Swift native stub conforming to `HybridLayaModelSpec`.
- **Changes**: Provide `HybridLayaModel` implementing required methods with descriptive errors/stubs when run on iOS.

### File: `src/layaModel.ts`
- **Purpose**: High-level TypeScript client library for Expo / React Native applications.
- **Changes**: Provide `createLayaModel`, prompt formatting, helper methods (`decideChoice`, `decideScore`, `decideBool`), and type definitions.

### File: `src/index.ts`
- **Purpose**: Library public entry point.
- **Changes**: Re-export `createLayaModel`, types (`LayaModelInstance`, `LayaChoiceResult`, `LayaScoreResult`, `LayaBoolResult`, `LayaDecisionResult`).

### File: `src/__mocks__/react-native-nitro-modules.ts`
- **Purpose**: Jest test environment mock for Nitro Modules.
- **Changes**: Add `HybridLayaModel` mock supporting test decisions.

### File: `src/__tests__/layaModel.test.ts`
- **Purpose**: Unit test suite for Laya model public interface and decision workflows.
- **Changes**: Test prompt rendering, choice/score/bool resolution, calibration, and error handling.

## Implementation Sequence
1. **T1**: Create `src/specs/LayaModel.nitro.ts` and update `nitro.json`.
2. **T2**: Run `npx nitrogen` to generate C++, Kotlin, and Swift binding files.
3. **T3**: Implement `android/.../HybridLayaModel.kt` and `laya/` components.
4. **T4**: Implement `ios/HybridLayaModel.swift` for the iOS target.
5. **T5**: Build `src/layaModel.ts` implementing high-level APIs and question types.
6. **T6**: Re-export in `src/index.ts` and update `src/__mocks__/react-native-nitro-modules.ts`.
7. **T7**: Add test suite in `src/__tests__/layaModel.test.ts` and verify with `npm test`.

## Validation Checklist
- [ ] `src/specs/LayaModel.nitro.ts` compiles cleanly with TypeScript.
- [ ] `npx nitrogen` executes without errors and generates spec classes.
- [ ] Android Kotlin files compile cleanly against `com.google.ai.edge.litert`.
- [ ] `src/layaModel.ts` correctly formats choice, score, and bool decision requests.
- [ ] `npm test` passes all tests with high coverage.

## Performance Targets
- Graph execution latency: < 60ms on mobile GPU.
- Host embedding lookup: < 15ms via memory mapping.
- Memory: Read-only memory-mapped token table without duplicating 393 MB in JVM heap.

## Reference Links
- Full Plan: `.agents/plans/laya-decision-model/plan.md`
- Graph Specification: `.agents/plans/laya-decision-model/graph.json`
- Active Context: `.agents/core/activeContext.md`
- Hugging Face Model: https://huggingface.co/litert-community/Laya-Multilingual-LiteRT
- LiteRT Model Zoo Reference: https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert_model_zoo/android/app/src/main/java/com/google/ai/edge/examples/litert_model_zoo/models/typed_decisions/laya
