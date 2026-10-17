# Implementation Handoff: Kev LiteRT System-1 Decision Model

## Execution Summary
- **Feature**: Kev LiteRT System-1 Decision Model (`Kev-0.8B-LiteRT`)
- **Branch**: `feature/kev-decision-model`
- **Plan Score**: 9.5/10
- **Estimated Effort**: 1 day
- **Critical Path**: Nitro Spec (T1) -> Nitrogen Gen (T2) -> Native Kotlin Engine (T3) & Swift Bridge (T4) -> TypeScript Runtime & Head Math (T5) -> Exports & Mocks (T6) -> Tests & Verification (T7)

## Architecture Decision Map
1. **Nitro HybridObject Core**:
   - `KevModel` specification defines the low-level model loader (`loadRowModel`, `loadPairModel`), execution methods (`readoutRow`, `runState`, `readoutBranch`), and lifecycle management (`close`, `isReady`).
   - Keeps heavy C++/JNI bindings minimal and focused strictly on compiled graph tensor execution.
2. **Native Runtime with LiteRT**:
   - Kotlin implementation directly instantiates `com.google.ai.edge.litert.CompiledModel`.
   - Uses `Accelerator.GPU` with `CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM` by default, falling back gracefully to CPU if GPU initialization fails.
   - For `KevPairGraph`, prefill output buffers are wired directly into question step input buffers with zero copying through the application layer.
3. **TypeScript Ergonomics & Host Contract**:
   - High-level `createKevModel()` factory wraps the hybrid object.
   - Handles text prompt construction, special token delimiters (STATE 248060, QUESTION 248061, OPTION_START 248049, OPTION_END 248050, DECIDE 248062, PAD 248044), and input sanitization (`<|name|>` -> `<¦name¦>`).
   - Implements the pointer head projection:
     $z_k = \frac{(h_{\text{opt}_k} \cdot W_k^T + b_k) \cdot (h_{\text{decide}} \cdot W_q^T + b_q)}{16 \cdot T}$ ($T = 2.3510958125672174$)
     followed by softmax and typed result formatting (`choice`, `noul`, `score`).

## File-by-File Implementation Guide

### File: `src/specs/KevModel.nitro.ts`
- **Purpose**: Defines the Nitro HybridObject interface for Kev model inference.
- **Changes**: Add `KevModel`, `KevRowConfig`, `KevPairConfig`, and readout response interfaces.
- **Integration Points**: Registered in `nitro.json`.

### File: `nitro.json`
- **Purpose**: Configuration for Nitro Modules code generation and autolinking.
- **Changes**: Register `"KevModel"` mapping to Swift (`HybridKevModel`) and Kotlin (`HybridKevModel`).

### File: `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridKevModel.kt`
- **Purpose**: Android Kotlin native runtime for Kev models using Google LiteRT.
- **Changes**: Implement `HybridKevModelSpec` using `CompiledModel`, `KevRowGraph`, and `KevPairGraph` with GPU options and buffer management.

### File: `ios/HybridKevModel.swift`
- **Purpose**: iOS Swift native stub / runtime conforming to `HybridKevModelSpec`.
- **Changes**: Provide `HybridKevModel` implementing required methods with descriptive errors/stubs when run on iOS without LiteRT iOS frameworks.

### File: `src/kevModel.ts`
- **Purpose**: High-level TypeScript client library for Expo / React Native applications.
- **Changes**: Provide `createKevModel`, prompt rendering for states/questions/options, delimiter injection, pointer head math, confidence calculations, and decision routing (`auto`, `row`, `pair`).

### File: `src/index.ts`
- **Purpose**: Library public entry point.
- **Changes**: Re-export `createKevModel`, types (`KevModelInstance`, `KevChoiceResult`, `KevNoulResult`, `KevScoreResult`), and delimiter constants.

### File: `src/__mocks__/react-native-nitro-modules.ts`
- **Purpose**: Jest test environment mock for Nitro Modules.
- **Changes**: Add `HybridKevModel` mock supporting test row readouts and state execution.

### File: `src/__tests__/kevModel.test.ts`
- **Purpose**: Unit test suite for Kev model public interface and decision workflows.
- **Changes**: Test prompt rendering, token delimiters, pointer head projection, softmax, confidence scoring, choice/noul/score resolution, and error handling.

## Implementation Sequence
1. **T1**: Create `src/specs/KevModel.nitro.ts` and update `nitro.json`.
2. **T2**: Run `npx nitrogen` to generate C++, Kotlin, and Swift binding files.
3. **T3**: Implement `android/.../HybridKevModel.kt` using `CompiledModel` and GPU acceleration.
4. **T4**: Implement `ios/HybridKevModel.swift` for the iOS target.
5. **T5**: Build `src/kevModel.ts` implementing high-level APIs, pointer head math, and question types.
6. **T6**: Re-export in `src/index.ts` and update `src/__mocks__/react-native-nitro-modules.ts`.
7. **T7**: Add test suite in `src/__tests__/kevModel.test.ts` and verify with `npm test`.

## Validation Checklist
- [ ] `src/specs/KevModel.nitro.ts` compiles cleanly with TypeScript.
- [ ] `npx nitrogen` executes without errors and generates spec classes.
- [ ] Android Kotlin files compile cleanly against `com.google.ai.edge.litert`.
- [ ] `src/kevModel.ts` correctly tokenizes delimiters and calculates softmax & confidence.
- [ ] `npm test` passes all tests with high coverage.

## Performance Targets
- Row graph readout latency: < 50ms on mobile GPU.
- Memory: Shared tensor buffer allocation in pair graph avoiding duplicate GPU memory usage.

## Reference Links
- Full Plan: `.agents/plans/kev-decision-model/plan.md`
- Graph Specification: `.agents/plans/kev-decision-model/graph.json`
- Active Context: `.agents/core/activeContext.md`
- Hugging Face Model: https://huggingface.co/litert-community/Kev-0.8B-LiteRT
