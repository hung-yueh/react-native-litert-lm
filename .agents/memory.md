# Project Memory

## Last Completed Task
LayaTokenizer BPE merge table null/malformed entry hardening: fixed fatal `NullPointerException: null cannot be cast to non-null type kotlin.collections.List<*>` on initialization by replacing unchecked casts with safe parsers (`entry as? List<*>`, `entry as? String` for space-separated pair formats, null guards, and safe vocabulary lookups). Rebuilt and packaged `react-native-litert-lm-0.7.5.tgz`.

## Architecture Notes
- LayaModel (`HybridLayaModelSpec`): Two-stage graph execution pipeline (main graph `laya_ml_s256_embeds_wfp16.tflite` + action head `laya_ml_act_head_fp32.tflite`) for sub-60ms System-1 reactive decision making in 100+ languages across `choice`, `score`, and `bool`/`noul` question types.
- Zero-copy host embedding table lookup (`LayaEmbeddings`): Memory-maps `token_embeddings_fp16.bin` (393.2 MB) read-only without duplicating the table in JVM heap memory.
- Laya Asset Bundle Layout: Requires 5 coordinated files in the model directory (`tokenizer.json`, `token_embeddings_fp16.bin`, `token_embeddings.json`, `laya_ml_s256_embeds_wfp16.tflite`, `laya_ml_act_head_fp32.tflite`, `laya_ml_calibration.json`).
- MetricModel (`HybridMetricModelSpec`): 13-dimension classification engine with INT8 quantization, hash-based tokenization, and fallback heuristics for low-latency metric scoring.
- LiteRT-LM updated to 0.19.0 across Android Maven dependencies, iOS frameworks, and C++ engine bindings.
- Forwarded `speculativeDecodingMode`, `maxCachedProducts`, and `cacheDir` from `LLMConfig` into `useModel` native config memoization.

## Environment / Config
- Node >= 22
- LiteRT-LM v0.19.0
- React Native Nitro Modules >= 0.37.1

## Dependencies Added
- None added to npm packages; bumped internal package version to 0.7.5.

## Known Blockers / Next Steps
- Kev LiteRT System-1 Decision Model implementation (KevModel spec, pair graph state prefill & question step caching).
