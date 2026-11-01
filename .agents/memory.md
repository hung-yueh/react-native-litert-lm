# Project Memory

## Last Completed Task
Release v0.7.4 prep and Metric Classification Model integration: upgraded LiteRT-LM to v0.19.0, added 13-dimension MetricModel hybrid spec with Android (Kotlin) & iOS (Swift) implementations, forwarded new speculativeDecodingMode/maxCachedProducts/cacheDir options in useModel, built tarball react-native-litert-lm-0.7.4.tgz, and passed all unit and lint tests.

## Architecture Notes
- MetricModel (`HybridMetricModelSpec`): 13-dimension classification engine with INT8 quantization, hash-based tokenization, and fallback heuristics for low-latency metric scoring.
- LiteRT-LM updated to 0.19.0 across Android Maven dependencies, iOS frameworks, and C++ engine bindings.
- Forwarded `speculativeDecodingMode`, `maxCachedProducts`, and `cacheDir` from `LLMConfig` into `useModel` native config memoization.

## Environment / Config
- Node >= 22
- LiteRT-LM v0.19.0
- React Native Nitro Modules >= 0.37.1

## Dependencies Added
- None added to npm packages; bumped internal package version to 0.7.4.

## Known Blockers / Next Steps
- Continue Kev LiteRT System-1 Decision Model implementation (KevModel spec, pair graph state prefill & question step caching).
