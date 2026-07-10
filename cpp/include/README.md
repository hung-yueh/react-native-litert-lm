# LiteRT-LM Headers Fallback

This directory contains the LiteRT-LM C API headers (`litert_lm_engine.h`,
`capabilities_c.h`) used by the iOS C++ implementation. They are vendored
copies pinned to the LiteRT-LM version in `package.json` (`litertLm.version`).

## If Headers Are Missing

If you get compilation errors like `litert_lm_engine.h: No such file or directory`, re-download the headers (replace `v0.14.0` with the pinned version):

```bash
curl -sL "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/v0.14.0/c/engine.h" -o litert_lm_engine.h
curl -sL "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/v0.14.0/schema/capabilities/capabilities_c.h" -o capabilities_c.h
```

The expected directory structure:

```
cpp/include/
├── litert_lm_engine.h   # LiteRT-LM C API header (upstream c/engine.h)
├── capabilities_c.h     # Model capability probing (upstream schema/capabilities/capabilities_c.h)
├── stb_image.h          # Image loading for multimodal
└── README.md
```

## Note

On **Android**, headers are provided by the `litertlm-android` AAR via Prefab — this directory is only needed for the **iOS** build which uses the raw C API via the prebuilt XCFramework (which bundles its own copies of these headers).
