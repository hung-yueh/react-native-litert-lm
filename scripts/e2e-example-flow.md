# E2E flow — example app integration check

Scripted end-to-end pass over `example/` that exercises the JS↔native
integration no unit tier can see (this is the tier that would have caught the
`useModel` → `streamToolCalls` drop). Run it on a simulator/emulator before a
release, after Nitro/spec changes, or when touching `hooks.ts` /
`modelFactory.ts` / the example app.

Driveable manually or with any UI driver (Argent MCP / Maestro / XCUITest).
Each step lists the action and the **assertion** that makes it a test.

## Setup

```bash
npm run build
cd example && npm install && npx expo prebuild --clean
npx expo run:ios          # or: npx expo run:android (arm64 device/emulator)
```

Use **Gemma 3 1B** for speed if pre-seeded, otherwise the in-app Gemma 4 E2B
download (needs ~3 GB free disk + Wi-Fi).

## Flow

| # | Action | Assert |
|---|--------|--------|
| 1 | Launch app | Model picker renders; **pre-flight verdict chip** shows `safe`/`tight`/`critical` *before any download* (estimate path works with no model present) |
| 2 | Open the **Memory** tab | Live RSS value > 0 and updating ~1 Hz (`getMemoryUsage` polling); no crash on repeated tab switches |
| 3 | Start model download | Progress bar advances monotonically 0→100 % (`ModelRegistry` progress callback); app remains responsive while downloading |
| 4 | Kill the app mid-download, relaunch, start again | Download restarts cleanly; **no corrupt partial file is loaded** (`.tmp` staging) |
| 5 | Let load complete | `isReady`; memory panel shows post-load estimate + forecast (`memoryEstimate`, `memoryForecast` populated via the hook) |
| 6 | Send a text prompt | Tokens **stream incrementally** (not one blob); UI stays interactive during generation (JSI thread not blocked) |
| 7 | Send 3–4 more prompts | Context forecast (`remainingTokens`) decreases; RSS sparkline records new snapshots after each turn (ring-buffer tracker) |
| 8 | iOS only: enable **Tools** toggle, ask "What's the weather in Tokyo?" | A typed `toolCall` event renders in the tool panel (requires `streamToolCalls` to survive the hook — regression check for the v0.5.x bug) |
| 9 | Attach the bundled `test.jpeg`, ask "Describe this image" (multimodal model) | Non-empty description; no temp-file leak (`litert_buf_*` count in tmp dir returns to baseline) |
| 10 | Change **maxContextTokens** in settings | Hook reloads the model automatically (config-change effect); verdict re-computes for the new KV budget |
| 11 | Tap **Unload** | RSS drops by roughly the model size within seconds (deterministic teardown, not GC); UI returns to load screen without crash |
| 12 | Reload the (now cached) model | Load completes with **no download step** (cache hit); inference works again on the same instance |
| 13 | Background the app, trigger memory pressure (`xcrun simctl spawn booted memory_pressure -S -l critical` / `adb shell am send-trim-memory <pid> RUNNING_CRITICAL`) | Warning banner shows level `critical` (pressure-source → JS callback path) |
| 14 | Delete the model from the model manager | File gone from cache list; app returns to pristine state |

## Recording the flow (optional)

With Argent connected, record steps 1–12 once via `flow-start-recording` /
`flow-finish-recording` and replay with `flow-execute` for regression runs —
see `.claude/rules/argent.md`. Keep the recorded flow under `.argent/flows/`.

## Pass criteria

Every assertion holds, zero crashes, and peak RSS during the run stays within
the budget recorded in `memory-baseline.json` (±10 %) once baselines exist.
