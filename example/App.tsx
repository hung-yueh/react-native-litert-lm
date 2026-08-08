import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ActivityIndicator,
  Image,
  KeyboardAvoidingView,
  Modal,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from "react-native-safe-area-context";
import {
  useModel,
  estimateMemory,
  checkBackendSupport,
  checkMultimodalSupport,
  GEMMA_3N_E2B_IT_INT4,
  GEMMA_4_E2B_IT,
  type MemoryEstimate,
  type MemoryUsage,
  type StreamEvent,
} from "react-native-litert-lm";
import {
  EmptyState,
  MessageBubble,
  ShimmerText,
  type ChatMsg,
} from "./src/components/ChatView";
import { MemoryPanel } from "./src/components/MemoryPanel";
import { Pill, ProgressBar, SectionLabel } from "./src/components/ui";
import { fmtBytes } from "./src/format";
import { T, VERDICT_COLORS } from "./src/theme";

// ─── Assets ──────────────────────────────────────────────────────────────────
const TEST_IMAGE_ASSET = require("./test.jpeg");
const TEST_AUDIO_ASSET = require("./test.wav");

// ─── Models ──────────────────────────────────────────────────────────────────
const MODELS = {
  gemma3n: {
    label: "Gemma 3n E2B",
    size: "3.7 GB",
    sizeBytes: 3_655_827_456,
    url: GEMMA_3N_E2B_IT_INT4,
    fileName: "gemma-3n-E2B-it-int4.litertlm",
  },
  gemma4: {
    label: "Gemma 4 E2B",
    size: "2.6 GB",
    sizeBytes: 2_580_000_000,
    url: GEMMA_4_E2B_IT,
    fileName: "gemma-4-E2B-it.litertlm",
  },
} as const;
type ModelKey = keyof typeof MODELS;

const CONTEXT_SIZES = [1024, 4096, 8192] as const;

const WEATHER_TOOL = {
  name: "get_current_weather",
  description: "Get the current weather for a location",
  parametersJson: JSON.stringify({
    type: "object",
    properties: {
      location: { type: "string", description: "City, e.g. San Francisco" },
      unit: { type: "string", enum: ["celsius", "fahrenheit"] },
    },
    required: ["location"],
  }),
};

const CIRCLE = 42;

// ═════════════════════════════════════════════════════════════════════════════
export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <Main />
    </SafeAreaProvider>
  );
}

function Main() {
  // ── State ──────────────────────────────────────────────────────────────────
  const insets = useSafeAreaInsets();
  const [sel, setSel] = useState<ModelKey>("gemma3n");
  const [backend, setBackend] = useState<"cpu" | "gpu">("cpu");
  const [contextTokens, setContextTokens] = useState<number>(4096);
  const [enableSpeculativeDecoding, setEnableSpeculativeDecoding] =
    useState(false);
  const [enableTools, setEnableTools] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [chat, setChat] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState<ChatMsg | null>(null);
  const [busy, setBusy] = useState(false);
  const [attachment, setAttachment] = useState<"image" | "audio" | null>(null);
  const [liveUsage, setLiveUsage] = useState<MemoryUsage | null>(null);
  const [forceLoad, setForceLoad] = useState(false);
  const forceLoadPending = useRef(false);
  const [showScrollDown, setShowScrollDown] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  const config = useMemo(
    () => ({
      backend,
      systemPrompt: "You are a helpful assistant. Keep responses concise.",
      maxContextTokens: contextTokens,
      maxOutputTokens: 1024,
      autoLoad: false,
      enableMemoryTracking: true,
      maxMemorySnapshots: 100,
      enableSpeculativeDecoding,
      streamToolCalls: enableTools,
      tools: enableTools ? [WEATHER_TOOL] : undefined,
      ...(forceLoad && { forceLoad }),
    }),
    [backend, contextTokens, enableSpeculativeDecoding, enableTools, forceLoad],
  );

  const {
    model,
    isReady,
    downloadProgress,
    error,
    load,
    deleteModel,
    memorySummary,
    memoryEstimate,
    memoryForecast,
    memoryWarning,
  } = useModel(MODELS[sel].url, config);

  // ── Pre-flight estimate before anything is downloaded ─────────────────────
  const preflight: MemoryEstimate | null = useMemo(() => {
    try {
      const available =
        liveUsage?.availableMemoryBytes ??
        model?.getMemoryUsage().availableMemoryBytes;
      if (!available) return null;
      return estimateMemory({
        modelFileSizeBytes: MODELS[sel].sizeBytes,
        availableMemoryBytes: available,
        config: { backend, maxContextTokens: contextTokens },
      });
    } catch {
      return null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel, backend, contextTokens, liveUsage, isReady]);

  const refreshUsage = useCallback(() => {
    try {
      if (model) setLiveUsage(model.getMemoryUsage());
    } catch {}
  }, [model]);

  useEffect(() => {
    refreshUsage();
  }, [refreshUsage, isReady]);

  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 80);
  }, [chat, streaming]);

  // "Try anyway" flips forceLoad into the config; retry once it has landed.
  useEffect(() => {
    if (forceLoad && forceLoadPending.current) {
      forceLoadPending.current = false;
      load();
    }
  }, [forceLoad, load]);

  // ── Send (typed streaming events) ──────────────────────────────────────────
  const send = useCallback(async () => {
    if (!model || busy) return;
    const msg = input.trim();
    if (!msg && !attachment) return;

    setInput("");
    setBusy(true);
    const currentAttachment = attachment;
    setAttachment(null);

    const displayMsg = currentAttachment
      ? `[${currentAttachment === "image" ? "🖼" : "🎧"}] ${msg}`
      : msg;
    setChat((prev) => [
      ...prev,
      { role: "user", text: displayMsg, ts: Date.now() },
    ]);

    const acc: ChatMsg = { role: "model", text: "", ts: Date.now() };
    setStreaming({ ...acc });

    try {
      const parts: any[] = [];
      if (currentAttachment === "image") {
        const uri = Image.resolveAssetSource(TEST_IMAGE_ASSET).uri;
        parts.push({
          type: "image",
          imageBuffer: await (await fetch(uri)).arrayBuffer(),
        });
      } else if (currentAttachment === "audio") {
        const uri = Image.resolveAssetSource(TEST_AUDIO_ASSET).uri;
        parts.push({
          type: "audio",
          audioBuffer: await (await fetch(uri)).arrayBuffer(),
        });
      }
      if (msg) parts.push({ type: "text", text: msg });

      // executeWithEvents: raw token stream → typed token/toolCall/thinking
      await model.executeWithEvents(parts, (event: StreamEvent) => {
        if (event.type === "token") acc.text += event.text;
        else if (event.type === "thinking")
          acc.thinking = (acc.thinking ?? "") + event.text;
        else if (event.type === "toolCall")
          acc.toolCall = (acc.toolCall ?? "") + event.text;
        if (!event.done) setStreaming({ ...acc });
      });

      acc.text = acc.text.trim();
      try {
        const stats = model.getStats();
        if (stats?.tokensPerSecond) {
          acc.meta = `${stats.tokensPerSecond.toFixed(1)} tok/s · ${(
            stats.timeToFirstToken * 1000
          ).toFixed(0)} ms to first token`;
        }
      } catch {}
      setChat((prev) => [...prev, { ...acc }]);
      setStreaming(null);
      refreshUsage();
    } catch (e: any) {
      setChat((prev) => [
        ...prev,
        { role: "model", text: `Error: ${e.message}`, error: true, ts: Date.now() },
      ]);
      setStreaming(null);
    } finally {
      setBusy(false);
    }
  }, [model, input, attachment, busy, refreshUsage]);

  const unload = useCallback(async () => {
    try {
      await model?.unload();
      setChat([]);
      refreshUsage();
    } catch {}
  }, [model, refreshUsage]);

  const onScroll = useCallback(
    (e: NativeSyntheticEvent<NativeScrollEvent>) => {
      const { contentOffset, contentSize, layoutMeasurement } = e.nativeEvent;
      const fromBottom =
        contentSize.height - layoutMeasurement.height - contentOffset.y;
      setShowScrollDown(fromBottom > 120);
    },
    [],
  );

  // ── Derived ────────────────────────────────────────────────────────────────
  const isDownloading = downloadProgress > 0 && downloadProgress < 1;
  // A failed load leaves downloadProgress at 1 — treat error as "not loading"
  // so the Load button comes back for a retry.
  const isLoading = downloadProgress === 1 && !isReady && !error;
  const canInteract = !isReady && !isDownloading && !isLoading;
  const gpuWarning = useMemo(() => checkBackendSupport("gpu"), []);
  const snapshots = model?.memoryTracker?.getSnapshots() ?? [];
  const errorIsMemory =
    error?.includes("MEMORY_ESTIMATE_EXCEEDED") ||
    error?.includes("Refusing to load");
  const canSend = (!!input.trim() || !!attachment) && !busy && isReady;

  return (
    <View style={[s.root, { paddingTop: insets.top }]}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        {/* ── Header ───────────────────────────────────────────────────────── */}
        <View style={s.header}>
          <TouchableOpacity
            testID="settings-btn"
            style={s.circleBtn}
            onPress={() => setShowSettings(true)}
          >
            <Text style={s.circleIcon}>⚙︎</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={s.namePill}
            onPress={() => setShowSettings(true)}
          >
            <Text style={s.nameText}>{MODELS[sel].label}</Text>
            <Text style={s.nameChevron}>⌄</Text>
          </TouchableOpacity>

          <TouchableOpacity
            testID="new-chat-btn"
            style={s.circleBtn}
            onPress={() => setChat([])}
          >
            <Text style={s.circleIcon}>✎</Text>
          </TouchableOpacity>
        </View>

        {/* ── Conversation ─────────────────────────────────────────────────── */}
        <ScrollView
          ref={scrollRef}
          style={{ flex: 1 }}
          contentContainerStyle={s.chatContent}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="interactive"
          onScroll={onScroll}
          scrollEventThrottle={64}
        >
          {!isReady && (
            <SetupCard
              label={MODELS[sel].label}
              size={MODELS[sel].size}
              backend={backend}
              contextTokens={contextTokens}
              preflight={preflight}
              isDownloading={isDownloading}
              isLoading={isLoading}
              downloadProgress={downloadProgress}
              error={error}
              errorIsMemory={!!errorIsMemory}
              canInteract={canInteract}
              onLoad={load}
              onForceLoad={() => {
                if (forceLoad) {
                  load();
                } else {
                  forceLoadPending.current = true;
                  setForceLoad(true);
                }
              }}
            />
          )}

          {isReady && chat.length === 0 && !streaming && (
            <EmptyState
              title="Ready when you are"
              sub={`${MODELS[sel].label} on ${backend.toUpperCase()} — everything stays on this device.`}
            >
              <View style={s.suggestRow}>
                {[
                  "Explain LoRA adapters in one line",
                  "Tell me a joke",
                  enableTools
                    ? "What's the weather in Tokyo?"
                    : "Write a haiku about RAM",
                ].map((q) => (
                  <TouchableOpacity
                    key={q}
                    style={s.suggestChip}
                    onPress={() => setInput(q)}
                  >
                    <Text style={s.suggestText}>{q}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </EmptyState>
          )}

          {chat.map((m, i) => (
            <MessageBubble key={i} msg={m} />
          ))}
          {streaming && <MessageBubble msg={streaming} isStreaming />}
        </ScrollView>

        {/* ── Scroll-to-bottom chevron ─────────────────────────────────────── */}
        {showScrollDown && (
          <TouchableOpacity
            style={s.scrollDown}
            onPress={() =>
              scrollRef.current?.scrollToEnd({ animated: true })
            }
          >
            <Text style={s.scrollDownIcon}>⌄</Text>
          </TouchableOpacity>
        )}

        {/* ── Composer ─────────────────────────────────────────────────────── */}
        {isReady && (
          <View
            style={[s.composer, { paddingBottom: insets.bottom + 8 }]}
          >
            <TouchableOpacity
              testID="attach-btn"
              style={s.circleBtn}
              disabled={busy}
              onPress={() => {
                const warning = checkMultimodalSupport();
                if (warning) return;
                setAttachment(
                  attachment === null
                    ? "image"
                    : attachment === "image"
                      ? "audio"
                      : null,
                );
              }}
            >
              <Text style={s.plusIcon}>＋</Text>
            </TouchableOpacity>

            <View style={s.inputPill}>
              {attachment && (
                <View style={s.attachmentChip}>
                  <Text style={s.attachmentText}>
                    {attachment === "image" ? "🖼 test.jpeg" : "🎧 test.wav"}
                  </Text>
                  <TouchableOpacity
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                    onPress={() => setAttachment(null)}
                  >
                    <Text style={s.attachmentRemove}>✕</Text>
                  </TouchableOpacity>
                </View>
              )}
              <TextInput
                testID="chat-input"
                style={s.input}
                placeholder={busy ? "Generating…" : "Message"}
                placeholderTextColor={T.dim}
                value={input}
                onChangeText={setInput}
                editable={!busy}
                onSubmitEditing={send}
                returnKeyType="send"
                multiline
              />
            </View>

            <TouchableOpacity
              testID="send-btn"
              style={s.circleBtn}
              onPress={send}
              disabled={!canSend}
            >
              {busy ? (
                <Text style={[s.sendIcon, { color: T.faint, fontSize: 14 }]}>
                  ■
                </Text>
              ) : (
                <Text
                  style={[s.sendIcon, { color: canSend ? T.sendActive : T.faint }]}
                >
                  ↑
                </Text>
              )}
            </TouchableOpacity>
          </View>
        )}
      </KeyboardAvoidingView>

      {/* ── Settings sheet ─────────────────────────────────────────────────── */}
      <Modal
        visible={showSettings}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowSettings(false)}
      >
        <View style={s.sheetRoot}>
          <View style={s.sheetHeader}>
            <Text style={s.sheetTitle}>Settings</Text>
            <TouchableOpacity
              testID="settings-done"
              style={s.circleBtn}
              onPress={() => setShowSettings(false)}
            >
              <Text style={s.circleIcon}>✕</Text>
            </TouchableOpacity>
          </View>
          <ScrollView
            contentContainerStyle={{
              padding: 16,
              paddingBottom: insets.bottom + 24,
              gap: 18,
            }}
          >
            <View>
              <SectionLabel>Model</SectionLabel>
              <View style={s.pillRow}>
                {(Object.keys(MODELS) as ModelKey[]).map((k) => (
                  <Pill
                    key={k}
                    testID={`model-${k}`}
                    label={MODELS[k].label}
                    sub={MODELS[k].size}
                    active={sel === k}
                    disabled={!canInteract}
                    onPress={() => setSel(k)}
                  />
                ))}
              </View>
            </View>

            <View>
              <SectionLabel>Backend</SectionLabel>
              <View style={s.pillRow}>
                <Pill
                  label="CPU"
                  active={backend === "cpu"}
                  disabled={!canInteract}
                  onPress={() => setBackend("cpu")}
                />
                <Pill
                  label="GPU"
                  sub={gpuWarning ? "unavailable" : undefined}
                  active={backend === "gpu"}
                  disabled={!canInteract || !!gpuWarning}
                  onPress={() => setBackend("gpu")}
                />
              </View>
            </View>

            <View>
              <SectionLabel>Context window (KV-cache memory)</SectionLabel>
              <View style={s.pillRow}>
                {CONTEXT_SIZES.map((n) => (
                  <Pill
                    key={n}
                    testID={`ctx-${n}`}
                    label={`${n / 1024}k`}
                    active={contextTokens === n}
                    disabled={!canInteract}
                    onPress={() => setContextTokens(n)}
                  />
                ))}
              </View>
            </View>

            <View>
              <SectionLabel>Features</SectionLabel>
              <View style={s.pillRow}>
                <Pill
                  label="Speculative"
                  sub="multi-token"
                  active={enableSpeculativeDecoding}
                  disabled={!canInteract}
                  onPress={() =>
                    setEnableSpeculativeDecoding(!enableSpeculativeDecoding)
                  }
                />
                <Pill
                  label="Tools"
                  sub="streamed calls"
                  active={enableTools}
                  disabled={!canInteract}
                  onPress={() => setEnableTools(!enableTools)}
                />
              </View>
            </View>

            <View>
              <SectionLabel>Memory</SectionLabel>
              <MemoryPanel
                estimate={memoryEstimate ?? preflight}
                forecast={memoryForecast}
                usage={liveUsage}
                summary={memorySummary}
                snapshots={snapshots}
                warning={memoryWarning}
                isReady={isReady}
                onUnload={unload}
              />
            </View>

            {isReady && (
              <TouchableOpacity
                style={s.dangerBtn}
                onPress={() => deleteModel(MODELS[sel].fileName).catch(() => {})}
              >
                <Text style={s.dangerText}>Delete cached model</Text>
              </TouchableOpacity>
            )}
          </ScrollView>
        </View>
      </Modal>
    </View>
  );
}

// ─── Setup / load card shown before the model is ready ───────────────────────
function SetupCard({
  label,
  size,
  backend,
  contextTokens,
  preflight,
  isDownloading,
  isLoading,
  downloadProgress,
  error,
  errorIsMemory,
  canInteract,
  onLoad,
  onForceLoad,
}: {
  label: string;
  size: string;
  backend: string;
  contextTokens: number;
  preflight: MemoryEstimate | null;
  isDownloading: boolean;
  isLoading: boolean;
  downloadProgress: number;
  error: string | null;
  errorIsMemory: boolean;
  canInteract: boolean;
  onLoad: () => void;
  onForceLoad: () => void;
}) {
  // A critical estimate or a refused load both offer a forced retry.
  const risky = preflight?.verdict === "critical" || errorIsMemory;
  return (
    <View style={s.setup}>
      <Text style={s.setupMark}>
        litert<Text style={{ fontWeight: "400" }}>·lm</Text>
      </Text>
      <Text style={s.setupTitle}>{label}</Text>
      <Text style={s.setupSub}>
        {size} · {backend.toUpperCase()} · {contextTokens / 1024}k context ·
        on-device
      </Text>

      {preflight && canInteract && (
        <View style={s.preflightRow}>
          <View
            style={[
              s.verdictDot,
              { backgroundColor: VERDICT_COLORS[preflight.verdict].fg },
            ]}
          />
          <Text style={s.preflightText}>
            <Text
              style={{
                color: VERDICT_COLORS[preflight.verdict].fg,
                fontWeight: "600",
              }}
            >
              {VERDICT_COLORS[preflight.verdict].label}.
            </Text>{" "}
            Needs ~{fmtBytes(preflight.totalEstimatedBytes)} of{" "}
            {fmtBytes(preflight.availableBytes)} available.
          </Text>
        </View>
      )}

      {canInteract && (
        <TouchableOpacity
          testID="load-btn"
          style={[s.loadBtn, risky && { backgroundColor: T.error }]}
          onPress={risky ? onForceLoad : onLoad}
        >
          <Text style={[s.loadBtnText, risky && { color: "#fff" }]}>
            {risky ? "Try anyway" : "Load model"}
          </Text>
        </TouchableOpacity>
      )}

      {isDownloading && (
        <View style={s.progressWrap}>
          <ShimmerText
            text={`Downloading ${(downloadProgress * 100).toFixed(0)}%`}
          />
          <View style={{ marginTop: 12, width: "100%" }}>
            <ProgressBar fraction={downloadProgress} color={T.text} />
          </View>
        </View>
      )}

      {isLoading && (
        <View style={s.progressWrap}>
          <ActivityIndicator color={T.dim} />
          <Text style={s.loadingText}>Loading engine…</Text>
        </View>
      )}

      {error && (
        <Text style={s.errorText}>
          {errorIsMemory ? "🛡 Pre-flight check refused the load: " : ""}
          {error}
        </Text>
      )}
    </View>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: T.bg },

  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: T.bg,
  },
  circleBtn: {
    width: CIRCLE,
    height: CIRCLE,
    borderRadius: CIRCLE / 2,
    backgroundColor: T.surface,
    alignItems: "center",
    justifyContent: "center",
  },
  circleIcon: { fontSize: 17, color: T.text },
  plusIcon: { fontSize: 20, color: T.text, marginTop: -1 },
  sendIcon: { fontSize: 20, fontWeight: "700" },
  namePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: T.surface,
    borderRadius: CIRCLE / 2,
    paddingHorizontal: 16,
    height: CIRCLE,
  },
  nameText: { fontSize: 16, fontWeight: "600", color: T.text },
  nameChevron: { fontSize: 13, color: T.dim, marginTop: -4 },

  chatContent: { paddingVertical: 8, paddingBottom: 16, flexGrow: 1 },
  suggestRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 24,
    justifyContent: "center",
  },
  suggestChip: {
    paddingHorizontal: 14,
    paddingVertical: 9,
    backgroundColor: T.surface,
    borderRadius: 20,
  },
  suggestText: { fontSize: 13, color: T.text, fontWeight: "500" },

  scrollDown: {
    position: "absolute",
    alignSelf: "center",
    bottom: 96,
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: T.elevated,
    alignItems: "center",
    justifyContent: "center",
  },
  scrollDownIcon: { color: T.text, fontSize: 16, marginTop: -6 },

  composer: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
    paddingHorizontal: 12,
    paddingTop: 8,
  },
  inputPill: {
    flex: 1,
    minHeight: CIRCLE,
    borderRadius: 24,
    backgroundColor: T.surface,
    justifyContent: "center",
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  input: {
    fontSize: 16,
    color: T.text,
    paddingVertical: Platform.OS === "ios" ? 4 : 2,
    maxHeight: 120,
  },
  attachmentChip: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 8,
    backgroundColor: T.elevated,
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 6,
    marginTop: 4,
    marginBottom: 2,
  },
  attachmentText: { color: T.text, fontSize: 12, fontWeight: "500" },
  attachmentRemove: { color: T.dim, fontSize: 12 },

  // Setup / load
  setup: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 32,
  },
  setupMark: {
    fontSize: 34,
    fontWeight: "800",
    color: T.text,
    opacity: 0.2,
    letterSpacing: -0.5,
    marginBottom: 18,
  },
  setupTitle: { fontSize: 20, fontWeight: "700", color: T.text },
  setupSub: { fontSize: 13, color: T.dim, marginTop: 4 },
  preflightRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 18,
    maxWidth: 320,
  },
  verdictDot: { width: 8, height: 8, borderRadius: 4 },
  preflightText: { flexShrink: 1, fontSize: 12, color: T.dim, lineHeight: 17 },
  loadBtn: {
    marginTop: 20,
    backgroundColor: T.text,
    paddingHorizontal: 28,
    paddingVertical: 12,
    borderRadius: 24,
  },
  loadBtnText: { color: T.bg, fontWeight: "700", fontSize: 15 },
  progressWrap: { marginTop: 24, alignItems: "center", width: "100%" },
  loadingText: { color: T.dim, fontSize: 14, marginTop: 10 },
  errorText: {
    fontSize: 12,
    color: T.error,
    marginTop: 16,
    lineHeight: 17,
    textAlign: "center",
  },

  // Settings sheet
  sheetRoot: { flex: 1, backgroundColor: T.bg },
  sheetHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 4,
  },
  sheetTitle: { fontSize: 22, fontWeight: "700", color: T.text },
  pillRow: { flexDirection: "row", gap: 8 },
  dangerBtn: {
    paddingVertical: 12,
    borderRadius: 14,
    backgroundColor: T.errorSoft,
    alignItems: "center",
  },
  dangerText: { color: T.error, fontWeight: "600", fontSize: 14 },
});
