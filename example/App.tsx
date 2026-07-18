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
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
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
import { ChatBubble, EmptyState, type ChatMsg } from "./src/components/ChatView";
import { MemoryPanel } from "./src/components/MemoryPanel";
import { Card, MetricChip, Pill, ProgressBar, PulseDot, SectionLabel } from "./src/components/ui";
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

type Tab = "chat" | "memory";

// ═════════════════════════════════════════════════════════════════════════════
export default function App() {
  return (
    <SafeAreaProvider>
      <Main />
    </SafeAreaProvider>
  );
}

function Main() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [tab, setTab] = useState<Tab>("chat");
  const [sel, setSel] = useState<ModelKey>("gemma3n");
  const [backend, setBackend] = useState<"cpu" | "gpu">("cpu");
  const [contextTokens, setContextTokens] = useState<number>(4096);
  const [enableSpeculativeDecoding, setEnableSpeculativeDecoding] = useState(false);
  const [enableTools, setEnableTools] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [chat, setChat] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState<ChatMsg | null>(null);
  const [busy, setBusy] = useState(false);
  const [attachment, setAttachment] = useState<"image" | "audio" | null>(null);
  const [liveUsage, setLiveUsage] = useState<MemoryUsage | null>(null);
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
    }),
    [backend, contextTokens, enableSpeculativeDecoding, enableTools],
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
    setChat((prev) => [...prev, { role: "user", text: displayMsg, ts: Date.now() }]);

    const acc: ChatMsg = { role: "model", text: "", ts: Date.now() };
    setStreaming({ ...acc });

    try {
      const parts: any[] = [];
      if (currentAttachment === "image") {
        const uri = Image.resolveAssetSource(TEST_IMAGE_ASSET).uri;
        parts.push({ type: "image", imageBuffer: await (await fetch(uri)).arrayBuffer() });
      } else if (currentAttachment === "audio") {
        const uri = Image.resolveAssetSource(TEST_AUDIO_ASSET).uri;
        parts.push({ type: "audio", audioBuffer: await (await fetch(uri)).arrayBuffer() });
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
      setChat((prev) => [...prev, { ...acc }]);
      setStreaming(null);
      refreshUsage();
    } catch (e: any) {
      setChat((prev) => [
        ...prev,
        { role: "model", text: `Error: ${e.message}`, ts: Date.now() },
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

  // ── Derived ────────────────────────────────────────────────────────────────
  const stats = model && isReady ? model.getStats() : null;
  const isDownloading = downloadProgress > 0 && downloadProgress < 1;
  const isLoading = downloadProgress === 1 && !isReady;
  const canInteract = !isReady && !isDownloading && !isLoading;
  const gpuWarning = useMemo(() => checkBackendSupport("gpu"), []);
  const snapshots = model?.memoryTracker?.getSnapshots() ?? [];
  const errorIsMemory = error?.includes("MEMORY_ESTIMATE_EXCEEDED") || error?.includes("Refusing to load");

  return (
    <SafeAreaView style={s.root}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        {/* ── Header ───────────────────────────────────────────────────────── */}
        <View style={s.header}>
          <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
            <PulseDot active={isReady} color={T.success} />
            <View>
              <Text style={s.brand}>
                litert<Text style={{ color: T.accent }}>·lm</Text>
              </Text>
              <Text style={s.tagline}>
                {isReady
                  ? `${MODELS[sel].label} · ${backend.toUpperCase()} · on-device`
                  : "on-device LLM inference"}
              </Text>
            </View>
          </View>
          <TouchableOpacity
            testID="settings-btn"
            style={s.iconBtn}
            onPress={() => setShowSettings(!showSettings)}
          >
            <Text style={{ fontSize: 16, color: T.text }}>
              {showSettings ? "✕" : "⚙"}
            </Text>
          </TouchableOpacity>
        </View>

        {/* ── Tabs ─────────────────────────────────────────────────────────── */}
        <View style={s.tabs}>
          {(
            [
              ["chat", "Chat"],
              ["memory", "Memory"],
            ] as [Tab, string][]
          ).map(([key, label]) => (
            <TouchableOpacity
              key={key}
              testID={`tab-${key}`}
              style={[s.tab, tab === key && s.tabActive]}
              onPress={() => setTab(key)}
            >
              <Text style={[s.tabText, tab === key && s.tabTextActive]}>
                {label}
              </Text>
              {key === "memory" && memoryWarning && (
                <View style={s.tabAlert} />
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Settings sheet ───────────────────────────────────────────────── */}
        {showSettings && (
          <ScrollView style={s.sheet} contentContainerStyle={{ gap: 14 }}>
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

            {isReady && (
              <TouchableOpacity
                style={s.dangerBtn}
                onPress={() => deleteModel(MODELS[sel].fileName).catch(() => {})}
              >
                <Text style={s.dangerText}>Delete cached model</Text>
              </TouchableOpacity>
            )}
          </ScrollView>
        )}

        {/* ── Load card (pre-flight estimate front and center) ─────────────── */}
        {!isReady && !showSettings && (
          <Card style={{ marginHorizontal: 16, marginBottom: 12 }}>
            <View style={s.loadHeader}>
              <View style={{ flex: 1 }}>
                <Text style={s.loadTitle}>
                  {isDownloading
                    ? `Downloading ${(downloadProgress * 100).toFixed(0)}%`
                    : isLoading
                      ? "Loading engine…"
                      : MODELS[sel].label}
                </Text>
                <Text style={s.loadSub}>
                  {MODELS[sel].size} · {backend.toUpperCase()} ·{" "}
                  {contextTokens / 1024}k context
                </Text>
              </View>
              {canInteract && (
                <TouchableOpacity
                  testID="load-btn"
                  style={[
                    s.loadBtn,
                    preflight?.verdict === "critical" && {
                      backgroundColor: T.error,
                    },
                  ]}
                  onPress={load}
                >
                  <Text style={s.loadBtnText}>
                    {preflight?.verdict === "critical" ? "Try anyway" : "Load"}
                  </Text>
                </TouchableOpacity>
              )}
              {(isDownloading || isLoading) && (
                <ActivityIndicator color={T.accent} />
              )}
            </View>

            {isDownloading && (
              <View style={{ marginTop: 12 }}>
                <ProgressBar fraction={downloadProgress} />
              </View>
            )}

            {preflight && !isDownloading && !isLoading && (
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
                      fontWeight: "700",
                    }}
                  >
                    {VERDICT_COLORS[preflight.verdict].label}.
                  </Text>{" "}
                  Needs ~{fmtBytes(preflight.totalEstimatedBytes)} of{" "}
                  {fmtBytes(preflight.availableBytes)} available.
                </Text>
              </View>
            )}

            {error && (
              <Text style={s.errorText}>
                {errorIsMemory ? "🛡 Pre-flight check refused the load: " : ""}
                {error}
              </Text>
            )}
          </Card>
        )}

        {/* ── Body ─────────────────────────────────────────────────────────── */}
        {tab === "memory" ? (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={{ padding: 16, paddingTop: 4, gap: 14 }}
          >
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
          </ScrollView>
        ) : (
          <>
            {isReady && (
              <View style={s.metricsBar}>
                <MetricChip
                  label="Speed"
                  value={stats?.tokensPerSecond ? stats.tokensPerSecond.toFixed(1) : "—"}
                  unit="tok/s"
                  color={T.success}
                />
                <MetricChip
                  label="First token"
                  value={stats?.timeToFirstToken ? `${(stats.timeToFirstToken * 1000).toFixed(0)}` : "—"}
                  unit="ms"
                  color={T.cyan}
                />
                <MetricChip
                  label="Context"
                  value={
                    memoryForecast
                      ? `${Math.round(memoryForecast.contextUsedFraction * 100)}%`
                      : "—"
                  }
                  unit="used"
                  color={memoryForecast?.nearingLimit ? T.warning : T.purple}
                />
              </View>
            )}

            <ScrollView
              ref={scrollRef}
              style={{ flex: 1 }}
              contentContainerStyle={s.chatContent}
              keyboardShouldPersistTaps="handled"
            >
              {!isReady && chat.length === 0 && (
                <EmptyState
                  icon="✦"
                  title="On-device intelligence"
                  sub={
                    "Load a model to start. The pre-flight memory check\nmakes sure it fits before a single byte downloads."
                  }
                />
              )}

              {isReady && chat.length === 0 && !streaming && (
                <EmptyState
                  icon="💬"
                  title="Ready"
                  sub={`${MODELS[sel].label} on ${backend.toUpperCase()} — everything stays on this device.`}
                >
                  <View style={s.suggestRow}>
                    {[
                      "Explain LoRA adapters in one line",
                      "Tell me a joke",
                      enableTools ? "What's the weather in Tokyo?" : "Write a haiku about RAM",
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
                <ChatBubble key={i} msg={m} />
              ))}
              {streaming && <ChatBubble msg={streaming} isStreaming />}
            </ScrollView>

            {/* ── Input bar ────────────────────────────────────────────────── */}
            {isReady && (
              <View>
                {attachment && (
                  <View style={s.attachmentPreview}>
                    <Text style={s.attachmentText}>
                      {attachment === "image" ? "🖼 test.jpeg" : "🎧 test.wav"}{" "}
                      attached (zero-copy ArrayBuffer)
                    </Text>
                    <TouchableOpacity onPress={() => setAttachment(null)}>
                      <Text style={{ color: T.dim, fontSize: 14 }}>✕</Text>
                    </TouchableOpacity>
                  </View>
                )}
                <View style={s.inputBar}>
                  <TouchableOpacity
                    testID="attach-btn"
                    style={s.iconBtn}
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
                    <Text style={{ fontSize: 16 }}>📎</Text>
                  </TouchableOpacity>
                  <TextInput
                    testID="chat-input"
                    style={s.input}
                    placeholder={busy ? "Generating…" : "Message"}
                    placeholderTextColor={T.faint}
                    value={input}
                    onChangeText={setInput}
                    editable={!busy}
                    onSubmitEditing={send}
                    returnKeyType="send"
                    multiline
                  />
                  <TouchableOpacity
                    testID="send-btn"
                    style={[
                      s.sendBtn,
                      ((!input.trim() && !attachment) || busy) && { opacity: 0.4 },
                    ]}
                    onPress={send}
                    disabled={(!input.trim() && !attachment) || busy}
                  >
                    {busy ? (
                      <ActivityIndicator color="#fff" size="small" />
                    ) : (
                      <Text style={s.sendIcon}>↑</Text>
                    )}
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: T.bg },

  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 10,
  },
  brand: { fontSize: 24, fontWeight: "900", color: T.text, letterSpacing: -0.5 },
  tagline: { fontSize: 11, color: T.dim, marginTop: 1, fontWeight: "500" },
  iconBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: T.card,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: T.border,
  },

  tabs: {
    flexDirection: "row",
    marginHorizontal: 16,
    marginBottom: 12,
    backgroundColor: T.card,
    borderRadius: 12,
    padding: 3,
    borderWidth: 1,
    borderColor: T.border,
  },
  tab: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 9,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    gap: 6,
  },
  tabActive: { backgroundColor: T.elevated },
  tabText: { fontSize: 13, fontWeight: "700", color: T.dim },
  tabTextActive: { color: T.text },
  tabAlert: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: T.warning,
  },

  sheet: {
    marginHorizontal: 16,
    marginBottom: 12,
    padding: 16,
    backgroundColor: T.surface,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: T.border,
    maxHeight: 380,
  },
  pillRow: { flexDirection: "row", gap: 8 },
  dangerBtn: {
    paddingVertical: 11,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: T.error,
    alignItems: "center",
  },
  dangerText: { color: T.error, fontWeight: "700", fontSize: 13 },

  loadHeader: { flexDirection: "row", alignItems: "center" },
  loadTitle: { fontSize: 16, fontWeight: "800", color: T.text },
  loadSub: { fontSize: 12, color: T.dim, marginTop: 2 },
  loadBtn: {
    backgroundColor: T.accent,
    paddingHorizontal: 22,
    paddingVertical: 11,
    borderRadius: 12,
  },
  loadBtnText: { color: "#fff", fontWeight: "800", fontSize: 14 },
  preflightRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 12,
  },
  verdictDot: { width: 8, height: 8, borderRadius: 4 },
  preflightText: { flex: 1, fontSize: 12, color: T.dim, lineHeight: 17 },
  errorText: { fontSize: 12, color: T.error, marginTop: 10, lineHeight: 17 },

  metricsBar: {
    flexDirection: "row",
    gap: 8,
    marginHorizontal: 16,
    marginBottom: 10,
  },

  chatContent: { paddingHorizontal: 16, paddingBottom: 12, flexGrow: 1 },
  suggestRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 20,
    justifyContent: "center",
  },
  suggestChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    backgroundColor: T.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: T.border,
  },
  suggestText: { fontSize: 13, color: T.accentGlow, fontWeight: "600" },

  attachmentPreview: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: T.accentSoft,
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  attachmentText: { color: T.accentGlow, fontSize: 12, fontWeight: "600" },
  inputBar: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: T.border,
    backgroundColor: T.bg,
  },
  input: {
    flex: 1,
    backgroundColor: T.surface,
    borderRadius: 22,
    paddingHorizontal: 18,
    paddingVertical: 11,
    color: T.text,
    fontSize: 15,
    borderWidth: 1,
    borderColor: T.border,
    maxHeight: 100,
  },
  sendBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: T.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  sendIcon: { color: "#fff", fontSize: 20, fontWeight: "900" },
});
