import React, { useEffect, useRef, useState } from "react";
import {
  Animated,
  Easing,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { MONO, T } from "../theme";

/** A chat turn. `thinking` / `toolCall` come from typed streaming events. */
export type ChatMsg = {
  role: "user" | "model";
  text: string;
  thinking?: string;
  toolCall?: string;
  /** e.g. "24.1 tok/s · 180 ms to first token" — shown under a done reply */
  meta?: string;
  error?: boolean;
  ts: number;
};

/** ChatGPT-style turn: user = right-aligned gray bubble, model = plain text. */
export function MessageBubble({
  msg,
  isStreaming,
}: {
  msg: ChatMsg;
  isStreaming?: boolean;
}) {
  if (msg.role === "user") {
    return (
      <SlideIn style={s.userRow}>
        <View style={s.userBubble}>
          <Text style={s.userText}>{msg.text}</Text>
        </View>
      </SlideIn>
    );
  }

  const isWaiting = !!isStreaming && msg.text.length === 0 && !msg.toolCall;
  const showTrace = !!msg.thinking && !isWaiting;

  return (
    <View style={s.assistantRow}>
      {showTrace ? <ThinkingTrace text={msg.thinking!} /> : null}
      {msg.toolCall ? <ToolCallBlock json={msg.toolCall} /> : null}
      {isWaiting ? (
        <ShimmerText text="Thinking" />
      ) : msg.text.length > 0 ? (
        <Text style={[s.assistantText, msg.error && s.errorText]}>
          {msg.text}
        </Text>
      ) : null}
      {msg.meta && !isStreaming ? (
        <Text style={s.metaText}>{msg.meta}</Text>
      ) : null}
    </View>
  );
}

/** User bubble slides up from the composer as it's sent (ChatGPT-style). */
function SlideIn({
  style,
  children,
}: {
  style?: object;
  children: React.ReactNode;
}) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(anim, {
      toValue: 1,
      duration: 450,
      easing: Easing.out(Easing.exp),
      useNativeDriver: true,
    }).start();
  }, [anim]);
  const translateY = anim.interpolate({
    inputRange: [0, 1],
    outputRange: [28, 0],
  });
  return (
    <Animated.View style={[style, { opacity: anim, transform: [{ translateY }] }]}>
      {children}
    </Animated.View>
  );
}

/** Pulsing status label — stand-in for the demo's Skia shimmer. */
export function ShimmerText({ text }: { text: string }) {
  const anim = useRef(new Animated.Value(0)).current;
  const fade = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    // Delay the fade-in so the label doesn't pop while the user's message is
    // still sliding up.
    Animated.timing(fade, {
      toValue: 1,
      duration: 300,
      delay: 450,
      useNativeDriver: true,
    }).start();
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(anim, {
          toValue: 1,
          duration: 750,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(anim, {
          toValue: 0,
          duration: 750,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [anim, fade]);
  const opacity = Animated.multiply(
    fade,
    anim.interpolate({ inputRange: [0, 1], outputRange: [0.4, 1] }),
  );
  return (
    <Animated.View style={[s.statusRow, { opacity }]}>
      <Text style={s.statusIcon}>✦</Text>
      <Text style={s.statusLabel}>{text}</Text>
    </Animated.View>
  );
}

/** Collapsed reasoning row (clock · label · chevron) that expands inline. */
function ThinkingTrace({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <View>
      <TouchableOpacity
        style={s.traceRow}
        hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}
        onPress={() => setOpen(!open)}
        activeOpacity={0.7}
      >
        <Text style={s.traceIcon}>◷</Text>
        <Text style={s.traceLabel} numberOfLines={1}>
          Thought about it
        </Text>
        <Text style={s.traceChevron}>{open ? "⌄" : "›"}</Text>
      </TouchableOpacity>
      {open ? <Text style={s.traceText}>{text.trim()}</Text> : null}
    </View>
  );
}

/** Tool-call content from the `toolCall` stream channel, rendered as code. */
function ToolCallBlock({ json }: { json: string }) {
  let pretty = json.trim();
  try {
    pretty = JSON.stringify(JSON.parse(pretty), null, 2);
  } catch {
    // Stream may still be partial — show raw
  }
  return (
    <View style={s.tool}>
      <Text style={s.toolTitle}>tool call</Text>
      <Text style={s.toolJson}>{pretty}</Text>
    </View>
  );
}

export function EmptyState({
  title,
  sub,
  children,
}: {
  title: string;
  sub?: string;
  children?: React.ReactNode;
}) {
  return (
    <View style={s.empty}>
      <Text style={s.emptyMark}>
        litert<Text style={{ fontWeight: "400" }}>·lm</Text>
      </Text>
      <Text style={s.emptyTitle}>{title}</Text>
      {sub ? <Text style={s.emptySub}>{sub}</Text> : null}
      {children}
    </View>
  );
}

const s = StyleSheet.create({
  userRow: {
    alignItems: "flex-end",
    paddingHorizontal: 16,
    paddingVertical: 4,
  },
  userBubble: {
    maxWidth: "82%",
    backgroundColor: T.surface,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  userText: { color: T.text, fontSize: 16, lineHeight: 21 },

  assistantRow: {
    paddingHorizontal: 16,
    paddingVertical: 4,
  },
  assistantText: { color: T.text, fontSize: 16, lineHeight: 22 },
  errorText: { color: T.error },
  metaText: { color: T.faint, fontSize: 12, marginTop: 8 },

  statusRow: { flexDirection: "row", alignItems: "center", gap: 6 },
  statusIcon: { color: T.dim, fontSize: 14 },
  statusLabel: { color: T.dim, fontSize: 16 },

  traceRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 6,
  },
  traceIcon: { color: T.dim, fontSize: 14 },
  traceLabel: { flex: 1, color: T.dim, fontSize: 16 },
  traceChevron: { color: T.dim, fontSize: 15 },
  traceText: {
    color: T.dim,
    fontSize: 14,
    lineHeight: 20,
    paddingBottom: 8,
  },

  tool: {
    backgroundColor: T.surface,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: T.border,
    padding: 12,
    marginVertical: 4,
  },
  toolTitle: {
    fontSize: 11,
    fontWeight: "600",
    color: T.dim,
    marginBottom: 6,
    textTransform: "uppercase",
    letterSpacing: 0.6,
  },
  toolJson: { fontSize: 13, color: T.text, fontFamily: MONO, lineHeight: 18 },

  empty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 48,
    paddingHorizontal: 24,
  },
  emptyMark: {
    fontSize: 34,
    fontWeight: "800",
    color: T.text,
    opacity: 0.2,
    letterSpacing: -0.5,
    marginBottom: 14,
  },
  emptyTitle: { fontSize: 17, fontWeight: "600", color: T.dim },
  emptySub: {
    fontSize: 14,
    color: T.faint,
    textAlign: "center",
    marginTop: 6,
    lineHeight: 21,
  },
});
