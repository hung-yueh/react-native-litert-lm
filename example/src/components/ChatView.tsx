import React, { useState } from "react";
import {
  Dimensions,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { MONO, T } from "../theme";

const { width: SCREEN_W } = Dimensions.get("window");

/** A chat turn. `thinking` / `toolCall` come from typed streaming events. */
export type ChatMsg = {
  role: "user" | "model";
  text: string;
  thinking?: string;
  toolCall?: string;
  ts: number;
};

export function ChatBubble({
  msg,
  isStreaming,
}: {
  msg: ChatMsg;
  isStreaming?: boolean;
}) {
  const isUser = msg.role === "user";
  return (
    <View style={[s.row, isUser && { justifyContent: "flex-end" }]}>
      {!isUser && (
        <View style={s.avatar}>
          <Text style={{ fontSize: 11, color: T.accentGlow }}>✦</Text>
        </View>
      )}
      <View style={{ maxWidth: SCREEN_W * 0.78, gap: 6 }}>
        {!isUser && msg.thinking ? (
          <ThinkingBlock text={msg.thinking} live={!!isStreaming && !msg.text} />
        ) : null}
        {!isUser && msg.toolCall ? <ToolCallChip json={msg.toolCall} /> : null}
        {msg.text || isUser || (!msg.thinking && !msg.toolCall) ? (
          <View style={[s.bubble, isUser ? s.bubbleUser : s.bubbleModel]}>
            <Text style={[s.bubbleText, isUser && { color: "#fff" }]}>
              {msg.text}
              {isStreaming ? <Text style={s.cursor}>▊</Text> : null}
            </Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}

/** Collapsible reasoning content from the `thinking` stream channel. */
function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  return (
    <TouchableOpacity
      style={s.thinking}
      onPress={() => setOpen(!open)}
      activeOpacity={0.8}
    >
      <View style={s.thinkingHeader}>
        <Text style={s.thinkingTitle}>
          {live ? "◌ Thinking…" : "◌ Reasoning"}
        </Text>
        <Text style={s.thinkingToggle}>{open ? "hide" : "show"}</Text>
      </View>
      {open && (
        <Text style={s.thinkingText} numberOfLines={live ? 4 : undefined}>
          {text.trim()}
        </Text>
      )}
    </TouchableOpacity>
  );
}

/** Tool-call content from the `toolCall` stream channel, rendered as code. */
function ToolCallChip({ json }: { json: string }) {
  let pretty = json.trim();
  try {
    pretty = JSON.stringify(JSON.parse(pretty), null, 2);
  } catch {
    // Stream may still be partial — show raw
  }
  return (
    <View style={s.tool}>
      <Text style={s.toolTitle}>⚙ Tool call</Text>
      <Text style={s.toolJson}>{pretty}</Text>
    </View>
  );
}

export function EmptyState({
  icon,
  title,
  sub,
  children,
}: {
  icon: string;
  title: string;
  sub: string;
  children?: React.ReactNode;
}) {
  return (
    <View style={s.empty}>
      <Text style={s.emptyIcon}>{icon}</Text>
      <Text style={s.emptyTitle}>{title}</Text>
      <Text style={s.emptySub}>{sub}</Text>
      {children}
    </View>
  );
}

const s = StyleSheet.create({
  row: { flexDirection: "row", alignItems: "flex-end", marginBottom: 12 },
  avatar: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: T.accentSoft,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 8,
    borderWidth: 1,
    borderColor: T.border,
  },
  bubble: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 18,
  },
  bubbleUser: { backgroundColor: T.accent, borderBottomRightRadius: 4 },
  bubbleModel: {
    backgroundColor: T.card,
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: T.border,
  },
  bubbleText: { fontSize: 15, color: T.text, lineHeight: 22 },
  cursor: { color: T.accentGlow, fontSize: 14 },

  thinking: {
    backgroundColor: "rgba(192,132,252,0.08)",
    borderWidth: 1,
    borderColor: "rgba(192,132,252,0.25)",
    borderRadius: 14,
    padding: 10,
  },
  thinkingHeader: { flexDirection: "row", justifyContent: "space-between" },
  thinkingTitle: { fontSize: 11, fontWeight: "700", color: T.purple },
  thinkingToggle: { fontSize: 11, color: T.faint, fontWeight: "600" },
  thinkingText: {
    fontSize: 12,
    color: T.dim,
    lineHeight: 17,
    marginTop: 6,
    fontStyle: "italic",
  },

  tool: {
    backgroundColor: "rgba(34,211,238,0.07)",
    borderWidth: 1,
    borderColor: "rgba(34,211,238,0.25)",
    borderRadius: 14,
    padding: 10,
  },
  toolTitle: { fontSize: 11, fontWeight: "700", color: T.cyan, marginBottom: 6 },
  toolJson: { fontSize: 11, color: T.text, fontFamily: MONO, lineHeight: 16 },

  empty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 48,
    paddingHorizontal: 24,
  },
  emptyIcon: { fontSize: 34, marginBottom: 12, color: T.accent },
  emptyTitle: { fontSize: 20, fontWeight: "800", color: T.text },
  emptySub: {
    fontSize: 14,
    color: T.dim,
    textAlign: "center",
    marginTop: 6,
    lineHeight: 21,
  },
});
