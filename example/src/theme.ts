import { Platform } from "react-native";

/**
 * Design tokens — ChatGPT-style pure-black dark theme (after
 * margelo/ai-chat-demo). iOS system grays throughout.
 */
export const T = {
  bg: "#000000",
  surface: "#1C1C1E", // systemGray6 dark — bubbles, pills, sheets
  card: "#1C1C1E",
  elevated: "#2C2C2E",
  accent: "#0A84FF", // iOS system blue
  accentSoft: "rgba(10,132,255,0.16)",
  accentGlow: "#64B5FF",
  success: "#30D158",
  successSoft: "rgba(48,209,88,0.14)",
  warning: "#FFD60A",
  warningSoft: "rgba(255,214,10,0.14)",
  error: "#FF453A",
  errorSoft: "rgba(255,69,58,0.14)",
  cyan: "#64D2FF",
  purple: "#BF5AF2",
  text: "#FFFFFF",
  dim: "#8E8E93", // systemGray — secondary text
  faint: "#48484A", // disabled tint
  border: "#2C2C2E",
  sendActive: "#FFFFFF",
} as const;

export const MONO = Platform.OS === "ios" ? "Menlo" : "monospace";

/** Verdict → color mapping used across estimate/forecast UI. */
export const VERDICT_COLORS = {
  safe: { fg: T.success, bg: T.successSoft, label: "Fits comfortably" },
  tight: { fg: T.warning, bg: T.warningSoft, label: "Tight fit" },
  critical: { fg: T.error, bg: T.errorSoft, label: "Won't fit" },
} as const;
