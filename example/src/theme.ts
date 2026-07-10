import { Platform } from "react-native";

/** Design tokens — a refined dark theme with a single indigo accent. */
export const T = {
  bg: "#0A0A0F",
  surface: "#12121A",
  card: "#181822",
  elevated: "#1F1F2C",
  accent: "#6366F1",
  accentSoft: "rgba(99,102,241,0.14)",
  accentGlow: "#A5B4FC",
  success: "#34D399",
  successSoft: "rgba(52,211,153,0.14)",
  warning: "#FBBF24",
  warningSoft: "rgba(251,191,36,0.14)",
  error: "#F87171",
  errorSoft: "rgba(248,113,113,0.14)",
  cyan: "#22D3EE",
  purple: "#C084FC",
  text: "#F4F4F7",
  dim: "#8B8B9E",
  faint: "#55556A",
  border: "#26263A",
} as const;

export const MONO = Platform.OS === "ios" ? "Menlo" : "monospace";

/** Verdict → color mapping used across estimate/forecast UI. */
export const VERDICT_COLORS = {
  safe: { fg: T.success, bg: T.successSoft, label: "Fits comfortably" },
  tight: { fg: T.warning, bg: T.warningSoft, label: "Tight fit" },
  critical: { fg: T.error, bg: T.errorSoft, label: "Won't fit" },
} as const;
