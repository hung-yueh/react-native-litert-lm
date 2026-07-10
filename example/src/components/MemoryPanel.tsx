import React from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import type {
  MemoryEstimate,
  MemoryForecast,
  MemoryTrackerSummary,
  MemoryUsage,
  MemoryWarningLevel,
  MemorySnapshot,
} from "react-native-litert-lm";
import { fmtBytes, fmtCount, fmtPct } from "../format";
import { T, VERDICT_COLORS } from "../theme";
import {
  Badge,
  Card,
  ProgressBar,
  SectionLabel,
  Sparkbars,
  StatRow,
} from "./ui";

/**
 * The memory dashboard — the library's crash-free story in one screen:
 * pre-flight estimate, context/KV forecast, live usage, snapshot history,
 * OS pressure warnings, and deterministic unload.
 */
export function MemoryPanel({
  estimate,
  forecast,
  usage,
  summary,
  snapshots,
  warning,
  isReady,
  onUnload,
}: {
  estimate: MemoryEstimate | null;
  forecast: MemoryForecast | null;
  usage: MemoryUsage | null;
  summary: MemoryTrackerSummary | null;
  snapshots: MemorySnapshot[];
  warning: MemoryWarningLevel | null;
  isReady: boolean;
  onUnload: () => void;
}) {
  return (
    <View style={{ gap: 14 }}>
      {warning && (
        <Card
          style={{
            backgroundColor:
              warning === "critical" ? T.errorSoft : T.warningSoft,
            borderColor: warning === "critical" ? T.error : T.warning,
          }}
        >
          <Text
            style={[
              s.warningTitle,
              { color: warning === "critical" ? T.error : T.warning },
            ]}
          >
            {warning === "critical"
              ? "⚠ Critical memory pressure"
              : "△ Memory pressure"}
          </Text>
          <Text style={s.warningBody}>
            The OS signaled {warning} pressure. The library surfaced it via
            setMemoryWarningCallback — free memory or unload the model.
          </Text>
        </Card>
      )}

      <Card>
        <SectionLabel>Pre-flight estimate</SectionLabel>
        {estimate ? (
          <>
            <View style={s.verdictRow}>
              <Badge
                label={VERDICT_COLORS[estimate.verdict].label}
                fg={VERDICT_COLORS[estimate.verdict].fg}
                bg={VERDICT_COLORS[estimate.verdict].bg}
              />
              <Text style={s.verdictPct}>
                {fmtPct(Math.min(estimate.usageFraction, 1.5))} of available
              </Text>
            </View>
            <View style={{ marginTop: 12, marginBottom: 4 }}>
              <ProgressBar
                fraction={estimate.usageFraction}
                color={VERDICT_COLORS[estimate.verdict].fg}
              />
            </View>
            <StatRow label="Weights" value={fmtBytes(estimate.modelBytes)} />
            <StatRow
              label={`KV cache (${fmtCount(estimate.maxContextTokens)} tokens)`}
              value={fmtBytes(estimate.kvCacheBytes)}
            />
            <StatRow label="Overhead" value={fmtBytes(estimate.overheadBytes)} />
            <View style={s.divider} />
            <StatRow
              label="Estimated total"
              value={fmtBytes(estimate.totalEstimatedBytes)}
              color={VERDICT_COLORS[estimate.verdict].fg}
            />
            <StatRow
              label="Available"
              value={fmtBytes(estimate.availableBytes)}
            />
            {estimate.verdict !== "safe" && (
              <Text style={s.recommendation}>{estimate.recommendation}</Text>
            )}
          </>
        ) : (
          <Text style={s.placeholder}>
            Select a model to see whether it fits in memory — before
            downloading a single byte.
          </Text>
        )}
      </Card>

      {isReady && forecast && (
        <Card>
          <SectionLabel>Context forecast</SectionLabel>
          <View style={s.verdictRow}>
            <Text style={s.forecastBig}>
              {fmtCount(forecast.contextTokensUsed)}
              <Text style={s.forecastDim}>
                {" "}
                / {fmtCount(forecast.maxContextTokens)} tokens
              </Text>
            </Text>
            {forecast.nearingLimit && (
              <Badge label="Nearing limit" fg={T.warning} bg={T.warningSoft} />
            )}
          </View>
          <View style={{ marginTop: 12, marginBottom: 4 }}>
            <ProgressBar
              fraction={forecast.contextUsedFraction}
              color={forecast.nearingLimit ? T.warning : T.cyan}
            />
          </View>
          <StatRow
            label="KV cache in use"
            value={fmtBytes(forecast.kvCacheBytesUsed)}
          />
          <StatRow
            label="Grows up to"
            value={`+${fmtBytes(forecast.kvCacheBytesRemaining)}`}
          />
        </Card>
      )}

      {usage && (
        <Card>
          <SectionLabel>Live usage</SectionLabel>
          <StatRow label="Process RSS" value={fmtBytes(usage.residentBytes)} />
          <StatRow
            label="Available headroom"
            value={fmtBytes(usage.availableMemoryBytes)}
            color={usage.isLowMemory ? T.error : T.success}
          />
          {summary && summary.snapshotCount > 1 && (
            <>
              <View style={{ marginTop: 10 }}>
                <Sparkbars
                  values={snapshots.map((sn) => sn.residentBytes)}
                  color={T.accent}
                />
              </View>
              <View style={s.sparkCaption}>
                <Text style={s.sparkLabel}>
                  RSS across {summary.snapshotCount} snapshots
                </Text>
                <Text style={s.sparkLabel}>
                  peak {fmtBytes(summary.peakResidentBytes)}
                </Text>
              </View>
            </>
          )}
        </Card>
      )}

      {isReady && (
        <TouchableOpacity testID="unload-btn" style={s.unloadBtn} onPress={onUnload}>
          <Text style={s.unloadText}>Unload model — free memory now</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  verdictRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  verdictPct: { fontSize: 12, color: T.dim, fontWeight: "600" },
  divider: { height: 1, backgroundColor: T.border, marginVertical: 6 },
  recommendation: {
    marginTop: 10,
    fontSize: 12,
    lineHeight: 17,
    color: T.dim,
  },
  placeholder: { fontSize: 13, color: T.dim, lineHeight: 19 },
  forecastBig: { fontSize: 20, fontWeight: "800", color: T.text },
  forecastDim: { fontSize: 13, fontWeight: "600", color: T.dim },
  warningTitle: { fontSize: 14, fontWeight: "800" },
  warningBody: { fontSize: 12, color: T.dim, marginTop: 4, lineHeight: 17 },
  sparkCaption: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 6,
  },
  sparkLabel: { fontSize: 10, color: T.faint, fontWeight: "600" },
  unloadBtn: {
    paddingVertical: 13,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: T.error,
    alignItems: "center",
    backgroundColor: T.errorSoft,
  },
  unloadText: { color: T.error, fontWeight: "700", fontSize: 13 },
});
