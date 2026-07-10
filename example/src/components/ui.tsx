import React, { useEffect, useRef } from "react";
import {
  Animated,
  Easing,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  type ViewStyle,
} from "react-native";
import { MONO, T } from "../theme";

// ─── Layout primitives ───────────────────────────────────────────────────────

export function Card({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: ViewStyle;
}) {
  return <View style={[s.card, style]}>{children}</View>;
}

export function SectionLabel({ children }: { children: string }) {
  return <Text style={s.sectionLabel}>{children}</Text>;
}

// ─── Controls ────────────────────────────────────────────────────────────────

export function Pill({
  label,
  sub,
  active,
  disabled,
  onPress,
  testID,
}: {
  label: string;
  sub?: string;
  active?: boolean;
  disabled?: boolean;
  onPress: () => void;
  testID?: string;
}) {
  return (
    <TouchableOpacity
      testID={testID}
      disabled={disabled}
      onPress={onPress}
      style={[s.pill, active && s.pillActive, disabled && { opacity: 0.4 }]}
    >
      <Text style={[s.pillText, active && s.pillTextActive]}>{label}</Text>
      {sub ? <Text style={s.pillSub}>{sub}</Text> : null}
    </TouchableOpacity>
  );
}

export function Badge({
  label,
  fg,
  bg,
}: {
  label: string;
  fg: string;
  bg: string;
}) {
  return (
    <View style={[s.badge, { backgroundColor: bg }]}>
      <View style={[s.badgeDot, { backgroundColor: fg }]} />
      <Text style={[s.badgeText, { color: fg }]}>{label}</Text>
    </View>
  );
}

// ─── Data display ────────────────────────────────────────────────────────────

export function ProgressBar({
  fraction,
  color = T.accent,
  height = 6,
}: {
  fraction: number;
  color?: string;
  height?: number;
}) {
  const clamped = Math.max(0, Math.min(1, fraction));
  return (
    <View style={[s.progressTrack, { height, borderRadius: height / 2 }]}>
      <View
        style={{
          width: `${clamped * 100}%`,
          height,
          borderRadius: height / 2,
          backgroundColor: color,
        }}
      />
    </View>
  );
}

export function MetricChip({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: string;
  unit?: string;
  color: string;
}) {
  return (
    <View style={s.metricChip}>
      <Text style={s.metricLabel}>{label}</Text>
      <Text style={[s.metricValue, { color }]}>
        {value}
        {unit ? <Text style={s.metricUnit}> {unit}</Text> : null}
      </Text>
    </View>
  );
}

export function StatRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <View style={s.statRow}>
      <Text style={s.statLabel}>{label}</Text>
      <Text style={[s.statValue, color ? { color } : null]}>{value}</Text>
    </View>
  );
}

/** Tiny bar-strip chart for memory snapshots — no chart library needed. */
export function Sparkbars({
  values,
  color = T.accent,
  height = 36,
}: {
  values: number[];
  color?: string;
  height?: number;
}) {
  const max = Math.max(...values, 1);
  return (
    <View style={[s.sparkRow, { height }]}>
      {values.slice(-40).map((v, i) => (
        <View
          key={i}
          style={{
            flex: 1,
            marginHorizontal: 0.5,
            height: Math.max(2, (v / max) * height),
            borderRadius: 1,
            backgroundColor: color,
            opacity: 0.35 + 0.65 * (v / max),
          }}
        />
      ))}
    </View>
  );
}

// ─── Motion ──────────────────────────────────────────────────────────────────

/** Soft breathing dot used as the "engine alive" indicator. */
export function PulseDot({
  active,
  color = T.accent,
  size = 10,
}: {
  active: boolean;
  color?: string;
  size?: number;
}) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (active) {
      const loop = Animated.loop(
        Animated.sequence([
          Animated.timing(anim, {
            toValue: 1,
            duration: 900,
            easing: Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0,
            duration: 900,
            easing: Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
        ]),
      );
      loop.start();
      return () => loop.stop();
    }
    anim.setValue(0);
  }, [active, anim]);

  const opacity = anim.interpolate({ inputRange: [0, 1], outputRange: [0.4, 1] });
  return (
    <Animated.View
      style={{
        width: size,
        height: size,
        borderRadius: size / 2,
        backgroundColor: active ? color : T.faint,
        opacity: active ? opacity : 1,
      }}
    />
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  card: {
    backgroundColor: T.surface,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: T.border,
    padding: 16,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: T.dim,
    textTransform: "uppercase",
    letterSpacing: 1.2,
    marginBottom: 8,
  },
  pill: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    alignItems: "center",
  },
  pillActive: { borderColor: T.accent, backgroundColor: T.accentSoft },
  pillText: { fontSize: 13, fontWeight: "700", color: T.dim },
  pillTextActive: { color: T.accentGlow },
  pillSub: { fontSize: 10, color: T.dim, marginTop: 2 },
  badge: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    gap: 6,
  },
  badgeDot: { width: 6, height: 6, borderRadius: 3 },
  badgeText: { fontSize: 12, fontWeight: "700" },
  progressTrack: {
    backgroundColor: T.card,
    overflow: "hidden",
    flex: 1,
  },
  metricChip: {
    flex: 1,
    backgroundColor: T.surface,
    borderRadius: 14,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: T.border,
  },
  metricLabel: {
    fontSize: 10,
    color: T.dim,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.6,
  },
  metricValue: { fontSize: 16, fontWeight: "800", fontFamily: MONO, marginTop: 2 },
  metricUnit: { fontSize: 10, fontWeight: "500", color: T.dim },
  statRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 7,
  },
  statLabel: { fontSize: 13, color: T.dim, fontWeight: "500" },
  statValue: { fontSize: 13, color: T.text, fontWeight: "700", fontFamily: MONO },
  sparkRow: { flexDirection: "row", alignItems: "flex-end" },
});
