import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../hooks/useTheme';

export type ActivityKind = 'thinking' | 'generating' | 'documents' | 'tool' | 'idle';

const LABELS: Record<Exclude<ActivityKind, 'idle'>, string> = {
  thinking: 'Thinking',
  generating: 'Generating response',
  documents: 'Reading local documents',
  tool: 'Running a tool',
};

/** A truthful activity indicator. It states what the app is doing right now. */
export function ActivityRow({ kind, elapsedMs }: { kind: ActivityKind; elapsedMs?: number }) {
  const theme = useTheme();
  const pulse = useRef(new Animated.Value(0.35)).current;

  useEffect(() => {
    if (kind === 'idle') return;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 700, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0.35, duration: 700, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [kind, pulse]);

  if (kind === 'idle') return null;
  const seconds = Math.max(0, Math.round((elapsedMs ?? 0) / 1000));

  return (
    <View style={styles.row} accessible accessibilityLiveRegion="polite">
      <Animated.View style={[styles.dot, { backgroundColor: theme.palette.pink, opacity: pulse }]} />
      <Text style={[styles.label, { color: theme.palette.ink }]}>{LABELS[kind]}</Text>
      <Text style={[styles.time, { color: theme.palette.mutedSoft }]}>
        {kind === 'generating' && seconds > 0 ? `${seconds}s` : seconds > 0 ? `${seconds}s elapsed` : ''}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 6 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  label: { fontSize: 13, fontWeight: '600' },
  time: { fontSize: 12 },
});
