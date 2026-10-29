import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Defs, Line, LinearGradient, Stop } from 'react-native-svg';
import { useTheme } from '../hooks/useTheme';

export function Mark({ size = 22 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" accessibilityLabel="Beta by helix">
      <Defs>
        <LinearGradient id="mark" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0" stopColor="#ff2f86" />
          <Stop offset="1" stopColor="#ffa63d" />
        </LinearGradient>
      </Defs>
      <Line x1={6} y1={6} x2={18} y2={18} stroke="url(#mark)" strokeWidth={2.6} strokeLinecap="round" />
      <Line x1={18} y1={6} x2={6} y2={18} stroke="url(#mark)" strokeWidth={2.6} strokeLinecap="round" />
      <Line x1={4.5} y1={9.4} x2={19.5} y2={9.4} stroke="url(#mark)" strokeWidth={2.6} strokeLinecap="round" />
      <Line x1={4.5} y1={14.6} x2={19.5} y2={14.6} stroke="url(#mark)" strokeWidth={2.6} strokeLinecap="round" />
    </Svg>
  );
}

export function Wordmark({ compact = false }: { compact?: boolean }) {
  const theme = useTheme();
  return (
    <View style={styles.row}>
      <Mark size={compact ? 20 : 24} />
      <Text style={[styles.beta, { color: theme.palette.ink, fontSize: compact ? 16 : 19 }]}>Beta</Text>
      <Text style={[styles.by, { color: theme.palette.muted }]}>by helix</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  beta: { fontWeight: '800', letterSpacing: -0.4 },
  by: { fontSize: 12, fontWeight: '600', marginTop: 3 },
});
