import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Defs, Line, LinearGradient, Path, Stop } from 'react-native-svg';
import { useTheme } from '../hooks/useTheme';

export function Mark({ size = 22 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 32 32" accessibilityLabel="Beta by helix">
      <Defs>
        <LinearGradient id="mark" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0" stopColor="#ff2f86" />
          <Stop offset="0.55" stopColor="#ff6a56" />
          <Stop offset="1" stopColor="#ffa63d" />
        </LinearGradient>
      </Defs>
      {/* Left flame swoosh */}
      <Path d="M8 6 C4 6 2 14 6 20 C8 26 14 26 16 20 C18 14 16 6 12 6 C10 6 8 6 8 6 Z" fill="url(#mark)" />
      {/* Right flame swoosh */}
      <Path d="M20 6 C24 6 26 14 22 20 C20 26 14 26 12 20 C10 14 12 6 16 6 C18 6 20 6 20 6 Z" fill="url(#mark)" opacity="0.9" />
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
