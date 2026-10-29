import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../hooks/useTheme';

export interface QuickAction {
  id: 'image' | 'summarize' | 'code' | 'brainstorm' | 'math';
  title: string;
  hint: string;
  glyph: string;
}

export const QUICK_ACTIONS: QuickAction[] = [
  { id: 'image', title: 'Analyse an image', hint: 'Local vision tool', glyph: '◨' },
  { id: 'summarize', title: 'Summarize text', hint: 'Paste text to condense', glyph: '≡' },
  { id: 'code', title: 'Write code', hint: 'Explain or draft a function', glyph: '</>' },
  { id: 'math', title: 'Solve math', hint: 'SymPy symbolic tool', glyph: '∑' },
];

export function QuickActions({ onSelect, disabled }: { onSelect: (action: QuickAction) => void; disabled?: boolean }) {
  const theme = useTheme();
  return (
    <View style={styles.grid}>
      {QUICK_ACTIONS.map((action) => (
        <Pressable
          key={action.id}
          onPress={() => onSelect(action)}
          disabled={disabled}
          accessibilityRole="button"
          accessibilityLabel={`${action.title}. ${action.hint}`}
          style={({ pressed }) => [
            styles.card,
            {
              backgroundColor: theme.palette.paperSoft,
              borderColor: pressed ? theme.palette.pink : theme.palette.line,
              opacity: disabled ? 0.5 : 1,
            },
          ]}
        >
          <Text style={[styles.glyph, { color: theme.palette.pink }]}>{action.glyph}</Text>
          <Text style={[styles.title, { color: theme.palette.ink }]} numberOfLines={1}>{action.title}</Text>
          <Text style={[styles.hint, { color: theme.palette.muted }]} numberOfLines={1}>{action.hint}</Text>
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  card: {
    flexGrow: 1,
    flexBasis: '47%',
    minHeight: 96,
    borderRadius: 18,
    borderWidth: 1,
    padding: 14,
    justifyContent: 'space-between',
  },
  glyph: { fontSize: 18, fontWeight: '800' },
  title: { fontSize: 15, fontWeight: '700', marginTop: 10 },
  hint: { fontSize: 12, marginTop: 2 },
});
