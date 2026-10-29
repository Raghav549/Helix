import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../hooks/useTheme';

export interface QuickAction {
  id: 'image' | 'summarize' | 'code' | 'brainstorm' | 'analyze' | 'more';
  title: string;
  hint: string;
  glyph: string;
}

export const QUICK_ACTIONS: QuickAction[] = [
  { id: 'image', title: 'Image', hint: 'Generate or inspect an image', glyph: '▭' },
  { id: 'summarize', title: 'Summarize', hint: 'Long text to key points', glyph: '≡' },
  { id: 'code', title: 'Code', hint: 'Apps, websites, scripts', glyph: '</>' },
  { id: 'brainstorm', title: 'Brainstorm', hint: 'Ideas and solutions', glyph: '✦' },
  { id: 'analyze', title: 'Analyze', hint: 'Inspect an uploaded image', glyph: '🔍' },
  { id: 'more', title: 'More', hint: 'Settings and tools', glyph: '•••' },
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
          <Text style={[styles.title, { color: theme.palette.ink }]} numberOfLines={1}>
            {action.title}
          </Text>
          <Text style={[styles.hint, { color: theme.palette.muted }]} numberOfLines={1}>
            {action.hint}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  card: {
    flexGrow: 1,
    flexBasis: '31%',
    minHeight: 100,
    borderRadius: 16,
    borderWidth: 1,
    padding: 12,
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  glyph: { fontSize: 22, fontWeight: '700', marginBottom: 6 },
  title: { fontSize: 13, fontWeight: '800', marginTop: 2 },
  hint: { fontSize: 10.5, marginTop: 2 },
});
