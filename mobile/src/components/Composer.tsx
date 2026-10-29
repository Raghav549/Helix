import React, { useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { useTheme } from '../hooks/useTheme';
import { WaveGlow } from './WaveGlow';

export interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  onAttach: () => void;
  busy: boolean;
  toolsEnabled: boolean;
  onToggleTools: () => void;
  disabled?: boolean;
  placeholder?: string;
}

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  onAttach,
  busy,
  toolsEnabled,
  onToggleTools,
  disabled,
  placeholder = 'Ask anything, or paste text to work on…',
}: ComposerProps) {
  const theme = useTheme();
  const [focused, setFocused] = useState(false);
  const canSend = value.trim().length > 0 && !disabled;

  return (
    <View style={[styles.wrap, { backgroundColor: theme.palette.paper, borderColor: theme.palette.line }]}>
      <WaveGlow active={busy} height={84} />
      <View style={[styles.inner, { backgroundColor: theme.palette.paper, borderColor: focused ? theme.palette.lineStrong : theme.palette.line }]}>
        <TextInput
          style={[styles.input, { color: theme.palette.ink }]}
          value={value}
          onChangeText={onChange}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          placeholderTextColor={theme.palette.mutedSoft}
          multiline
          maxLength={4000}
          accessibilityLabel="Message Beta by helix"
          textAlignVertical="top"
          blurOnSubmit={false}
        />
        <View style={styles.row}>
          <Pressable
            onPress={onAttach}
            disabled={busy || disabled}
            accessibilityRole="button"
            accessibilityLabel="Attach an image for analysis"
            style={({ pressed }) => [styles.iconButton, { borderColor: theme.palette.line, opacity: pressed ? 0.6 : 1 }]}
          >
            <Text style={[styles.icon, { color: theme.palette.ink }]}>＋</Text>
          </Pressable>
          <Pressable
            onPress={onToggleTools}
            accessibilityRole="switch"
            accessibilityState={{ checked: toolsEnabled }}
            accessibilityLabel="Use tools (symbolic math, sandbox, local document search)"
            style={styles.toolToggle}
          >
            <View style={[styles.pill, {
              backgroundColor: toolsEnabled ? theme.palette.paperTint : 'transparent',
              borderColor: toolsEnabled ? theme.palette.pink : theme.palette.line,
            }]}>
              <Text style={[styles.pillText, { color: toolsEnabled ? theme.palette.pink : theme.palette.muted }]}>
                {toolsEnabled ? 'Tools on' : 'Tools'}
              </Text>
            </View>
          </Pressable>
          <View style={{ flex: 1 }} />
          {busy ? (
            <Pressable
              onPress={onStop}
              accessibilityRole="button"
              accessibilityLabel="Stop generating"
              style={({ pressed }) => [styles.send, { backgroundColor: theme.palette.ink, opacity: pressed ? 0.7 : 1 }]}
            >
              <ActivityIndicator size="small" color={theme.palette.paper} />
            </Pressable>
          ) : (
            <Pressable
              onPress={onSend}
              disabled={!canSend}
              accessibilityRole="button"
              accessibilityLabel="Send message"
              accessibilityState={{ disabled: !canSend }}
              style={({ pressed }) => [
                styles.send,
                { backgroundColor: canSend ? theme.palette.ink : theme.palette.line, opacity: pressed ? 0.7 : 1 },
              ]}
            >
              <Text style={[styles.sendGlyph, { color: canSend ? theme.palette.paper : theme.palette.mutedSoft }]}>↑</Text>
            </Pressable>
          )}
        </View>
      </View>
      <Text style={[styles.note, { color: theme.palette.mutedSoft }]}>
        Beta can be wrong. Tool answers name the tool that produced them.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { borderTopWidth: 1, paddingTop: 8 },
  inner: { borderRadius: 24, borderWidth: 1, paddingHorizontal: 14, paddingTop: 10, paddingBottom: 8, marginHorizontal: 12 },
  input: { minHeight: 44, maxHeight: 140, fontSize: 16, lineHeight: 22, paddingTop: 2 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 6 },
  iconButton: { width: 36, height: 36, borderRadius: 18, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  icon: { fontSize: 18, lineHeight: 22, fontWeight: '600' },
  toolToggle: { paddingVertical: 4 },
  pill: { borderRadius: 999, borderWidth: 1, paddingHorizontal: 12, height: 30, justifyContent: 'center' },
  pillText: { fontSize: 12.5, fontWeight: '700' },
  send: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center' },
  sendGlyph: { fontSize: 20, fontWeight: '800', lineHeight: 24 },
  note: { fontSize: 10.5, textAlign: 'center', marginTop: 6, marginBottom: 4 },
});
