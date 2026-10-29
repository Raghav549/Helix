import React, { useEffect, useRef } from 'react';
import { Animated, Modal, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../hooks/useTheme';
import { formatRelativeTime } from '../lib/format';
import type { Session } from '../state/sessions';
import { Wordmark } from './Logo';
import type { ThemeMode } from '../theme/tokens';

export interface SidebarProps {
  visible: boolean;
  onClose: () => void;
  sessions: Session[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
  themeMode: ThemeMode;
  onCycleTheme: () => void;
  onOpenSettings: () => void;
}

export function Sidebar({
  visible,
  onClose,
  sessions,
  activeId,
  onSelect,
  onNewChat,
  onDelete,
  themeMode,
  onCycleTheme,
  onOpenSettings,
}: SidebarProps) {
  const theme = useTheme();
  const slide = useRef(new Animated.Value(-24)).current;
  const fade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fade, { toValue: visible ? 1 : 0, duration: 180, useNativeDriver: true }),
      Animated.timing(slide, { toValue: visible ? 0 : -24, duration: 200, useNativeDriver: true }),
    ]).start();
  }, [visible, fade, slide]);


  return (
    <Modal visible={visible} transparent animationType="none" onRequestClose={onClose} statusBarTranslucent>
      <Animated.View style={[StyleSheet.absoluteFill, { backgroundColor: 'rgba(12,12,15,0.45)', opacity: fade }]}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} accessibilityLabel="Close menu" />
      </Animated.View>
      <Animated.View
        style={[
          styles.panel,
          { backgroundColor: theme.palette.paper, borderRightColor: theme.palette.line, transform: [{ translateX: slide }] },
        ]}
      >
        <View style={styles.header}>
          <Wordmark />
          <Pressable onPress={onClose} accessibilityRole="button" accessibilityLabel="Close menu" hitSlop={10}>
            <Text style={[styles.close, { color: theme.palette.muted }]}>✕</Text>
          </Pressable>
        </View>

        <Pressable
          onPress={onNewChat}
          accessibilityRole="button"
          accessibilityLabel="Start a new chat"
          style={({ pressed }) => [styles.newChat, { borderColor: theme.palette.line, backgroundColor: pressed ? theme.palette.paperTint : theme.palette.paperSoft }]}
        >
          <Text style={[styles.newChatText, { color: theme.palette.ink }]}>＋  New chat</Text>
        </Pressable>

        <Text style={[styles.section, { color: theme.palette.mutedSoft }]}>History</Text>
        <ScrollView style={styles.list} contentContainerStyle={{ paddingBottom: 12 }}>
          {sessions.length === 0 && (
            <Text style={[styles.empty, { color: theme.palette.muted }]}>Conversations you start appear here, stored on this device.</Text>
          )}
          {sessions.map((session) => {
            const active = session.id === activeId;
            return (
              <View key={session.id} style={[styles.item, active && { backgroundColor: theme.palette.paperTint }]}>
                <Pressable
                  onPress={() => onSelect(session.id)}
                  accessibilityRole="button"
                  accessibilityLabel={`Open chat: ${session.title}`}
                  style={styles.itemMain}
                >
                  <Text numberOfLines={1} style={[styles.itemTitle, { color: theme.palette.ink }]}>{session.title}</Text>
                  <Text style={[styles.itemMeta, { color: theme.palette.mutedSoft }]}>{formatRelativeTime(session.updatedAt)}</Text>
                </Pressable>
                <Pressable
                  onPress={() => onDelete(session.id)}
                  accessibilityRole="button"
                  accessibilityLabel={`Delete chat: ${session.title}`}
                  hitSlop={8}
                  style={styles.delete}
                >
                  <Text style={{ color: theme.palette.mutedSoft, fontSize: 15 }}>🗑</Text>
                </Pressable>
              </View>
            );
          })}
        </ScrollView>

        <View style={[styles.footer, { borderTopColor: theme.palette.line }]}>
          <Pressable onPress={onOpenSettings} accessibilityRole="button" accessibilityLabel="Open settings" style={styles.footerRow}>
            <Text style={[styles.footerText, { color: theme.palette.ink }]}>Settings</Text>
            <Text style={[styles.footerHint, { color: theme.palette.mutedSoft }]}>API, theme</Text>
          </Pressable>
          <Pressable onPress={onCycleTheme} accessibilityRole="button" accessibilityLabel={`Theme: ${themeMode}. Change theme`} style={styles.footerRow}>
            <Text style={[styles.footerText, { color: theme.palette.ink }]}>Theme</Text>
            <Text style={[styles.footerHint, { color: theme.palette.mutedSoft }]}>{themeMode}</Text>
          </Pressable>
        </View>
      </Animated.View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  panel: { position: 'absolute', top: 0, bottom: 0, left: 0, width: '84%', maxWidth: 360, borderRightWidth: 1, paddingTop: 52, paddingHorizontal: 16 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 },
  close: { fontSize: 18 },
  newChat: { borderWidth: 1, borderRadius: 14, paddingVertical: 12, paddingHorizontal: 14 },
  newChatText: { fontSize: 15, fontWeight: '700' },
  section: { fontSize: 11, fontWeight: '800', letterSpacing: 1, textTransform: 'uppercase', marginTop: 22, marginBottom: 8 },
  list: { flex: 1 },
  empty: { fontSize: 13, lineHeight: 19, paddingVertical: 8 },
  item: { flexDirection: 'row', alignItems: 'center', borderRadius: 12, marginBottom: 2 },
  itemMain: { flex: 1, paddingVertical: 10, paddingHorizontal: 10 },
  itemTitle: { fontSize: 14.5, fontWeight: '600' },
  itemMeta: { fontSize: 11.5, marginTop: 2 },
  delete: { paddingHorizontal: 10, paddingVertical: 10 },
  footer: { borderTopWidth: 1, paddingVertical: 8 },
  footerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 10 },
  footerText: { fontSize: 14, fontWeight: '600' },
  footerHint: { fontSize: 12.5 },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
});
