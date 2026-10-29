import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useTheme } from '../hooks/useTheme';
import { splitCodeBlocks } from '../lib/markdown';
import type { StoredMessage } from '../state/sessions';

export function MessageBubble({ message, onRetry }: { message: StoredMessage; onRetry?: () => void }) {
  const theme = useTheme();
  const isUser = message.role === 'user';
  const sources = message.sources ?? [];

  return (
    <View style={[styles.wrap, isUser ? styles.userWrap : styles.assistantWrap]}>
      {!isUser && (
        <Text style={[styles.author, { color: theme.palette.muted }]}>
          Beta{message.streaming ? ' · streaming' : ''}
        </Text>
      )}
      <View
        style={[
          styles.bubble,
          isUser
            ? { backgroundColor: theme.palette.ink, borderTopRightRadius: 6 }
            : { backgroundColor: theme.palette.paperSoft, borderColor: theme.palette.line, borderWidth: 1, borderTopLeftRadius: 6 },
        ]}
      >
        {message.content.length === 0 && message.streaming ? (
          <Text style={[styles.body, { color: theme.palette.muted }]}>…</Text>
        ) : (
          splitCodeBlocks(message.content).map((part, index) =>
            part.kind === 'code' ? (
              <View key={index} style={[styles.code, { backgroundColor: theme.palette.paperTint, borderColor: theme.palette.line }]}>
                <Text style={[styles.codeText, { color: theme.palette.inkSoft }]} selectable>{part.value}</Text>
              </View>
            ) : (
              <Text key={index} style={[styles.body, { color: isUser ? theme.palette.paper : theme.palette.ink }]} selectable>
                {part.value.trim()}
              </Text>
            ),
          )
        )}
      </View>
      {sources.length > 0 && (
        <View style={styles.sources}>
          <Text style={[styles.author, { color: theme.palette.muted }]}>Sources</Text>
          {sources.slice(0, 4).map((source, index) => (
            <Text key={index} style={[styles.source, { color: theme.palette.muted }]} numberOfLines={2}>
              • {source.title || source.source || 'local document'}
            </Text>
          ))}
        </View>
      )}
      {message.error && (
        <View style={styles.errorRow}>
          <Text style={[styles.error, { color: theme.palette.danger }]} accessibilityLiveRegion="polite">{message.error}</Text>
          {onRetry && (
            <Text onPress={onRetry} style={[styles.retry, { color: theme.palette.pink }]} accessibilityRole="button">
              Retry
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: 16, maxWidth: '100%' },
  userWrap: { alignItems: 'flex-end' },
  assistantWrap: { alignItems: 'flex-start' },
  author: { fontSize: 11, fontWeight: '700', letterSpacing: 0.6, textTransform: 'uppercase', marginBottom: 6 },
  bubble: { borderRadius: 18, paddingHorizontal: 14, paddingVertical: 11, maxWidth: '92%' },
  body: { fontSize: 15.5, lineHeight: 23 },
  code: { borderRadius: 12, borderWidth: 1, padding: 10, marginVertical: 6 },
  codeText: { fontFamily: 'monospace', fontSize: 13, lineHeight: 19 },
  sources: { marginTop: 8, maxWidth: '92%' },
  source: { fontSize: 12, lineHeight: 17 },
  errorRow: { marginTop: 8, flexDirection: 'row', gap: 12, alignItems: 'center' },
  error: { fontSize: 12.5, flexShrink: 1 },
  retry: { fontSize: 12.5, fontWeight: '700' },
});
