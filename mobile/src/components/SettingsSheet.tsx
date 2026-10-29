import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Modal, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { useTheme } from '../hooks/useTheme';
import { normalizeBaseUrl } from '../lib/sse';
import type { Credentials } from '../api/client';

export interface SettingsSheetProps {
  visible: boolean;
  onClose: () => void;
  baseUrl: string;
  credentials: Credentials | null;
  onSave: (value: { baseUrl: string | null; auth: Credentials | null }) => void;
  onClearHistory: () => void;
  testConnection: (url: string, auth: Credentials | null) => Promise<{ ok: boolean; message: string }>;
}

export function SettingsSheet({ visible, onClose, baseUrl, credentials, onSave, onClearHistory, testConnection }: SettingsSheetProps) {
  const theme = useTheme();
  const [url, setUrl] = useState(baseUrl);
  const [username, setUsername] = useState(credentials?.username ?? '');
  const [password, setPassword] = useState(credentials?.password ?? '');
  const [state, setState] = useState<{ busy: boolean; message: string | null; ok: boolean }>({ busy: false, message: null, ok: true });

  useEffect(() => {
    if (!visible) return;
    setUrl(baseUrl);
    setUsername(credentials?.username ?? '');
    setPassword(credentials?.password ?? '');
    setState({ busy: false, message: null, ok: true });
  }, [visible, baseUrl, credentials]);

  const normalized = normalizeBaseUrl(url);
  const auth = username.trim() ? { username: username.trim(), password } : null;

  const runTest = async () => {
    if (!normalized) {
      setState({ busy: false, ok: false, message: 'Enter an https:// origin (http:// only for localhost).' });
      return;
    }
    setState({ busy: true, ok: true, message: null });
    const result = await testConnection(normalized, auth).catch((error: unknown) => ({
      ok: false,
      message: error instanceof Error ? error.message : 'Connection failed.',
    }));
    setState({ busy: false, ok: result.ok, message: result.message });
  };

  const save = () => {
    if (url.trim() && !normalized) {
      setState({ busy: false, ok: false, message: 'That URL is not a plain https origin.' });
      return;
    }
    onSave({ baseUrl: normalized, auth });
    setPassword('');
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose} statusBarTranslucent>
      <View style={styles.backdrop}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} accessibilityLabel="Close settings" />
        <View style={[styles.sheet, { backgroundColor: theme.palette.paper, borderColor: theme.palette.line }]}>
          <ScrollView keyboardShouldPersistTaps="handled">
            <Text style={[styles.title, { color: theme.palette.ink }]}>Settings</Text>

            <Text style={[styles.label, { color: theme.palette.muted }]}>API base URL</Text>
            <TextInput
              style={[styles.input, { color: theme.palette.ink, borderColor: theme.palette.line, backgroundColor: theme.palette.paperSoft }]}
              value={url}
              onChangeText={setUrl}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              placeholder="https://helix-web-lemon.vercel.app"
              placeholderTextColor={theme.palette.mutedSoft}
              accessibilityLabel="API base URL"
            />
            {!normalized && (
              <Text style={[styles.hint, { color: theme.palette.danger }]}>
                Use an https:// origin. Plain http:// is allowed only for localhost during development.
              </Text>
            )}

            <Text style={[styles.label, { color: theme.palette.muted }]}>HTTP Basic auth (optional)</Text>
            <TextInput
              style={[styles.input, { color: theme.palette.ink, borderColor: theme.palette.line, backgroundColor: theme.palette.paperSoft }]}
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
              autoCorrect={false}
              placeholder="username"
              placeholderTextColor={theme.palette.mutedSoft}
              accessibilityLabel="Basic auth username"
            />
            <TextInput
              style={[styles.input, { marginTop: 8, color: theme.palette.ink, borderColor: theme.palette.line, backgroundColor: theme.palette.paperSoft }]}
              value={password}
              onChangeText={setPassword}
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
              placeholder="password"
              placeholderTextColor={theme.palette.mutedSoft}
              accessibilityLabel="Basic auth password"
            />
            <Text style={[styles.hint, { color: theme.palette.mutedSoft }]}>
              Stored in the Android Keystore via expo-secure-store, never in plain storage and never displayed again.
            </Text>

            {state.message && (
              <Text style={[styles.hint, { color: state.ok ? theme.palette.success : theme.palette.danger }]} accessibilityLiveRegion="polite">
                {state.message}
              </Text>
            )}

            <View style={styles.row}>
              <Pressable onPress={save} style={[styles.button, { backgroundColor: theme.palette.ink }]} accessibilityRole="button">
                <Text style={[styles.buttonText, { color: theme.palette.paper }]}>Save</Text>
              </Pressable>
              <Pressable onPress={runTest} style={[styles.button, { borderWidth: 1, borderColor: theme.palette.line }]} accessibilityRole="button">
                {state.busy ? <ActivityIndicator size="small" color={theme.palette.ink} /> : <Text style={[styles.buttonText, { color: theme.palette.ink }]}>Test</Text>}
              </Pressable>
            </View>

            <Text style={[styles.label, { color: theme.palette.muted }]}>Device data</Text>
            <Text style={[styles.hint, { color: theme.palette.muted }]}>
              Chats stay in this app's local storage. Only the messages you send leave the device.
            </Text>
            <Pressable onPress={onClearHistory} style={[styles.button, styles.danger, { borderColor: theme.palette.danger }]} accessibilityRole="button">
              <Text style={[styles.buttonText, { color: theme.palette.danger }]}>Delete all chats on this device</Text>
            </Pressable>

            <Pressable onPress={onClose} style={styles.close} accessibilityRole="button">
              <Text style={[styles.buttonText, { color: theme.palette.muted }]}>Close</Text>
            </Pressable>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(12,12,15,0.45)', justifyContent: 'flex-end' },
  sheet: { borderTopLeftRadius: 24, borderTopRightRadius: 24, borderWidth: 1, padding: 20, paddingBottom: 28, maxHeight: '92%' },
  title: { fontSize: 20, fontWeight: '800', marginBottom: 8 },
  label: { fontSize: 11.5, fontWeight: '800', letterSpacing: 0.8, textTransform: 'uppercase', marginTop: 16, marginBottom: 6 },
  input: { borderWidth: 1, borderRadius: 12, paddingHorizontal: 12, paddingVertical: 11, fontSize: 14.5 },
  hint: { fontSize: 12.5, lineHeight: 18, marginTop: 6 },
  row: { flexDirection: 'row', gap: 10, marginTop: 14 },
  button: { flex: 1, borderRadius: 12, paddingVertical: 12, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: 'transparent' },
  buttonText: { fontSize: 14, fontWeight: '700' },
  danger: { marginTop: 12 },
  close: { alignItems: 'center', paddingVertical: 12, marginTop: 4 },
});
