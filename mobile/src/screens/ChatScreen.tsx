import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Constants from 'expo-constants';
import * as ImagePicker from 'expo-image-picker';
import * as SecureStore from 'expo-secure-store';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { ApiError, HelixClient, type Credentials } from '../api/client';
import { Composer } from '../components/Composer';
import { MessageBubble } from '../components/MessageBubble';
import { QuickActions, type QuickAction } from '../components/QuickActions';
import { SettingsSheet } from '../components/SettingsSheet';
import { Sidebar } from '../components/Sidebar';
import { Wordmark } from '../components/Logo';
import { useTheme } from '../hooks/useTheme';
import { formatAnalysis, quickActionPrompt } from '../lib/analysis';
import { greetingFor, normalizeBaseUrl } from '../lib/sse';
import {
  SessionRepository,
  appendMessage,
  contextFor,
  emptySession,
  newId,
  sortSessions,
  updateMessage,
  type Session,
  type StoredMessage,
} from '../state/sessions';
import type { ThemeMode } from '../theme/tokens';

const DEFAULT_BASE_URL =
  (Constants.expoConfig?.extra as { apiBaseUrl?: string } | undefined)?.apiBaseUrl ??
  'https://helix-web-lemon.vercel.app';
const BASE_URL_KEY = 'helix.baseUrl.v1';
const THEME_KEY = 'helix.theme.v1';
const CREDENTIALS_KEY = 'helix.basicAuth.v1';

const repo = new SessionRepository(AsyncStorage);

export function ChatScreen() {
  const insets = useSafeAreaInsets();
  const theme = useTheme();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [toolsEnabled, setToolsEnabled] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [themeMode, setThemeMode] = useState<ThemeMode>('dark');
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL);
  const [credentials, setCredentials] = useState<Credentials | null>(null);
  const cancelRef = useRef<{ cancelled: boolean } | null>(null);
  const sessionsRef = useRef<Session[]>([]);
  const listRef = useRef<FlatList<StoredMessage>>(null);
  const pendingContent = useRef<Map<string, string>>(new Map());

  const client = useMemo(() => new HelixClient({ baseUrl, auth: credentials }), [baseUrl, credentials]);
  sessionsRef.current = sessions;
  const active = sessions.find((session) => session.id === activeId) ?? null;
  const messages = active?.messages ?? [];
  const greeting = useMemo(() => greetingFor(new Date().getHours()), []);

  useEffect(() => {
    let mounted = true;
    (async () => {
      const [loaded, storedUrl, storedTheme, storedAuth] = await Promise.all([
        repo.load(),
        AsyncStorage.getItem(BASE_URL_KEY),
        AsyncStorage.getItem(THEME_KEY),
        SecureStore.getItemAsync(CREDENTIALS_KEY).catch(() => null),
      ]);
      if (!mounted) return;
      const valid = storedUrl ? normalizeBaseUrl(storedUrl) : null;
      if (valid) setBaseUrl(valid);
      if (storedTheme === 'light' || storedTheme === 'dark' || storedTheme === 'system') setThemeMode(storedTheme);
      if (storedAuth) {
        try {
          const parsed = JSON.parse(storedAuth) as Credentials;
          if (typeof parsed?.username === 'string' && typeof parsed?.password === 'string') setCredentials(parsed);
        } catch {
          // A corrupt credential blob is ignored; the user can re-enter it.
        }
      }
      setSessions(loaded);
      setActiveId(loaded[0]?.id ?? null);
    })().catch(() => undefined);
    return () => {
      mounted = false;
    };
  }, []);

  const commit = useCallback((next: Session[]) => {
    const sorted = sortSessions(next);
    sessionsRef.current = sorted;
    setSessions(sorted);
    repo.save(sorted).catch(() => undefined);
  }, []);

  const mutateSession = useCallback(
    (sessionId: string, updater: (session: Session) => Session) => {
      commit(sessionsRef.current.map((session) => (session.id === sessionId ? updater(session) : session)));
    },
    [commit],
  );

  const ensureVisible = useCallback(() => {
    requestAnimationFrame(() => listRef.current?.scrollToEnd({ animated: true }));
  }, []);

  const ensureSession = useCallback((): Session => {
    const existing = sessionsRef.current.find((session) => session.id === activeId);
    if (existing) return existing;
    const fresh = emptySession();
    commit([fresh, ...sessionsRef.current]);
    setActiveId(fresh.id);
    return fresh;
  }, [activeId, commit]);

  const flushPending = useCallback(
    (sessionId: string, messageId: string) => {
      const text = pendingContent.current.get(messageId);
      if (text === undefined) return;
      pendingContent.current.delete(messageId);
      mutateSession(sessionId, (session) => updateMessage(session, messageId, { content: text, streaming: true }));
    },
    [mutateSession],
  );

  const runTurn = useCallback(
    async (session: Session, text: string) => {
      const assistantId = newId('m');
      const history = contextFor(session);
      const next = sessionsRef.current.map((item) =>
        item.id === session.id
          ? appendMessage(
              appendMessage(item, { id: newId('m'), role: 'user', content: text, createdAt: Date.now() }),
              { id: assistantId, role: 'assistant', content: '', createdAt: Date.now(), streaming: true },
            )
          : item,
      );
      commit(next);
      ensureVisible();

      setBusy(true);
      const cancelState = { cancelled: false };
      cancelRef.current = cancelState;
      let buffer = '';
      let lastFlush = 0;

      try {
        const reply = await client.stream(
          { message: text, history, maxNewTokens: 220, signal: cancelState },
          {
            onMeta: (meta) => {
              const sources = Array.isArray(meta.sources) ? meta.sources : [];
              if (sources.length > 0) {
                mutateSession(session.id, (item) => updateMessage(item, assistantId, { sources }));
              } else {
              }
            },
            onDelta: (delta) => {
              buffer += delta;
              const time = Date.now();
              if (time - lastFlush > 60) {
                lastFlush = time;
                pendingContent.current.set(assistantId, buffer);
                flushPending(session.id, assistantId);
                ensureVisible();
              }
            },
            onDone: () => undefined,
            onError: () => undefined,
          },
        );
        mutateSession(session.id, (item) =>
          updateMessage(item, assistantId, { content: reply, streaming: false, error: undefined }),
        );
      } catch (error) {
        const apiError = error instanceof ApiError ? error : new ApiError('offline', 'The request failed.');
        const partial = buffer;
        mutateSession(session.id, (item) =>
          updateMessage(item, assistantId, {
            content: partial,
            streaming: false,
            error: apiError.kind === 'cancelled' ? 'Generation stopped.' : apiError.message,
          }),
        );
      } finally {
        pendingContent.current.delete(assistantId);
        setBusy(false);
        cancelRef.current = null;
        ensureVisible();
      }
    },
    [client, commit, ensureVisible, flushPending, mutateSession],
  );

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || busy) return;
    const session = ensureSession();
    setInput('');
    void runTurn(session, text);
  }, [busy, ensureSession, input, runTurn]);

  const stop = useCallback(() => {
    if (cancelRef.current) cancelRef.current.cancelled = true;
  }, []);

  const pushError = useCallback(
    (sessionId: string, message: string) => {
      mutateSession(sessionId, (session) =>
        appendMessage(session, { id: newId('m'), role: 'assistant', content: '', createdAt: Date.now(), error: message }),
      );
    },
    [mutateSession],
  );

  const retry = useCallback(
    (failed: StoredMessage) => {
      if (busy) return;
      const session = sessionsRef.current.find((item) => item.id === activeId);
      if (!session) return;
      const index = session.messages.findIndex((message) => message.id === failed.id);
      const previousUser = [...session.messages.slice(0, Math.max(0, index))].reverse().find((m) => m.role === 'user');
      if (!previousUser) return;
      const trimmed: Session = { ...session, messages: session.messages.filter((message) => message.id !== failed.id) };
      commit(sessionsRef.current.map((item) => (item.id === session.id ? trimmed : item)));
      void runTurn(trimmed, previousUser.content);
    },
    [activeId, busy, commit, runTurn],
  );

  const analyzeImage = useCallback(async () => {
    if (busy) return;
    const session = ensureSession();
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync().catch(() => null);
    if (!permission?.granted) {
      pushError(session.id, 'Photo permission was not granted, so no image was read.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85 }).catch(() => null);
    if (!picked || picked.canceled || picked.assets.length === 0) return;
    const asset = picked.assets[0];
    if (!asset) return;
    const label = asset.fileName ?? 'selected image';
    mutateSession(session.id, (item) =>
      appendMessage(item, { id: newId('m'), role: 'user', content: `Analyse this image: ${label}`, createdAt: Date.now() }),
    );
    setBusy(true);
    try {
      const payload = await client.analyzeImage({ uri: asset.uri, name: label, type: asset.mimeType ?? 'image/jpeg' });
      mutateSession(session.id, (item) =>
        appendMessage(item, {
          id: newId('m'),
          role: 'assistant',
          content: formatAnalysis(payload, label),
          createdAt: Date.now(),
        }),
      );
    } catch (error) {
      pushError(session.id, error instanceof ApiError ? error.message : 'The image tool could not be reached.');
    } finally {
      setBusy(false);
      ensureVisible();
    }
  }, [busy, client, ensureSession, ensureVisible, mutateSession, pushError]);

  const onQuickAction = useCallback(
    (action: QuickAction) => {
      if (action.id === 'image') {
        void analyzeImage();
        return;
      }
      setToolsEnabled(action.id === 'math');
      setInput(quickActionPrompt(action.id));
    },
    [analyzeImage],
  );

  const newChat = useCallback(() => {
    setDrawerOpen(false);
    const existingEmpty = sessionsRef.current.find((session) => session.messages.length === 0);
    if (existingEmpty) {
      setActiveId(existingEmpty.id);
      return;
    }
    const fresh = emptySession();
    commit([fresh, ...sessionsRef.current]);
    setActiveId(fresh.id);
  }, [commit]);

  const deleteSession = useCallback(
    (id: string) => {
      commit(sessionsRef.current.filter((session) => session.id !== id));
      if (activeId === id) setActiveId(sessionsRef.current[0]?.id ?? null);
    },
    [activeId, commit],
  );

  const cycleTheme = useCallback(() => {
    const next: ThemeMode = themeMode === 'system' ? 'light' : themeMode === 'light' ? 'dark' : 'system';
    setThemeMode(next);
    AsyncStorage.setItem(THEME_KEY, next).catch(() => undefined);
  }, [themeMode]);

  const saveSettings = useCallback(
    (value: { baseUrl: string | null; auth: Credentials | null }) => {
      setBaseUrl(value.baseUrl ?? DEFAULT_BASE_URL);
      setCredentials(value.auth);
      setSettingsOpen(false);
      AsyncStorage.setItem(BASE_URL_KEY, value.baseUrl ?? DEFAULT_BASE_URL).catch(() => undefined);
      const write = value.auth
        ? SecureStore.setItemAsync(CREDENTIALS_KEY, JSON.stringify(value.auth))
        : SecureStore.deleteItemAsync(CREDENTIALS_KEY);
      write.catch(() => undefined);
      setHealthChecked(false);
    },
    [],
  );

  const testConnection = useCallback(async (url: string, auth: Credentials | null) => {
    try {
      const payload = await new HelixClient({ baseUrl: url, auth }).health();
      const ready = payload.model_ready === true;
      return {
        ok: ready,
        message: ready
          ? 'Connected. The model is loaded.'
          : 'Connected, but the server reports the model is not loaded.',
      };
    } catch (error) {
      return { ok: false, message: error instanceof Error ? error.message : 'Connection failed.' };
    }
  }, []);

  const clearHistory = useCallback(() => {
    void repo.clear();
    sessionsRef.current = [];
    setSessions([]);
    setActiveId(null);
    setSettingsOpen(false);
  }, []);

  return (
    <View style={[styles.root, { backgroundColor: theme.palette.paper, paddingTop: insets.top }]}>
      <View style={[styles.header, { borderBottomColor: theme.palette.line }]}>
        <Pressable onPress={() => setDrawerOpen(true)} accessibilityRole="button" accessibilityLabel="Open menu" style={styles.menu} hitSlop={8}>
          <View style={[styles.menuLine, { backgroundColor: theme.palette.ink }]} />
          <View style={[styles.menuLine, { backgroundColor: theme.palette.ink, width: 14 }]} />
        </Pressable>
        <Wordmark compact />
        <Pressable onPress={newChat} accessibilityRole="button" accessibilityLabel="New chat" style={styles.menu} hitSlop={8}>
          <Text style={[styles.newGlyph, { color: theme.palette.ink }]}>＋</Text>
        </Pressable>
      </View>

      <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <MessageBubble message={item} onRetry={item.error ? () => retry(item) : undefined} />
          )}
          ListHeaderComponent={
            messages.length === 0 ? (
              <View style={styles.intro}>
                <Text style={[styles.greeting, { color: theme.palette.ink }]} accessibilityRole="header">{greeting}</Text>
                <Text style={[styles.sub, { color: theme.palette.muted }]}>
                  How can I help you today?
                </Text>
                <QuickActions onSelect={onQuickAction} disabled={busy} />
              </View>
            ) : null
          }
          contentContainerStyle={styles.list}
          onContentSizeChange={ensureVisible}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="interactive"
        />

        <View style={{ paddingBottom: Math.max(insets.bottom, 8) }}>
          <Composer
            value={input}
            onChange={setInput}
            onSend={send}
            onStop={stop}
            onAttach={() => void analyzeImage()}
            busy={busy}
            toolsEnabled={toolsEnabled}
            onToggleTools={() => setToolsEnabled((value) => !value)}
            placeholder={
              toolsEnabled
                ? 'Ask for math, a sandbox run or a local-document search…'
                : 'Ask anything, or paste text to work on…'
            }
          />
        </View>
      </KeyboardAvoidingView>

      <Sidebar
        visible={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        sessions={sessions}
        activeId={activeId}
        onSelect={(id) => {
          setActiveId(id);
          setDrawerOpen(false);
        }}
        onNewChat={newChat}
        onDelete={deleteSession}
        themeMode={themeMode}
        onCycleTheme={cycleTheme}
        onOpenSettings={() => {
          setDrawerOpen(false);
          setSettingsOpen(true);
        }}
      />

      <SettingsSheet
        visible={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        baseUrl={baseUrl}
        credentials={credentials}
        onSave={saveSettings}
        onClearHistory={clearHistory}
        testConnection={testConnection}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: 1,
  },
  menu: { width: 34, height: 30, justifyContent: 'center', gap: 5 },
  menuLine: { height: 2.5, width: 20, borderRadius: 2 },
  newGlyph: { fontSize: 22, fontWeight: '600', alignSelf: 'flex-end' },
  list: { paddingHorizontal: 16, paddingTop: 8, paddingBottom: 12 },
  intro: { paddingTop: 14 },
  greeting: { fontSize: 30, fontWeight: '800', letterSpacing: -0.9, lineHeight: 36 },
  sub: { fontSize: 14.5, lineHeight: 21, marginTop: 8, marginBottom: 18 },
});
