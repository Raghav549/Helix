/**
 * Local conversation store.
 *
 * Messages live on the device only (AsyncStorage). The server keeps no history
 * for a request beyond the `history` array the app chooses to send, so the
 * store also decides how much context is forwarded.
 */
import { titleFor } from '../lib/sse';

export interface StoredMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: number;
  error?: string;
  streaming?: boolean;
  sources?: Array<{ title?: string; source?: string; score?: number }>;
}

export interface Session {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: StoredMessage[];
}

export interface StorageLike {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
  removeItem(key: string): Promise<void>;
}

export const STORAGE_KEY = 'helix.sessions.v1';
export const MAX_SESSIONS = 40;
export const MAX_CONTEXT_MESSAGES = 8;

let counter = 0;

export function newId(prefix: string): string {
  counter += 1;
  return `${prefix}-${Date.now().toString(36)}-${counter.toString(36)}`;
}

export function emptySession(now = Date.now()): Session {
  return { id: newId('s'), title: 'New chat', createdAt: now, updatedAt: now, messages: [] };
}

export function appendMessage(session: Session, message: StoredMessage): Session {
  const messages = [...session.messages, message];
  const firstUser = messages.find((item) => item.role === 'user');
  return {
    ...session,
    title: session.title === 'New chat' && firstUser ? titleFor(firstUser.content) : session.title,
    updatedAt: message.createdAt,
    messages,
  };
}

export function updateMessage(session: Session, id: string, patch: Partial<StoredMessage>): Session {
  return {
    ...session,
    updatedAt: Date.now(),
    messages: session.messages.map((item) => (item.id === id ? { ...item, ...patch } : item)),
  };
}

/** History sent to the API: completed turns only, trimmed to the tail. */
export function contextFor(session: Session): Array<{ role: 'user' | 'assistant'; content: string }> {
  return session.messages
    .filter((item) => item.role !== 'system' && !item.error && !item.streaming && item.content.trim().length > 0)
    .slice(-MAX_CONTEXT_MESSAGES)
    .map((item) => ({ role: item.role as 'user' | 'assistant', content: item.content }));
}

export function sortSessions(sessions: Session[]): Session[] {
  return [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
}

export function parseSessions(raw: string | null): Session[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is Session => {
        const candidate = item as Partial<Session>;
        return Boolean(candidate) && typeof candidate.id === 'string' && Array.isArray(candidate.messages);
      })
      .map((item) => ({
        ...item,
        title: typeof item.title === 'string' && item.title ? item.title : 'New chat',
        createdAt: Number(item.createdAt) || Date.now(),
        updatedAt: Number(item.updatedAt) || Date.now(),
        messages: item.messages
          .filter((m) => m && typeof m.content === 'string' && (m.role === 'user' || m.role === 'assistant' || m.role === 'system'))
          .map((m) => ({
            id: typeof m.id === 'string' ? m.id : newId('m'),
            role: m.role,
            content: m.content,
            createdAt: Number(m.createdAt) || Date.now(),
            error: typeof m.error === 'string' ? m.error : undefined,
            sources: Array.isArray(m.sources) ? m.sources : undefined,
            streaming: false,
          })),
      }))
      .slice(0, MAX_SESSIONS);
  } catch {
    return [];
  }
}

export class SessionRepository {
  private storage: StorageLike;
  private cache: Session[] | null = null;

  constructor(storage: StorageLike) {
    this.storage = storage;
  }

  async load(): Promise<Session[]> {
    if (this.cache) return this.cache;
    const raw = await this.storage.getItem(STORAGE_KEY);
    this.cache = sortSessions(parseSessions(raw));
    return this.cache;
  }

  async save(sessions: Session[]): Promise<void> {
    const trimmed = sortSessions(sessions).slice(0, MAX_SESSIONS).map((session) => ({
      ...session,
      messages: session.messages.map((message) => ({ ...message, streaming: false })),
    }));
    this.cache = trimmed;
    await this.storage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  }

  async clear(): Promise<void> {
    this.cache = [];
    await this.storage.removeItem(STORAGE_KEY);
  }
}
