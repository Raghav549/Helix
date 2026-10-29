/** Streaming helpers shared with the tests. Pure functions only. */

export interface StreamEvent {
  meta?: Record<string, unknown>;
  delta?: string;
  done?: boolean;
  error?: string;
  code?: string;
  request_id?: string;
}

export interface ParsedChunk {
  events: StreamEvent[];
  rest: string;
}

/**
 * Extract complete server-sent events from a text buffer.
 *
 * Handles CRLF, multi-line `data:` fields, comment lines and the case where a
 * chunk boundary splits an event. Malformed JSON is reported as an error event
 * instead of being dropped silently.
 */
export function parseSseChunk(buffer: string): ParsedChunk {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const events: StreamEvent[] = [];
  let rest = normalized;
  for (;;) {
    const boundary = rest.indexOf('\n\n');
    if (boundary === -1) break;
    const raw = rest.slice(0, boundary);
    rest = rest.slice(boundary + 2);
    const dataLines = raw
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim());
    if (dataLines.length === 0) continue;
    try {
      const payload = JSON.parse(dataLines.join('\n')) as StreamEvent;
      if (payload && typeof payload === 'object') events.push(payload);
    } catch {
      events.push({ error: 'The server sent an event that was not valid JSON.', code: 'malformed_event' });
    }
  }
  return { events, rest };
}

export function extractReply(events: StreamEvent[]): string {
  return events.map((event) => (typeof event.delta === 'string' ? event.delta : '')).join('');
}

/** A short, stable title for a conversation based on its first message. */
export function titleFor(message: string, max = 48): string {
  const clean = String(message || '').replace(/\s+/g, ' ').trim();
  if (!clean) return 'New chat';
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

export function greetingFor(hour: number, name?: string): string {
  const part = hour < 5 ? 'night' : hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening';
  const person = (name || '').trim();
  return person ? `Good ${part}, ${person}.` : `Good ${part}.`;
}

/** Validate a configurable API base URL: https, or http only for local dev. */
export function normalizeBaseUrl(value: string): string | null {
  const raw = String(value || '').trim().replace(/\/+$/, '');
  if (!raw) return null;
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  const loopback = ['127.0.0.1', 'localhost', '10.0.2.2'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) return null;
  if (url.pathname !== '/' || url.search || url.hash) return null;
  return url.origin;
}
