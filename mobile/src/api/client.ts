/**
 * HELIX HTTP client for React Native.
 *
 * React Native has no ReadableStream on the response, so streaming is done
 * with XMLHttpRequest (incremental `onprogress`) and falls back to a single
 * `/api/chat` request when incremental updates are unavailable. Every error is
 * mapped to a typed `ApiError` so the UI never has to guess.
 */
import type { StreamEvent } from '../lib/sse';
import { parseSseChunk } from '../lib/sse';
import { basicAuthHeader } from '../lib/base64';

export type ApiErrorKind =
  | 'offline'
  | 'unauthorized'
  | 'rate_limited'
  | 'too_large'
  | 'not_found'
  | 'bad_gateway'
  | 'model_unavailable'
  | 'server'
  | 'timeout'
  | 'cancelled'
  | 'malformed';

export class ApiError extends Error {
  kind: ApiErrorKind;
  status: number;
  requestId: string | null;

  constructor(kind: ApiErrorKind, message: string, status = 0, requestId: string | null = null) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.requestId = requestId;
  }
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatMeta {
  model?: string;
  sources?: Array<{ title?: string; source?: string; score?: number }>;
  grounded_in_local_documents?: boolean;
  notice?: string;
  degraded_reason?: string;
  [key: string]: unknown;
}

export interface StreamHandlers {
  onMeta?: (meta: ChatMeta) => void;
  onDelta: (text: string) => void;
  onDone: (full: string) => void;
  onError: (error: ApiError) => void;
}

export interface StreamOptions {
  message: string;
  history?: ChatMessage[];
  maxNewTokens?: number;
  temperature?: number;
  signal?: { cancelled: boolean };
}

function kindForStatus(status: number): ApiErrorKind {
  if (status === 401 || status === 403) return 'unauthorized';
  if (status === 404) return 'not_found';
  if (status === 413) return 'too_large';
  if (status === 429) return 'rate_limited';
  if (status === 502 || status === 503 || status === 504) return 'bad_gateway';
  if (status >= 500) return 'server';
  return 'server';
}

const MESSAGES: Record<ApiErrorKind, string> = {
  offline: 'No connection to HELIX. Check your network and try again.',
  unauthorized: 'The server rejected these credentials. Add them in Settings.',
  rate_limited: 'Too many requests. Wait a moment before trying again.',
  too_large: 'That request was too large for the server.',
  not_found: 'The HELIX API route is missing on this deployment.',
  bad_gateway: 'The HELIX model service is not ready yet. Try again shortly.',
  model_unavailable: 'The model is not loaded on the server right now.',
  server: 'The server returned an error.',
  timeout: 'The request timed out.',
  cancelled: 'Generation stopped.',
  malformed: 'The server sent a response this app could not read.',
};

/** Build an ApiError from an HTTP response body that may or may not be JSON. */
export function errorFromResponse(status: number, body: string, requestId?: string | null): ApiError {
  let message = '';
  let kind = kindForStatus(status);
  let id = requestId ?? null;
  const text = String(body || '').trim();
  if (text.startsWith('{')) {
    try {
      const parsed = JSON.parse(text) as { error?: string; code?: string; request_id?: string };
      if (typeof parsed.error === 'string') message = parsed.error;
      if (parsed.code === 'model_unavailable') kind = 'model_unavailable';
      if (typeof parsed.request_id === 'string') id = parsed.request_id;
    } catch {
      message = '';
    }
  }
  if (!message) message = MESSAGES[kind];
  const withId = id ? `${message} (ref ${id.slice(0, 8)})` : message;
  return new ApiError(kind, withId, status, id);
}

export interface Credentials {
  username: string;
  password: string;
}

export interface ClientOptions {
  baseUrl: string;
  timeoutMs?: number;
  auth?: Credentials | null;
}

export class HelixClient {
  readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly auth: Credentials | null;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? 120_000;
    this.auth = options.auth ?? null;
  }

  /** Auth headers are attached server-side only; they are never rendered or logged. */
  private authHeaders(): Record<string, string> {
    if (!this.auth || !this.auth.username) return {};
    return { authorization: basicAuthHeader(this.auth.username, this.auth.password) };
  }

  private url(path: string): string {
    return `${this.baseUrl}${path.startsWith('/') ? path : `/${path}`}`;
  }

  async getJson<T>(path: string, timeoutMs = 15_000): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(this.url(path), {
        method: 'GET',
        headers: { accept: 'application/json', ...this.authHeaders() },
        signal: controller.signal,
      });
      const text = await response.text();
      if (!response.ok) throw errorFromResponse(response.status, text, response.headers.get('x-request-id'));
      try {
        return JSON.parse(text) as T;
      } catch {
        throw new ApiError('malformed', 'The server returned a non-JSON response.', response.status);
      }
    } catch (error) {
      throw this.wrap(error);
    } finally {
      clearTimeout(timer);
    }
  }

  /** Multipart upload to the local vision tool. Returns the raw JSON payload. */
  async analyzeImage(asset: { uri: string; name?: string; type?: string }): Promise<Record<string, unknown>> {
    const form = new FormData();
    form.append('image', {
      uri: asset.uri,
      name: asset.name ?? 'upload.jpg',
      type: asset.type ?? 'image/jpeg',
    } as unknown as Blob);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60_000);
    try {
      const response = await fetch(this.url('/api/image/analyze'), {
        method: 'POST',
        body: form,
        headers: { accept: 'application/json', ...this.authHeaders() },
        signal: controller.signal,
      });
      const text = await response.text();
      if (!response.ok) throw errorFromResponse(response.status, text, response.headers.get('x-request-id'));
      try {
        return JSON.parse(text) as Record<string, unknown>;
      } catch {
        throw new ApiError('malformed', 'The image tool returned a non-JSON response.', response.status);
      }
    } catch (error) {
      throw this.wrap(error);
    } finally {
      clearTimeout(timer);
    }
  }

  health() {
    return this.getJson<Record<string, unknown>>('/api/health', 10_000);
  }

  async chat(options: StreamOptions): Promise<string> {
    const body = JSON.stringify({
      message: options.message,
      history: (options.history ?? []).slice(-8),
      max_new_tokens: options.maxNewTokens ?? 160,
      temperature: options.temperature ?? 0.7,
    });
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(this.url('/api/chat'), {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json', ...this.authHeaders() },
        body,
        signal: controller.signal,
      });
      const text = await response.text();
      if (!response.ok) throw errorFromResponse(response.status, text, response.headers.get('x-request-id'));
      const parsed = JSON.parse(text) as { reply?: string; response?: string; error?: string };
      const reply = parsed.reply ?? parsed.response ?? '';
      if (!reply) throw new ApiError('malformed', 'The server returned an empty reply.', response.status);
      return reply;
    } catch (error) {
      throw this.wrap(error);
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Stream a reply. Resolves when the stream finishes, rejects with ApiError.
   * If incremental transport fails before any token arrives, it retries once
   * over the non-streaming endpoint so the user still gets an answer.
   */
  stream(options: StreamOptions, handlers: StreamHandlers): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const cancelState = options.signal;
      let pending = '';
      let consumed = 0;
      let full = '';
      let settled = false;
      let gotData = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        xhr.abort();
        reject(new ApiError('timeout', MESSAGES.timeout));
      }, this.timeoutMs);

      const finish = (fn: () => void) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        fn();
      };

      const consume = (chunk: string) => {
        if (settled) return;
        const { events, rest } = parseSseChunk(pending + chunk);
        pending = rest;
        for (const event of events) {
          if (event.error) {
            finish(() => {
              const error = new ApiError('server', event.error!, 200, event.request_id ?? null);
              handlers.onError(error);
              reject(error);
            });
            return;
          }
          if (event.meta) {
            gotData = true;
            handlers.onMeta?.(event.meta as ChatMeta);
          }
          if (typeof event.delta === 'string' && event.delta) {
            gotData = true;
            full += event.delta;
            handlers.onDelta(event.delta);
          }
          if (event.done) {
            finish(() => {
              handlers.onDone(full);
              resolve(full);
            });
            return;
          }
        }
      };

      xhr.open('POST', this.url('/api/chat/stream'));
      xhr.setRequestHeader('content-type', 'application/json');
      xhr.setRequestHeader('accept', 'text/event-stream');
      const auth = this.authHeaders();
      if (auth.authorization) xhr.setRequestHeader('authorization', auth.authorization);
      xhr.timeout = this.timeoutMs;

      const pull = () => {
        const text = xhr.responseText ?? '';
        if (text.length <= consumed) return;
        const chunk = text.slice(consumed);
        consumed = text.length;
        consume(chunk);
      };

      xhr.onprogress = pull;

      xhr.onerror = () => {
        if (cancelState?.cancelled) {
          finish(() => reject(new ApiError('cancelled', MESSAGES.cancelled)));
          return;
        }
        this.fallback(options, handlers, full)
          .then((reply) => finish(() => resolve(reply)))
          .catch((error) => finish(() => reject(error)));
      };

      xhr.onabort = () => {
        finish(() => reject(new ApiError('cancelled', MESSAGES.cancelled)));
      };

      xhr.ontimeout = () => {
        finish(() => reject(new ApiError('timeout', MESSAGES.timeout)));
      };

      xhr.onloadend = () => {
        if (settled) return;
        if (xhr.status >= 200 && xhr.status < 300) {
          pull();
          if (settled) return;
          if (full) {
            finish(() => {
              handlers.onDone(full);
              resolve(full);
            });
          } else {
            this.fallback(options, handlers, '')
              .then((reply) => finish(() => resolve(reply)))
              .catch((error) => finish(() => reject(error)));
          }
          return;
        }
        const error = errorFromResponse(xhr.status, xhr.responseText, xhr.getResponseHeader('x-request-id'));
        finish(() => {
          handlers.onError(error);
          reject(error);
        });
      };

      if (cancelState) {
        const poll = setInterval(() => {
          if (cancelState.cancelled && !settled) {
            clearInterval(poll);
            xhr.abort();
          }
        }, 120);
        const stopPolling = () => clearInterval(poll);
        void stopPolling;
        setTimeout(stopPolling, this.timeoutMs + 50);
      }

      try {
        xhr.send(
          JSON.stringify({
            message: options.message,
            history: (options.history ?? []).slice(-8),
            max_new_tokens: options.maxNewTokens ?? 160,
            temperature: options.temperature ?? 0.7,
          }),
        );
      } catch {
        void gotData;
        this.fallback(options, handlers, '')
          .then((reply) => finish(() => resolve(reply)))
          .catch((err) => finish(() => reject(this.wrap(err))));
      }
    });
  }

  private async fallback(options: StreamOptions, handlers: StreamHandlers, partial: string): Promise<string> {
    const reply = await this.chat(options);
    if (partial) {
      handlers.onDelta(reply.slice(partial.length));
    } else {
      handlers.onDelta(reply);
    }
    handlers.onDone(reply);
    return reply;
  }

  private wrap(error: unknown): ApiError {
    if (error instanceof ApiError) return error;
    const name = (error as { name?: string } | null)?.name;
    if (name === 'AbortError') return new ApiError('timeout', MESSAGES.timeout);
    if (name === 'TypeError') return new ApiError('offline', MESSAGES.offline);
    return new ApiError('offline', MESSAGES.offline);
  }
}
