/** Presentation helpers with no React Native dependency. */

export function formatRelativeTime(timestamp: number, now = Date.now()): string {
  const seconds = Math.max(0, Math.round((now - timestamp) / 1000));
  if (seconds < 60) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? 'yesterday' : `${days}d ago`;
}

export function truncate(text: string, max: number): string {
  const clean = String(text || '').replace(/\s+/g, ' ').trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return 'unknown';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function summarizeHealth(health: Record<string, unknown> | null): {
  label: string;
  tone: 'ready' | 'degraded' | 'down';
} {
  if (!health) return { label: 'Offline', tone: 'down' };
  if (health.model_ready === true) return { label: 'Ready', tone: 'ready' };
  if (typeof health.degraded_reason === 'string') return { label: 'Degraded', tone: 'degraded' };
  return { label: 'Unknown', tone: 'degraded' };
}
